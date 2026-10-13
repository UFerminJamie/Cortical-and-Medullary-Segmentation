from __future__ import annotations

import math
import os
import shlex
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .catalog import Catalog
from .config import PROJECT, Config, allocated_gpu
from .snapshots import create_snapshot, file_stamp
from .storage import Store, atomic_json, now, read_json
from .training import selection_snapshot, validate_selection


class Jobs:
    def __init__(self, config: Config, store: Store, catalog: Catalog):
        self.config, self.store, self.catalog = config, store, catalog
        self._threads = {}
        self._processes = {}
        self._last_recovery = {}

    def submit(self, kind: str, *, slide_id: str | None = None, model_id: str = "baseline",
               threshold: float = -1.0, budget: int = 100_000, reviewer: str = "", pass_id: str = "pass-1",
               executor: str | None = None, slide_ids: list[str] | None = None,
               training_samples=None, validation_samples=None, model_name="", c_values=None) -> dict:
        if not self.config.enable_compute:
            raise ValueError("GPU prediction and training are disabled in this deployment")
        selected = kind == "train" and (training_samples is not None or validation_samples is not None)
        if selected:
            training_samples, validation_samples, model_name, c_values = validate_selection(
                self.catalog, training_samples, validation_samples, model_name, c_values)
            executor = "slurm"
        if kind == "train" and not selected and not Path(self.config.original_manifest).is_file():
            raise ValueError("Training requires the original split workbook; configure HISTOLOOP_ORIGINAL_MANIFEST")
        executor = executor or self.config.executor
        if executor not in ("local", "slurm"):
            raise ValueError("Choose the session GPU or a Slurm GPU job")
        if executor == "local" and (self.config.device != "gpu" or not allocated_gpu()):
            raise ValueError("A GPU compute allocation is required to use the session GPU. Choose a Slurm GPU job or start a GPU viewer allocation.")
        if kind not in ("predict", "train") or not math.isfinite(threshold):
            raise ValueError("Invalid job or threshold")
        if budget < 0 or budget > 1_000_000:
            raise ValueError("Additional tile budget must be 0–1,000,000 (0 means all)")
        if pass_id not in {item["id"] for item in self.store.passes()}:
            raise ValueError("Unknown review pass")
        model = self.store.model(model_id, self.config.baseline_model)
        if kind == "predict":
            ids = slide_ids if slide_ids is not None else [slide_id]
            if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
                raise ValueError("Select at least one slide to predict")
            ids = list(dict.fromkeys(ids))
            slides = [self.catalog.slide(value) for value in ids]
            slide_id = ids[0] if len(ids) == 1 else None
            slide = slides[0] if len(ids) == 1 else None
        else:
            ids, slides, slide = [], [], None
        with self.store.lock():
            for active in self.listing(refresh=False):
                if active["status"] in ("preparing", "submitting", "pending", "running", "submitted", "unknown"):
                    if executor == "local" and active["executor"] == "local":
                        raise ValueError("The session's GPU has an active review job. Wait for it or use the Slurm executor.")
                    active_ids = active.get("slide_ids") or [active.get("slide_id")]
                    if active["kind"] == kind and (kind == "train" or set(active_ids).intersection(ids)):
                        raise ValueError("A job of this type is already active for this slide or training set")
            job_id = f"{kind}-{uuid.uuid4().hex[:16]}"
            directory = self.store.root / "jobs" / job_id
            directory.mkdir(parents=True, exist_ok=True)
            job = {"id": job_id, "kind": kind, "status": "preparing", "created_at": now(),
                "slide_id": slide_id, "slide_ids": ids,
                "slide_name": slide["name"] if slide else (f"{len(ids)} slides" if ids else None),
                "total": len(ids), "completed": 0, "failed": 0,
                "model_id": model_id, "model_path": model["path"], "threshold": threshold,
                "budget": budget, "reviewer": reviewer, "pass_id": pass_id,
                "executor": executor, "directory": str(directory)}
            job["slurm_name"] = "rv_" + job_id.split("-", 1)[1]
            job["submitter_pid"] = os.getpid()
            job["model_stamp"] = None if selected else file_stamp(model["path"])
            job["parent_metadata"] = read_json(Path(model["path"]).parent / "metadata.json", {})
            if selected:
                job.update(training_mode="selected_samples", training_samples=training_samples,
                    validation_samples=validation_samples, model_name=model_name, c_values=c_values,
                    total=len(c_values), budget=0)
            atomic_json(directory / "job.json", job)
        try:
            if selected:
                thread = threading.Thread(target=self._prepare_selection, args=(dict(job),), daemon=True, name=f"launch-{job_id}")
                self._threads[job_id] = thread
                thread.start()
                return job
            elif kind == "train":
                snapshot = create_snapshot(self.catalog, self.store, directory / "snapshot")
                job["corrected_tiles"] = snapshot["corrected_tiles"]
                job["snapshot_path"] = str(directory / "snapshot/snapshot.json")
            else:
                # Freeze identities cheaply; validate/load geometry on the GPU
                # worker rather than reading hundreds of PT matrices in HTTP.
                job["slides"] = [{"id": value, "feature_stamp": file_stamp(self.catalog.slide(value)["pt_path"])} for value in ids]
            request = {"config": {**self.config.dict(), "executor": executor}, "job": job}
            atomic_json(directory / "request.json", request)
            job["status"] = "submitting"
            atomic_json(directory / "job.json", job)
            thread = threading.Thread(target=self._launch, args=(dict(job),), daemon=True, name=f"launch-{job_id}")
            self._threads[job_id] = thread
            thread.start()
            return job
        except Exception as error:
            job.update(status="failed", error=str(error), finished_at=now())
            atomic_json(directory / "job.json", job)
            raise

    def _prepare_selection(self, job):
        directory = Path(job["directory"])
        try:
            snapshot = selection_snapshot(self.catalog, self.store, directory / "snapshot", job)
            job.update(snapshot_path=str(directory / "snapshot/snapshot.json"),
                training_tiles=sum(row["tiles"] for row in snapshot["slides"] if row["split"] == "training"),
                validation_tiles=sum(row["tiles"] for row in snapshot["slides"] if row["split"] == "validation"))
            atomic_json(directory / "request.json", {"config": {**self.config.dict(), "executor": "slurm"}, "job": job})
            job["status"] = "submitting"
            self._publish_launch(job)
            self._launch(job)
        except Exception as error:
            job.update(status="failed", error=str(error), finished_at=now())
            self._publish_launch(job)

    def _recover_id(self, job: dict) -> str | None:
        name = job.get("slurm_name")
        if not name:
            return None
        commands = [["squeue", f"--name={name}", "--noheader", "--format=%A"],
            ["sacct", f"--name={name}", f"--starttime={job['created_at'][:10]}", "--format=JobIDRaw", "--noheader", "--parsable2"]]
        for command in commands:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5)
                ids = {line.strip().strip("|") for line in result.stdout.splitlines() if line.strip().strip("|").isdigit()}
                if len(ids) == 1:
                    return ids.pop()
            except (OSError, subprocess.TimeoutExpired):
                continue
        return None

    def _publish_launch(self, job: dict):
        path = Path(job["directory"]) / "job.json"
        with self.store.lock():
            current = read_json(path, job)
            # A short worker can finish before sbatch's response reaches this thread.
            if current["status"] in ("running", "completed", "completed_with_errors", "failed"):
                for key, value in current.items():
                    if key not in ("slurm_id", "pid"):
                        job[key] = value
            atomic_json(path, job)

    def _launch(self, job: dict):
        directory = Path(job["directory"])
        command = [self.config.python, "-m", "SVC_GPU.human_loop.worker", "--request", str(directory / "request.json")]
        try:
            stdout = directory / "stdout.log"
            stderr = directory / "stderr.log"
            if job["executor"] == "slurm":
                script = directory / "run.sh"
                limits = '\n'.join(f'export {key}="${{SLURM_CPUS_PER_TASK:-{self.config.job_cpus}}}"' for key in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
                script.write_text("#!/bin/bash\nset -euo pipefail\n" + limits + "\ncd " + shlex.quote(str(PROJECT)) + "\nexec " + shlex.join(command) + "\n")
                arguments = ["sbatch", "--parsable", f"--account={self.config.account}", f"--qos={self.config.qos}",
                    f"--partition={self.config.gpu_partition}", f"--gres={self.config.gpu_resource}",
                    f"--cpus-per-task={self.config.job_cpus}", f"--mem={self.config.job_memory}",
                    f"--time={self.config.job_time}", f"--job-name={job['slurm_name']}",
                    f"--output={stdout}", f"--error={stderr}", str(script)]
                result = subprocess.run(arguments, capture_output=True, text=True, timeout=60, check=True)
                slurm_id = result.stdout.strip().split(";")[0]
                if not slurm_id.isdigit():
                    raise RuntimeError(f"Slurm returned an unexpected job ID: {slurm_id}")
                job.update(status="pending", slurm_id=slurm_id)
            elif job["executor"] == "local":
                # The child inherits the allocation and CUDA device mask. CUDA is
                # initialized in the worker, keeping GPU libraries out of the UI.
                environment = os.environ.copy()
                affinity = len(os.sched_getaffinity(0))
                cpus = str(max(1, min(affinity, int(environment.get("SLURM_CPUS_PER_TASK") or affinity))))
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = cpus
                with stdout.open("a") as out, stderr.open("a") as err:
                    process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=out, stderr=err, start_new_session=True)
                self._processes[job["id"]] = process
                job.update(status="submitted", pid=process.pid, worker_node=socket.gethostname())
            else:
                raise ValueError("Job executor must be slurm or local")
            self._publish_launch(job)
        except subprocess.TimeoutExpired:
            slurm_id = self._recover_id(job)
            if slurm_id:
                job.update(status="pending", slurm_id=slurm_id)
            else:
                job.update(status="unknown", error="Slurm submission response timed out. Checking by unique job name; duplicate submissions are blocked.")
            self._publish_launch(job)
        except Exception as error:
            message = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
            job.update(status="failed", error=message, finished_at=now())
            self._publish_launch(job)

    def listing(self, refresh: bool = True) -> list[dict]:
        jobs = [read_json(path) for path in sorted((self.store.root / "jobs").glob("*/job.json"))]
        if refresh:
            for job in jobs:
                thread = self._threads.get(job["id"])
                if job["executor"] == "slurm" and not job.get("slurm_id") and job["status"] in ("submitting", "unknown") and not (thread and thread.is_alive()):
                    if time.monotonic() - self._last_recovery.get(job["id"], 0) >= 30:
                        self._last_recovery[job["id"]] = time.monotonic()
                        slurm_id = self._recover_id(job)
                        if slurm_id:
                            job.update(status="pending", slurm_id=slurm_id)
                            self._publish_launch(job)
                        elif job["status"] == "submitting":
                            job.update(status="unknown", error="Server restarted during submission. Checking Slurm by unique job name.")
                            self._publish_launch(job)
        active = [job for job in jobs if job["executor"] == "slurm" and job.get("slurm_id") and job["status"] in ("pending", "running", "submitted", "unknown")]
        if refresh and active:
            try:
                response = subprocess.run(["sacct", "-j", ",".join(job["slurm_id"] for job in active),
                    "--format=JobIDRaw,State", "--parsable2", "--noheader"], capture_output=True, text=True, timeout=5)
                states = dict(line.split("|", 1) for line in response.stdout.splitlines() if "|" in line and "." not in line.split("|", 1)[0])
                for job in active:
                    state = states.get(job["slurm_id"], "").split()[0].rstrip("+") if states.get(job["slurm_id"]) else ""
                    if state == "RUNNING":
                        job["status"] = "running"
                    elif state in ("PENDING", "CONFIGURING"):
                        job["status"] = "pending"
                    elif state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"):
                        job.update(status="failed", error=f"Slurm state: {state}")
                    elif state == "COMPLETED":
                        current = read_json(Path(job["directory"]) / "job.json")
                        if current["status"] not in ("completed", "completed_with_errors"):
                            job.update(status="failed", error="Worker exited without publishing a result; inspect its logs")
                        else:
                            job.update(current)
                    with self.store.lock():
                        current = read_json(Path(job["directory"]) / "job.json")
                        if current["status"] not in ("completed", "completed_with_errors", "failed"):
                            if current["status"] == "running" and job["status"] == "pending":
                                job["status"] = "running"
                            current["status"] = job["status"]
                            if job["status"] == "failed" and job.get("error"):
                                current.update(error=job["error"], finished_at=now())
                            # Scheduler polling owns status, not worker progress,
                            # outputs or resource metadata written meanwhile.
                            atomic_json(Path(job["directory"]) / "job.json", current)
                        job.update(current)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if refresh:
            for job in jobs:
                if job["executor"] != "local":
                    continue
                process = self._processes.get(job["id"])
                owner = job.get("worker_node") or job.get("resources", {}).get("hostname")
                # PIDs are node-local. Other viewers share job records, but
                # must never probe a GPU worker's PID on their own CPU node.
                if process is None and owner != socket.gethostname():
                    continue
                exit_code = process.poll() if process is not None else None
                if exit_code is not None:
                    self._processes.pop(job["id"], None)
                    if job["status"] not in ("completed", "completed_with_errors", "failed"):
                        with self.store.lock():
                            current = read_json(Path(job["directory"]) / "job.json")
                            if current["status"] not in ("completed", "completed_with_errors", "failed"):
                                current.update(status="failed", error=f"Worker exited with code {exit_code} without publishing a result; inspect its logs", finished_at=now())
                                atomic_json(Path(job["directory"]) / "job.json", current)
                            job.update(current)
                    continue
                if job.get("pid") and job["status"] in ("submitted", "running"):
                    try:
                        os.kill(job["pid"], 0)
                    except ProcessLookupError:
                        with self.store.lock():
                            path = Path(job["directory"]) / "job.json"
                            current = read_json(path, job)
                            if current["status"] in ("submitted", "running"):
                                current.update(status="failed", error="Worker stopped without publishing a result", finished_at=now())
                                atomic_json(path, current)
                            job.update(current)
                    except PermissionError:
                        # An uninspectable PID is not evidence of an exited worker.
                        continue
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def logs(self, job_id: str) -> dict:
        job = next((item for item in self.listing(refresh=False) if item["id"] == job_id), None)
        if not job:
            raise KeyError("Unknown job")
        result = {}
        for key in ("stdout", "stderr"):
            path = Path(job["directory"]) / f"{key}.log"
            if path.exists():
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - 16000))
                    result[key] = handle.read().decode(errors="replace")
            else:
                result[key] = ""
        return result
