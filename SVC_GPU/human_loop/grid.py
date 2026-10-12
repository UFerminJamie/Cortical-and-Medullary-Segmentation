"""Deterministic original-tile subsets, including rounded WSI geometry."""
import math
import numpy as np

FACTORS = {"0": 1, "50": 2, "75": 4}


def grid_info(coordinates, tile_size):
    origin = coordinates.min(axis=0).astype(int)
    differences = []
    for axis in range(2):
        differences.extend(int(value) for value in np.diff(np.unique(coordinates[:, axis])) if value > 0)
    lattice = math.gcd(*differences) if differences else int(tile_size)
    source_steps = {}
    for overlap, factor in FACTORS.items():
        expected = tile_size / factor
        candidates = {math.floor(expected), math.ceil(expected)}
        if factor == 1:
            candidates.update((tile_size - 1, tile_size + 1))
        valid = [step for step in candidates if step > 0 and lattice % step == 0]
        if valid:
            source_steps[overlap] = min(valid, key=lambda step: (abs(step - expected), -step))
    native = next((mode for mode in FACTORS if mode in source_steps), None)
    step = source_steps[native] if native is not None else lattice
    return {"format": 2, "origin": origin.tolist(), "native_step": step,
            "native_overlap": int(native) if native is not None else None,
            "actual_overlap": round(100 * (1 - step / tile_size), 3),
            "tile_size": int(tile_size), "regular": native is not None, "source_steps": source_steps}


def _profile(info, source_overlap):
    if source_overlap == "auto":
        overlap = info["native_overlap"]
        if overlap is None:
            return {}
        mode = str(overlap)
    else:
        mode = str(source_overlap)
        if mode not in FACTORS:
            raise ValueError("Source overlap must be auto, 0, 50, or 75")
        if mode not in info["source_steps"]:
            raise ValueError("Source overlap does not match this sample's coordinate lattice")
    source_step = info["source_steps"][mode]
    source_factor = FACTORS[mode]
    steps = {}
    for target, factor in FACTORS.items():
        if factor > source_factor:
            continue
        multiplier = source_factor // factor
        if target == "0":
            multiplier = math.ceil(info["tile_size"] / source_step)
        steps[target] = source_step * multiplier
    return steps


def grid_mask(coordinates, tile_size, mode="native", source_overlap="auto", info=None):
    if info is None or info.get("format") != 2:
        info = grid_info(coordinates, tile_size)
    steps = _profile(info, source_overlap)
    if str(mode) == "native":
        return np.ones(len(coordinates), dtype=bool)
    if str(mode) not in FACTORS:
        raise ValueError("Grid must be native, 0, 50, or 75")
    if str(mode) not in steps:
        raise ValueError("That overlap grid is not available from these existing tiles")
    return np.all((coordinates - info["origin"]) % steps[str(mode)] == 0, axis=1)


def describe_grid(data):
    info = data.get("grid")
    if info is None or info.get("format") != 2:
        info = grid_info(data["coordinates"], data["tile_size"])
    if "profiles" in info:
        return info
    profiles = {}
    for source in ("auto", "0", "50", "75"):
        try:
            steps = _profile(info, source)
        except ValueError:
            continue
        counts = {"native": len(data["coordinates"])}
        for mode in steps:
            counts[mode] = int(grid_mask(data["coordinates"], data["tile_size"], mode, source, info).sum())
        profiles[source] = {"steps": steps, "counts": counts}
    result = {**info, "profiles": profiles, "counts": profiles["auto"]["counts"]}
    data["grid"] = result
    return result
