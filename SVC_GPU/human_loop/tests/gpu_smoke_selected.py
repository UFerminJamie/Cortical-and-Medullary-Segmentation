"""Isolated, synthetic end-to-end selected-sample SVC search via real sbatch."""
from dataclasses import replace
from pathlib import Path
import json
import time
import uuid

from .conftest import setup
from SVC_GPU.human_loop.config import PACKAGE
from SVC_GPU.human_loop.jobs import Jobs
from SVC_GPU.human_loop.storage import atomic_json


def main():
    root = PACKAGE / 'test_artifacts' / ('selected_gpu_' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    fixture = setup.__wrapped__(root)
    config, app, _, slides = next(fixture)
    try:
        config = replace(config, executor='slurm', job_cpus=2, job_memory='8G', job_time='00:10:00',
            python=str(PACKAGE / '.venv/bin/python'), original_manifest=str(root / 'missing.xlsx'))
        atomic_json(Path(config.baseline_model).parent / 'metadata.json', {'feature_dimension': 16, 'n_components': 4})
        manager = Jobs(config, app.state.store, app.state.catalog)
        entry = lambda key: {'id': slides[key]['id'], 'prediction_id': 'baseline', 'revision': 0}
        job = manager.submit('train', model_name='Synthetic GPU search', c_values=[.1, 2.], threshold=0,
            reviewer='GPU verification', training_samples=[entry('A')], validation_samples=[entry('B')])
        print(json.dumps({'job_id': job['id'], 'report_root': str(root)}), flush=True)
        deadline = time.monotonic() + 900
        previous = None
        while time.monotonic() < deadline:
            current = next(j for j in manager.listing() if j['id'] == job['id'])
            state = (current['status'], current.get('slurm_id'), current.get('completed', 0))
            if state != previous:
                print(json.dumps({'status': state[0], 'slurm_id': state[1], 'completed': state[2]}), flush=True)
                previous = state
            if current['status'] in ('completed', 'completed_with_errors', 'failed'):
                if current['status'] != 'completed':
                    raise AssertionError({'job': current, 'logs': manager.logs(job['id'])})
                models = [app.state.store.model(mid, config.baseline_model) for mid in current['result']['model_ids']]
                assert len(models) == 2
                for model in models:
                    assert model['metadata']['gpu']['name']
                    assert model['metadata']['training_samples'][0]['id'] == slides['A']['id']
                    assert model['metadata']['validation_samples'][0]['id'] == slides['B']['id']
                    assert model['validation_metrics']['tiles'] == 4
                    assert Path(model['path']).with_name('validation_confusion_matrix.png').is_file()
                assert current['resources']['slurm_job_id'] == current['slurm_id']
                report = {'passed': True, 'job': current, 'models': models}
                atomic_json(root / 'validation.json', report)
                print(json.dumps({'passed': True, 'report': str(root / 'validation.json'), 'slurm_id': current['slurm_id']}), flush=True)
                return
            time.sleep(5)
        raise TimeoutError('GPU validation has not finished; inspect ' + job['directory'])
    finally:
        fixture.close()


if __name__ == '__main__':
    main()
