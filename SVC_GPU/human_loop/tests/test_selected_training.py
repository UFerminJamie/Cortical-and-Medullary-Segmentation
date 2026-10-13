"""Selection isolation, frozen labels, Slurm submission, and candidate metrics."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from SVC_GPU.human_loop.storage import read_json
from SVC_GPU.human_loop.training import selection_snapshot
from SVC_GPU.human_loop.worker import train_selected
from conftest import correction


def selection(slide, revision=0):
    return {"id": slide["id"], "prediction_id": "baseline", "revision": revision}


def payload(slides):
    return {"kind": "train", "model_name": "Kidney test", "c_values": [.1, 2], "threshold": 0,
        "reviewer": "Test", "training_samples": [selection(slides["A"])],
        "validation_samples": [selection(slides["B"])]}


@pytest.mark.parametrize("change", [
    {"validation_samples": []}, {"c_values": [0]}, {"c_values": [True]},
    {"model_name": " "}, {"c_values": "2"}, {"c_values": [-1, 2]},
])
def test_selection_rejects_invalid_request(setup, change):
    _, _, client, slides = setup
    response = client.post('/api/jobs', json={**payload(slides), **change})
    assert response.status_code == 400


def test_selection_prevents_validation_overlap(setup):
    _, _, client, slides = setup
    body = payload(slides)
    body['validation_samples'] = body['training_samples']
    response = client.post('/api/jobs', json=body)
    assert response.status_code == 400 and 'overlap' in response.json()['detail']


def test_selected_submission_uses_slurm_without_original_workbook(setup, monkeypatch):
    config, app, client, slides = setup
    config.original_manifest = str(Path(config.state_dir) / 'missing.xlsx')
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='12345\n', stderr='', returncode=0)
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.subprocess.run', submit)
    response = client.post('/api/jobs', json={**payload(slides), 'executor': 'local'})
    assert response.status_code == 202
    job = response.json()
    app.state.jobs._threads[job['id']].join(timeout=10)
    saved = read_json(Path(job['directory']) / 'job.json')
    assert saved['status'] == 'pending' and saved['executor'] == 'slurm'
    assert saved['training_tiles'] == 4 and saved['validation_tiles'] == 4
    assert calls[0][0] == 'sbatch' and '--gres=gpu:l4:1' in calls[0]
    frozen = read_json(Path(saved['snapshot_path']))
    assert [r['split'] for r in frozen['slides']] == ['training', 'validation']


def test_frozen_labels_use_predictions_overrides_and_exclusions(setup, tmp_path):
    _, app, client, slides = setup
    sid = slides['A']['id']
    correction(client, sid, label=1).raise_for_status()
    correction(client, sid, label=-2, polygon=[[256,256],[512,256],[512,512],[256,512]]).raise_for_status()
    body = payload(slides)
    body['training_samples'] = [selection(slides['A'], 2)]
    frozen = selection_snapshot(app.state.catalog, app.state.store, tmp_path / 'snapshot', body)
    row = frozen['slides'][0]
    with np.load(row['labels_path']) as saved:
        assert saved['tile_ids'].tolist() == [0, 1, 2]
        assert saved['labels'].tolist() == [1, 0, 1]
        assert saved['human_override'].tolist() == [True, False, False]


def test_snapshot_is_immutable_and_stale_selection_fails(setup, tmp_path):
    _, app, client, slides = setup
    body = payload(slides)
    frozen = selection_snapshot(app.state.catalog, app.state.store, tmp_path / 'snapshot', body)
    path = Path(frozen['slides'][0]['labels_path'])
    before = path.read_bytes()
    correction(client, slides['A']['id'], label=1).raise_for_status()
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='changed'):
        selection_snapshot(app.state.catalog, app.state.store, tmp_path / 'stale', body)


def test_picker_and_thumbnail_include_combined_labels(setup):
    _, _, client, slides = setup
    correction(client, slides['A']['id'], label=1).raise_for_status()
    rows = client.get('/api/training/samples').json()
    sample = next(s for s in rows if s['id'] == slides['A']['id'])
    assert sample['medulla_fraction'] == .75 and sample['revision'] == 1
    image = client.get(f"/api/training/samples/{sample['id']}/thumbnail?prediction_id=baseline&revision=1")
    assert image.status_code == 200 and image.content.startswith(b'\xff\xd8')
    assert client.get(f"/api/training/samples/{sample['id']}/thumbnail?prediction_id=baseline&revision=0").status_code == 409


def test_candidates_fit_only_training_and_publish_validation_metrics(setup, tmp_path, monkeypatch):
    config, app, _, slides = setup
    body = payload(slides)
    directory = tmp_path / 'search'
    snapshot = selection_snapshot(app.state.catalog, app.state.store, directory / 'snapshot', body)
    seen = []
    pool = SimpleNamespace(free_all_blocks=lambda: None)
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (64 * 1024**3, 64 * 1024**3))), get_default_memory_pool=lambda: pool)
    monkeypatch.setattr('SVC_GPU.human_loop.worker.require_gpu', lambda: (cp, None, None, None))
    def fit(x, y, **kwargs):
        seen.append((x.clone(), y.clone(), kwargs['c_value']))
        assert len(x) == 4 and y.tolist() == [0, 0, 1, 1]
        return {'gpu': {'name': 'test'}, 'candidate': kwargs['c_value']}
    monkeypatch.setattr('SVC_GPU.human_loop.worker.train_gpu_pipeline', fit)
    monkeypatch.setattr('SVC_GPU.human_loop.worker.predict_gpu', lambda bundle, x, **kw: (np.array([0, 1, 1, 1]), np.array([-2., .2, .5, 2.])))
    def save(bundle, path, metadata):
        path.write_bytes(b'test candidate');return path
    monkeypatch.setattr('SVC_GPU.human_loop.worker.save_bundle', save)
    monkeypatch.setattr('SVC_GPU.human_loop.worker.save_confusion_matrix', lambda metrics, path: path.write_bytes(b'test matrix'))
    job = {**body, 'id': 'train-selection', 'directory': str(directory), 'snapshot_path': str(directory / 'snapshot/snapshot.json')}
    result = train_selected(config, app.state.store, job)
    assert [c for _, _, c in seen] == [.1, 2]
    assert result['completed'] == 2 and result['failed'] == 0
    models = [m for m in app.state.store.models(config.baseline_model) if m['id'] != 'baseline']
    assert len(models) == 2
    for model in models:
        metrics = model['validation_metrics']
        assert metrics['accuracy'] == .75 and metrics['balanced_accuracy'] == .75
        assert metrics['confusion_matrix'] == [[1, 1], [0, 2]]
        assert metrics['per_class']['cortical']['sensitivity'] == .5
        assert metrics['per_class']['medullary']['specificity'] == .5
        assert [s['id'] for s in model['metadata']['validation_samples']] == [slides['B']['id']]
        assert Path(model['path']).with_name('validation_confusion_matrix.png').is_file()
    assert (directory / 'artifacts/training_tile_provenance.csv').is_file()
    assert len(snapshot['slides']) == 2
