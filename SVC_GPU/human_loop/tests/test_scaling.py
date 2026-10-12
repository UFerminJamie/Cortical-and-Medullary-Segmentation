import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import correction
from SVC_GPU.human_loop.catalog import Catalog
from SVC_GPU.human_loop.grid import grid_info, grid_mask
from SVC_GPU.human_loop.storage import atomic_json, read_json
from SVC_GPU.human_loop.worker import predict_group
from SVC_GPU.human_loop.wsi import WsiPool


def overlap_sample(config, app, slide):
    path = Path(slide['pt_path'])
    payload = torch.load(path, weights_only=False)
    coords = np.array([(x, y) for y in range(0, 257, 64) for x in range(0, 257, 64)])
    payload['coords'] = torch.from_numpy(coords)
    payload['features'] = torch.randn(len(coords), 16)
    torch.save(payload, path)
    pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'decision_score': np.linspace(-2, 2, len(coords)),
                  'predicted_label_encoded': np.arange(len(coords)) % 2}).to_csv(slide['baseline_csv'], index=False)
    return coords


def test_grid_subsets_preserve_original_ids_and_phase():
    coords = np.array([(x, y) for y in range(13, 1038, 128) for x in range(21, 1046, 128)])
    info = grid_info(coords, 512)
    assert info['native_overlap'] == 75 and info['origin'] == [21, 13]
    assert grid_mask(coords, 512, '0').sum() == 9
    assert grid_mask(coords, 512, '50').sum() == 25
    assert grid_mask(coords, 512, '75').all()
    with pytest.raises(ValueError, match='coordinate lattice'):
        grid_mask(coords, 512, '0', '50')
    native0 = np.array([[0, 0], [512, 0]])
    with pytest.raises(ValueError, match='not available'):
        grid_mask(native0, 512, '50')


@pytest.mark.parametrize('size,stride,multiplier',[(507,127,4),(509,127,5),(507,126,5),(507,254,2)])
def test_rounded_trident_geometry_has_strict_no_overlap(size,stride,multiplier):
    coords=np.array([(x,y) for y in range(11,11+stride*12,stride) for x in range(19,19+stride*12,stride)])
    info=grid_info(coords,size)
    assert info['native_overlap'] in (50,75)
    selected=coords[grid_mask(coords,size,'0')]
    for axis in range(2):
        gaps=np.diff(np.unique(selected[:,axis]))
        assert (gaps>=size).all() and (gaps==stride*multiplier).all()
    assert grid_mask(coords,size,str(info['native_overlap'])).all()


def test_old_grid_clients_are_rejected_before_saving(setup):
    config,app,client,slides=setup
    response=correction(client,slides['A']['id'],grid_mode='0',grid_format=1)
    assert response.status_code==409 and not app.state.store.events(slides['A']['id'])


def test_region_bulk_and_undo_respect_grid_even_without_filters(setup, monkeypatch):
    config, app, client, slides = setup
    slide = slides['A'];coords = overlap_sample(config, app, slide)
    detail = client.get('/api/slides/' + slide['id']).json()
    assert detail['grid']['counts'] == {'native': 25, '0': 4, '50': 9, '75': 25}
    polygon = [[0, 0], [512, 0], [512, 512], [0, 512]]
    result = correction(client, slide['id'], polygon=polygon, grid_mode='0', visible_only=False).json()
    expected = np.flatnonzero(grid_mask(coords, 256, '0')).tolist()
    assert result['tile_ids'] == expected
    labels = app.state.store.resolved(slide['id'], 25)[0]
    assert np.flatnonzero(labels != -1).tolist() == expected
    event = app.state.store.events(slide['id'])[-1]
    assert event['grid_mode'] == '0' and event['grid_origin'] == [0, 0]
    plan = client.post('/api/bulk-labels/plan', json={'slides': [{'id': slide['id']}], 'grid_mode': '50'}).json()
    assert plan['tiles'] == 9 and plan['slides'][0]['revision'] == 1
    monkeypatch.setattr('SVC_GPU.human_loop.bulk.threading.Thread.start', lambda self: None)
    batch = client.post('/api/bulk-labels', json={'reviewer': 'Test', 'pass_id': 'pass-1',
        'slides': plan['slides'], 'label': 0}).json()
    app.state.bulk.run(read_json(app.state.store.root / 'bulk' / batch['id'] / 'request.json'))
    status = app.state.bulk.listing()[0]
    assert status['tiles'] == 9
    event = app.state.store.events(slide['id'])[-1]
    assert event['grid_mode'] == '50'
    assert client.post('/api/slides/' + slide['id'] + '/undo', json={'reviewer': 'Test', 'pass_id': 'pass-1',
        'revision': 2, 'target_id': event['id']}).status_code == 200
    assert np.array_equal(app.state.store.resolved(slide['id'], 25)[0], labels)


def test_view_alignment_cache_and_jpeg_browser_caching(setup, monkeypatch):
    config, app, client, slides = setup
    slide = slides['A'];sid = slide['id']
    correction(client, sid)
    response = client.get('/api/slides/' + sid + '/view')
    length = int.from_bytes(response.content[:4], 'little')
    detail = json.loads(response.content[4:4+length])
    assert (4 + length) % 4 == 0 and detail['revision'] == 1
    array = np.frombuffer(response.content, '<f4', offset=4+length).reshape(-1, 5)
    assert array[:, 4].tolist() == [1, -1, -1, -1]
    jpeg = client.get(f"/api/wsi/{sid}_files/{detail['max_level']}/0_0.jpg")
    assert jpeg.headers['cache-control'].startswith('private')
    assert client.get('/api/slides').headers['content-encoding'] == 'gzip'
    app.state.catalog.tiles(sid)
    fresh = Catalog(config, app.state.store)
    monkeypatch.setattr('SVC_GPU.human_loop.catalog.pd.read_csv', lambda *args, **kwargs: pytest.fail('Validated arrays should be loaded from the node cache'))
    assert fresh.tiles(sid)['fingerprint'] == detail['fingerprint']
    fresh.read_cache.close()


def test_wsi_singleflight_decodes_once_and_does_not_block_other_slides(monkeypatch):
    pool = WsiPool();entered = threading.Event();release = threading.Event();calls = []
    def decode(path, *args):
        calls.append(path)
        if path == 'slow':
            entered.set();assert release.wait(3)
        return path.encode()
    monkeypatch.setattr(pool, '_decode', decode)
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(pool.tile, 'slow', 1, 0, 0)
        assert entered.wait(3)
        second = executor.submit(pool.tile, 'slow', 1, 0, 0)
        fast = executor.submit(pool.tile, 'fast', 1, 0, 0)
        assert fast.result(timeout=1) == b'fast'
        release.set();assert first.result() == second.result() == b'slow'
    assert calls.count('slow') == 1


def test_group_job_freezes_sources_and_blocks_overlapping_submissions(setup, monkeypatch):
    config, app, client, slides = setup
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.threading.Thread.start', lambda self: None)
    ids = [slides[key]['id'] for key in 'ABC']
    result = client.post('/api/jobs', json={'kind': 'predict', 'slide_ids': ids, 'reviewer': 'Test', 'executor': 'slurm'})
    assert result.status_code == 202, result.text
    job = result.json()
    assert job['total'] == 3 and job['slide_id'] is None
    request = read_json(Path(job['directory']) / 'request.json')
    assert [entry['id'] for entry in request['job']['slides']] == ids
    assert all(entry['feature_stamp']['size'] > 0 for entry in request['job']['slides'])
    duplicate = client.post('/api/jobs', json={'kind': 'predict', 'slide_id': ids[1], 'reviewer': 'Test', 'executor': 'slurm'})
    assert duplicate.status_code == 400


def test_group_partial_failure_publishes_successes_and_preserves_human_labels(setup, monkeypatch):
    config, app, client, slides = setup
    ids = [slides[key]['id'] for key in 'AB']
    correction(client, ids[0]);before = app.state.store.events(ids[0])
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.threading.Thread.start', lambda self: None)
    job = app.state.jobs.submit('predict', slide_ids=ids, reviewer='Test')
    monkeypatch.setattr('SVC_GPU.human_loop.worker.load_bundle', lambda path: {'metadata': {'feature_dimension': 16}})
    call = [0]
    def prediction(bundle, features, **kwargs):
        call[0] += 1
        if call[0] == 2:
            raise ValueError('Synthetic sample failure')
        return np.ones(len(features), np.int8), np.ones(len(features), np.float32)
    monkeypatch.setattr('SVC_GPU.human_loop.worker.predict_gpu', prediction)
    result = predict_group(config, app.state.store, app.state.catalog, job)
    assert result['completed'] == 1 and result['failed'] == 1
    assert app.state.catalog.current_tiles(ids[0])['prediction_id'] == job['id']
    assert app.state.catalog.current_tiles(ids[1])['prediction_id'] == 'baseline'
    assert app.state.store.events(ids[0]) == before
    assert (Path(job['directory']) / 'outputs' / ids[0] / 'predictions' / (slides['A']['name'] + '.csv')).exists()


def test_first_prediction_can_run_without_baseline_csv(setup, monkeypatch):
    config, app, client, slides = setup
    slide = slides['C'];Path(slide['baseline_csv']).unlink();app.state.catalog.refresh_datasets()
    assert client.get('/api/slides/' + slide['id']).status_code == 404
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.threading.Thread.start', lambda self: None)
    job = app.state.jobs.submit('predict', slide_ids=[slide['id']], reviewer='Test')
    monkeypatch.setattr('SVC_GPU.human_loop.worker.load_bundle', lambda path: {'metadata': {'feature_dimension': 16}})
    monkeypatch.setattr('SVC_GPU.human_loop.worker.predict_gpu', lambda bundle, features, **kwargs:
        (np.zeros(len(features), np.int8), -np.ones(len(features), np.float32)))
    assert predict_group(config, app.state.store, app.state.catalog, job)['tiles'] == 4
    assert client.get('/api/slides/' + slide['id']).json()['prediction_id'] == job['id']
    row = next(row for row in client.get('/api/slides').json() if row['id'] == slide['id'])
    assert row['prediction_available']


def test_submitted_sources_reject_changed_features_without_publishing(setup, monkeypatch):
    config, app, client, slides = setup
    slide=slides['A']
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.threading.Thread.start',lambda self:None)
    job=app.state.jobs.submit('predict',slide_ids=[slide['id']],reviewer='Test')
    path=Path(slide['pt_path']);path.touch()
    monkeypatch.setattr('SVC_GPU.human_loop.worker.load_bundle',lambda path:{'metadata':{'feature_dimension':16}})
    monkeypatch.setattr('SVC_GPU.human_loop.worker.predict_gpu',lambda *args,**kwargs:pytest.fail('Changed features must never reach the predictor'))
    with pytest.raises(ValueError,match='All selected slides failed'):
        predict_group(config,app.state.store,app.state.catalog,job)
    assert not (app.state.store.root/'predictions'/slide['id']/'active.json').exists()


@pytest.mark.parametrize('owner',['gpu-node',None])
def test_other_node_or_legacy_workers_are_not_reaped_by_cpu_viewer(setup,monkeypatch,owner):
    config,app,client,slides=setup
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.socket.gethostname',lambda:'cpu-node')
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.os.kill',lambda *args:pytest.fail('Never inspect a remote/unknown worker PID'))
    directory=app.state.store.root/'jobs/remote-worker'
    job={'id':'remote-worker','directory':str(directory),'executor':'local','status':'running',
         'pid':987654321,'worker_node':owner,'created_at':'2026-10-07T00:00:00Z'}
    atomic_json(directory/'job.json',job)
    assert app.state.jobs.listing()[0]['status']=='running'
    assert read_json(directory/'job.json')==job


def test_local_pid_reaper_preserves_racing_completion(setup,monkeypatch):
    config,app,client,slides=setup
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.socket.gethostname',lambda:'gpu-node')
    directory=app.state.store.root/'jobs/short-worker'
    job={'id':'short-worker','directory':str(directory),'executor':'local','status':'running',
         'pid':987654321,'worker_node':'gpu-node','created_at':'2026-10-07T00:00:00Z'}
    atomic_json(directory/'job.json',job)
    def finish_and_exit(*args):
        atomic_json(directory/'job.json',{**job,'status':'completed','result':{'tiles':4}})
        raise ProcessLookupError()
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.os.kill',finish_and_exit)
    assert app.state.jobs.listing()[0]['status']=='completed'
    assert read_json(directory/'job.json')['status']=='completed'


def test_scheduler_poll_preserves_worker_progress_written_during_accounting(setup,monkeypatch):
    from types import SimpleNamespace
    config,app,client,slides=setup
    directory=app.state.store.root/'jobs/slurm-progress'
    job={'id':'slurm-progress','directory':str(directory),'executor':'slurm','status':'pending',
         'slurm_id':'12345','completed':0,'total':3,'created_at':'2026-10-07T00:00:00Z'}
    atomic_json(directory/'job.json',job)
    def accounting(*args,**kwargs):
        atomic_json(directory/'job.json',{**job,'status':'running','completed':2,'results':[{'slide_id':'a'},{'slide_id':'b'}]})
        return SimpleNamespace(stdout='12345|RUNNING\n',returncode=0)
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.subprocess.run',accounting)
    result=app.state.jobs.listing()[0]
    assert result['status']=='running' and result['completed']==2 and len(result['results'])==2
