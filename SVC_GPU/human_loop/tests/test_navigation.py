"""Conditional navigation, cancellation, and resource bounds without real data writes."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from conftest import correction
from test_datasets import multi_config
from SVC_GPU.human_loop.cache import BoundedCache
from SVC_GPU.human_loop.server import create_app
from SVC_GPU.human_loop.storage import atomic_json
from SVC_GPU.human_loop.storage import Store, Conflict
from SVC_GPU.human_loop.wsi import WsiPool


def test_lru_enforces_bytes_count_and_replacement():
    cache = BoundedCache(2, 8)
    cache.put('a', b'1234', 4);cache.put('b', b'5678', 4)
    assert cache.get('a') == b'1234'
    cache.put('c', b'abcde', 5)
    assert cache.get('a') is None and cache.get('b') is None
    cache.put('c', b'1', 1);cache.put('huge', b'x' * 9, 9)
    assert cache.metrics()['bytes'] == 1
    cache.clear();assert cache.metrics()['entries'] == 0


def test_conditional_views_invalidate_after_edit_and_refresh(setup):
    config, app, client, slides = setup
    sid = slides['A']['id'];path = f'/api/slides/{sid}/view'
    first = client.get(path);etag = first.headers['etag']
    assert client.get(path, headers={'If-None-Match': etag}).status_code == 304
    before = client.get('/api/slides?compact=true');queue_tag = before.headers['etag']
    assert correction(client, sid).status_code == 200
    changed = client.get(path, headers={'If-None-Match': etag})
    assert changed.status_code == 200 and changed.headers['etag'] != etag
    length = int.from_bytes(changed.content[:4], 'little')
    assert json.loads(changed.content[4:4+length])['revision'] == 1
    assert client.get('/api/slides?compact=true', headers={'If-None-Match': queue_tag}).status_code == 200
    etag = changed.headers['etag']
    assert client.post('/api/datasets/refresh').status_code == 200
    assert client.get(path, headers={'If-None-Match': etag}).status_code == 200
    metrics = client.get('/api/performance').json()
    assert metrics['responses']['bytes'] <= config.response_cache_mib * 1024**2


def test_saved_predictions_reused_and_invalidated_on_replacement(setup, monkeypatch):
    config, app, client, slides = setup
    sid = slides['A']['id'];catalog = app.state.catalog;base = catalog.tiles(sid)
    path = app.state.store.root / 'predictions' / sid / 'saved.npz'
    path.parent.mkdir(parents=True)
    def write(score):
        np.savez(path, fingerprint=base['fingerprint'], scores=np.full(4, score, np.float32),
                 predictions=np.ones(4, np.int8))
    write(1)
    entry = {'path': str(path), 'model_id': 'baseline', 'job_id': 'saved', 'threshold': -1}
    atomic_json(path.parent / 'active.json', entry)
    view = client.get(f'/api/slides/{sid}/view');etag = view.headers['etag']
    with monkeypatch.context() as patch:
        patch.setattr('SVC_GPU.human_loop.catalog.np.load', lambda *a, **kw: pytest.fail('Warm prediction must not be reparsed'))
        assert catalog.current_tiles(sid)['scores'].tolist() == [1]*4
    write(2)
    updated = client.get(f'/api/slides/{sid}/view', headers={'If-None-Match': etag})
    assert updated.status_code == 200 and updated.headers['etag'] != etag
    assert catalog.current_tiles(sid)['scores'].tolist() == [2]*4


def test_concurrent_saves_commit_once(setup):
    config, app, client, slides = setup
    sid = slides['A']['id']
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda label: correction(client, sid, label=label, revision=0).status_code, [0, 1]))
    assert sorted(results) == [200, 409]
    assert len(app.state.store.events(sid)) == 1


def test_save_validates_events_even_with_stale_directory_stamp(setup):
    config, app, client, slides = setup
    sid = slides['A']['id'];store = app.state.store
    assert correction(client, sid).status_code == 200
    old = store.events(sid)
    # Another process publishes a new event; emulate an unchanged cached stat.
    other = Store(config.state_dir)
    other.append(sid, 4, 1, {'action': 'review', 'reviewer': 'Other', 'pass_id': 'pass-1'})
    directory = store.root / 'events' / sid
    store._event_cache[sid] = (directory.stat().st_mtime_ns, old)
    with pytest.raises(Conflict):
        store.append(sid, 4, 1, {'action': 'review', 'reviewer': 'Stale', 'pass_id': 'pass-1'})
    assert client.get('/api/slides/'+sid).json()['revision'] == 2
    assert correction(client, sid, revision=2).status_code == 200


def test_history_pages_preserve_order_and_legacy_response(setup):
    config, app, client, slides = setup
    sid=slides['A']['id']
    for index in range(4):assert correction(client,sid,label=index%2,revision=index).status_code==200
    path=f'/api/slides/{sid}/history'
    first=client.get(path,params={'limit':2}).json()
    assert [e['sequence'] for e in first['events']]==[4,3] and first['next_before']==3
    last=client.get(path,params={'limit':2,'before':3}).json()
    assert [e['sequence'] for e in last['events']]==[2,1] and last['next_before'] is None
    assert len(client.get(path).json()['events'])==4
    assert client.get(path,params={'limit':0}).status_code==400


def test_corrupted_wsi_is_reported_without_poisoning_reader_cache(setup):
    config,app,client,slides=setup
    slide=slides['A'];path=Path(slide['wsi_path']);original=path.read_bytes()
    path.write_bytes(b'corrupted WSI')
    assert client.get('/api/slides/'+slide['id']+'/view').status_code==400
    path.write_bytes(original)
    assert client.get('/api/slides/'+slide['id']+'/view').status_code==200


def test_wsi_eviction_closes_idle_readers_and_errors_can_retry(setup):
    config, app, client, slides = setup
    pool = WsiPool(capacity=1, cache_bytes=32)
    for row in slides.values():pool.info(row['wsi_path'])
    assert pool.metrics()['readers'] == 1 and len(pool._opens) == 64
    with pytest.raises(FileNotFoundError):pool.info('/missing/navigation-test.svs')
    pool.info(slides['A']['wsi_path']);pool.close()
    assert pool.metrics()['readers'] == 0 and pool.metrics()['jpeg_bytes'] == 0


def test_browser_latest_selection_save_guards_and_cache_validation(setup):
    sync_playwright = pytest.importorskip('playwright.sync_api').sync_playwright
    config, _, _, slides = setup
    app = create_app(multi_config(config), token='test-access')
    sock = socket.socket();sock.bind(('127.0.0.1', 0));port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True);thread.start()
    try:
        deadline = time.monotonic()+10
        while not server.started and time.monotonic()<deadline:time.sleep(.02)
        with sync_playwright() as manager:
            if not Path(manager.chromium.executable_path).is_file():pytest.skip('Chromium not installed')
            browser = manager.chromium.launch(headless=True,args=['--no-sandbox'])
            page = browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{port}/?token=test-access')
            page.wait_for_function('S.detail&&!S.loading')
            ids=[slides[key]['id'] for key in 'ABC']
            # Delay old requests in the browser, including their cancellation path.
            page.evaluate('''id=>{const original=window.fetch;window.fetch=async(path,init)=>{
              if(String(path).includes(id))await new Promise(r=>setTimeout(r,200));
              return original(path,init);};}''',ids[1])
            page.evaluate('''async ids=>{const old=loadSlide(ids[1]);await new Promise(r=>setTimeout(r,20));
              const latest=loadSlide(ids[2]);await Promise.all([old,latest]);}''',ids)
            assert page.evaluate('S.detail.id')==ids[2]
            assert page.evaluate('S.engine.pending.size')==0
            # Rapid dataset changes are accepted during loading; only newest data survives.
            page.evaluate('''async()=>{$('datasetSelect').value='uf';const old=switchDataset();
              $('datasetSelect').value='kpmp';const latest=switchDataset();await Promise.all([old,latest]);}''')
            assert page.evaluate('S.dataset')=='kpmp'
            assert page.evaluate('S.slides.every(r=>r.dataset_id===S.dataset)')
            assert page.evaluate('S.detail.dataset_id')=='kpmp'
            # A network failure leaves the region retryable and blocks navigation.
            page.route('**/corrections',lambda route:route.fulfill(status=503,json={'detail':'Test save outage'}))
            page.evaluate("async()=>{$('reviewer').value='Tester';setMode('rectangle');await saveRegion([[0,0],[256,0],[256,256],[0,256]]);}")
            assert page.evaluate('Boolean(S.pendingPolygon)')
            old_id=page.evaluate('S.detail.id')
            page.evaluate('(id)=>loadSlide(id)',ids[2]);assert page.evaluate('S.detail.id')==old_id
            page.evaluate("$('datasetSelect').value='uf';switchDataset()")
            assert page.evaluate('S.dataset')=='kpmp'
            page.unroute('**/corrections')
            page.evaluate('async()=>{await saveRegion(S.pendingPolygon);}')
            assert page.evaluate('S.pendingPolygon') is None, page.locator('#toasts').inner_text()
            page.evaluate('async ids=>{await loadSlide(ids[2]);await loadSlide(ids[0]);}',ids)
            assert page.evaluate('S.detail.revision')>=1
            assert page.evaluate('navigationCache.metrics().notModified')>0
            # Tiny limits force eviction while the mutable current view stays valid.
            bounds=page.evaluate('''async()=>{navigationCache.clear();navigationCache.maxBytes=1500;
              for(const row of S.slides)await navigationCache.read('/api/slides/'+row.id+'/view');
              return navigationCache.metrics();}''')
            assert bounds['bytes']<=1500
            page.evaluate('''async id=>{S.engine.worker.onerror({message:'Synthetic worker failure'});
              await loadSlide(id);}''',ids[2])
            assert page.evaluate('Boolean(S.data&&S.mask&&!S.loading)')
            assert page.evaluate('S.engine.worker') is None
            assert not errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10);sock.close()
