"""Browser coverage for sample selection, results, colors and lazy annotations."""
import os
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import uvicorn
from shapely.geometry import box, mapping

from SVC_GPU.evaluation import classification_metrics
from SVC_GPU.human_loop.storage import atomic_json


def test_training_workspace_and_viewer_controls(setup, monkeypatch):
    playwright = pytest.importorskip('playwright.sync_api').sync_playwright
    cfg, app, _, slides = setup
    sid = slides['A']['id']
    app.state.annotation_files.save(sid, 'cortical_gt.json', {'revision': None,
        'document': {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'id': 'one',
            'properties': {}, 'geometry': mapping(box(0, 0, 250, 250))}]}})
    monkeypatch.setattr('SVC_GPU.human_loop.jobs.subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='98765\n', stderr='', returncode=0))
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([-2., .2, .5, 2.]))
    rows = []
    for key in 'AB':
        rows.append({**slides[key], 'tiles': 4, 'prediction_id': 'baseline', 'revision': 0})
    atomic_json(app.state.store.root / 'models/candidate.json', {'id': 'candidate', 'name': 'Reviewed kidney · C=2',
        'path': cfg.baseline_model, 'validation_metrics': metrics, 'metadata': {'training_samples': [rows[0]],
            'validation_samples': [rows[1]], 'decision_threshold': 0, 'training_tiles': 4, 'validation_tiles': 4}})
    sock = socket.socket();sock.bind(('127.0.0.1', 0));port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True);thread.start()
    try:
        with playwright() as manager:
            executable = os.environ.get('HISTOLOOP_BROWSER', manager.chromium.executable_path)
            if not Path(executable).is_file():pytest.skip('Set HISTOLOOP_BROWSER to Chromium')
            browser = manager.chromium.launch(headless=True, executable_path=executable, args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1600, 'height': 1000});errors = [];page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{port}/?token=test-access')
            page.wait_for_function('S.detail&&!S.loading&&annotations.layers.length')
            page.evaluate('(id)=>loadSlide(id)', sid)
            page.wait_for_function('(id)=>S.detail?.id===id&&!S.loading&&annotations.layers.length', arg=sid)
            assert page.evaluate('annotations.loaded.size') == 0
            page.locator('#showAllAnnotations').click()
            page.wait_for_function('annotations.loaded.has("cortical_gt.json")')
            assert page.evaluate('annotations.pref("cortical_gt.json").visible')
            page.locator('#hideAllAnnotations').click()
            assert page.evaluate('annotations.layers.every(l=>!annotations.pref(l.id).visible&&!annotations.pref(l.id).expanded)')
            # Worker palette changes alter pixels without touching the label field.
            assert page.evaluate("async()=>{$('predictionView').value='aggregate-score';await updateMasks();const labels=Array.from(S.fieldInfo.labels),old=S.raster;const c=document.createElement('canvas');c.width=old.width;c.height=old.height;const ctx=c.getContext('2d');ctx.drawImage(old,0,0);const before=Array.from(ctx.getImageData(0,0,c.width,c.height).data);$('heatmapPalette').value='blue-red';await updateMasks();ctx.drawImage(S.raster,0,0);return JSON.stringify(before)!==JSON.stringify(Array.from(ctx.getImageData(0,0,c.width,c.height).data))&&JSON.stringify(labels)===JSON.stringify(Array.from(S.fieldInfo.labels));}")
            page.locator('#trainButton').click()
            page.wait_for_function('trainingWorkspace.samples.length===3&&!trainingWorkspace.loading')
            train = page.get_by_role('checkbox', name='A_PAS_1 training', exact=True)
            val = page.get_by_role('checkbox', name='B_PAS_1 validation', exact=True)
            train.check();val.check()
            assert page.get_by_role('checkbox', name='A_PAS_1 validation', exact=True).is_disabled()
            assert page.get_by_role('checkbox', name='B_PAS_1 training', exact=True).is_disabled()
            page.locator('#trainingSort').select_option('cohort')
            page.locator('#trainingSort').select_option('original_split')
            page.locator('#trainingSort').select_option('medulla_fraction')
            assert page.locator('#trainingSelectionCounts').inner_text() == '1 training samples · 1 validation samples'
            page.locator('#trainingModelName').fill('Browser kidney')
            page.locator('#trainingCValues').fill('0.1, 2')
            page.locator('#submitSelectedTraining').click()
            page.wait_for_function('S.jobs.some(j=>j.model_name==="Browser kidney")')
            job = page.evaluate('S.jobs.find(j=>j.model_name==="Browser kidney")')
            app.state.jobs._threads[job['id']].join(timeout=10)
            assert job['executor'] == 'slurm'
            page.locator('#trainingResultsButton').click()
            assert page.locator('#trainingResults table tbody tr').count() == 2
            page.get_by_role('button', name='Use model', exact=True).click()
            assert page.evaluate("$('modelSelect').value") == 'candidate'
            page.locator('.training-model-record summary').click()
            assert page.locator('.confusion-matrix').is_visible()
            assert 'A_PAS_1' in page.locator('.training-model-record').inner_text()
            page.locator('#trainDialog [aria-label="Close training workspace"]').click()
            page.evaluate("$('overlayEnabled').checked=false;$('overlaySource').value='human';loadSlide(S.slides.find(s=>s.id!==S.detail.id).id)")
            page.wait_for_function('(id)=>S.detail.id!==id&&!S.loading', arg=sid)
            assert page.evaluate("$('overlayEnabled').checked&&$('overlaySource').value==='combined'&&annotations.loaded.size===0")
            assert not errors, errors
            browser.close()
    finally:
        server.should_exit = True;thread.join(timeout=5)
