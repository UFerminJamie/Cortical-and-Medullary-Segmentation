import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from test_scaling import overlap_sample


def test_worker_grids_mobile_panels_and_group_submission(setup):
    sync_playwright = pytest.importorskip('playwright.sync_api').sync_playwright
    config, app, client, slides = setup
    overlap_sample(config, app, slides['A'])
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started and time.monotonic()<deadline:time.sleep(.02)
        assert server.started
        with sync_playwright() as manager:
            if not Path(manager.chromium.executable_path).is_file():pytest.skip('Chromium not installed')
            browser=manager.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':950})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{port}/?token=test-access')
            page.wait_for_function('S.detail && !S.loading')
            page.evaluate('(id)=>loadSlide(id)',slides['A']['id'])
            page.wait_for_function('(id)=>S.detail?.id===id&&!S.loading',arg=slides['A']['id'])
            page.evaluate("$('cortexComparison').value='all';$('medullaComparison').value='all';$('cortexEnabled').checked=true;$('medullaEnabled').checked=true;")
            for mode,count in [('0',4),('50',9),('75',25)]:
                actual=page.evaluate('''async mode=>{$('gridMode').value=mode;configureGrid();await updateMasks();
                  return [...S.mask].reduce((a,b)=>a+b,0);}''',mode)
                assert actual==count
            # A dropdown attempt cannot silently change an unfinished region's tile scope.
            for state in ['draft','pendingPolygon','saving']:
                payload=page.evaluate('''state=>{
                  S[state]=state==='saving'?true:[[0,0],[100,0],[100,100]];
                  $('gridMode').value='0';$('gridMode').dispatchEvent(new Event('change'));
                  $('sourceOverlap').value='50';$('sourceOverlap').dispatchEvent(new Event('change'));
                  const payload=gridPayload();S[state]=state==='saving'?false:state==='draft'?[]:null;
                  return payload;
                }''',state)
                assert payload['grid_mode']=='75' and payload['source_overlap']=='auto'
            # Annotation drawing does not recompute or repaint the prediction layer.
            assert page.evaluate('''async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);
              let calls=0;const original=renderOverlay;renderOverlay=()=>{calls++;original();};
              requestDraw();await new Promise(requestAnimationFrame);renderOverlay=original;return calls;}''')==0
            for width in [1440,1024,768,390]:
                page.set_viewport_size({'width':width,'height':850});page.wait_for_timeout(150)
                layout=page.evaluate("({document:document.documentElement.scrollWidth,width:innerWidth,viewer:$('viewerWrap').clientWidth,height:$('viewerWrap').clientHeight})")
                assert layout['document']<=width and layout['viewer']>0 and layout['height']>200,layout
            page.locator('#queueToggle').click()
            assert page.locator('#queuePanel').evaluate('panel=>!panel.inert')
            page.locator('#panelBackdrop').click(position={'x':380,'y':50})
            page.locator('#inspectorToggle').click()
            assert page.locator('#inspectorPanel').evaluate('panel=>!panel.inert')
            captured=[]
            def submission(route):
                payload=route.request.post_data_json;captured.append(payload)
                route.fulfill(status=202,json={'id':'predict-browser-test','kind':'predict','executor':'slurm',
                    'status':'pending','created_at':'2026-10-07T00:00:00Z','slide_name':'2 slides',
                    'slide_ids':payload['slide_ids'],'threshold':payload['threshold']})
            page.route('**/api/jobs',lambda route:submission(route) if route.request.method=='POST' else route.continue_())
            page.evaluate("$('reviewer').value='Test';S.selectedSlides=new Set(S.slides.slice(0,2).map(s=>s.id));$('predictionScope').value='selected';updatePredictionPlan();")
            page.evaluate("submitJob('predict')")
            assert len(captured)==1 and len(captured[0]['slide_ids'])==2 and captured[0]['executor']=='slurm'
            assert not errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10);sock.close()
