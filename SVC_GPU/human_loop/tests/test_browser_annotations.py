"""Actual browser editing, persistence, panels, thresholds and cancellation."""
import os
import socket
import threading
import time
from pathlib import Path
import pytest
import uvicorn
from shapely.geometry import box, mapping


def test_annotation_files_panels_and_prediction_workflow(setup):
    playwright=pytest.importorskip('playwright.sync_api').sync_playwright
    cfg,app,client,slides=setup
    files=app.state.annotation_files;sid=slides['A']['id']
    for name,geometry in [('cortical_gt.json',box(0,0,320,512)),('medullary_gt.json',box(250,100,500,450)),('Tissue_qc.json',box(0,0,512,512))]:
        files.save(sid,name,{'revision':None,'document':{'type':'FeatureCollection','metadata':{'retain':True},'features':[{'type':'Feature','id':'stable','properties':{'class':'original'},'geometry':mapping(geometry)}]}})
    files.regenerate(sid)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        with playwright() as manager:
            executable=os.environ.get('HISTOLOOP_BROWSER',manager.chromium.executable_path)
            if not Path(executable).is_file():pytest.skip('Set HISTOLOOP_BROWSER to installed Chromium')
            browser=manager.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1600,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{port}/?token=test-access')
            page.wait_for_function('S.detail&&!S.loading')
            page.evaluate('(id)=>loadSlide(id)',sid)
            page.wait_for_function('(id)=>S.detail?.id===id&&!S.loading&&annotations.layers.length',arg=sid)
            assert page.evaluate('annotations.loaded.size') == 0
            page.locator('#toggleAllAnnotations').click()
            assert page.locator('#toggleAllAnnotations').get_attribute('aria-pressed')=='true'
            page.wait_for_function('annotations.loaded.has("cortical_gt.json")')
            assert page.evaluate('annotations.layers.filter(l=>l.present).length')==5
            page.wait_for_function('annotations.layers.filter(l=>l.present).every(l=>annotations.loaded.has(l.id))')
            page.locator('#toggleAllAnnotations').click()
            assert page.locator('#toggleAllAnnotations').get_attribute('aria-pressed')=='false'
            page.locator('#toggleAllAnnotations').click()
            page.locator('[data-layer="cortical_gt.json"] .layer-expand').click()
            assert page.locator('[data-layer="cortical_gt.json"] .region-row').count()==1
            before=files.layer_path(sid,'cortical_gt.json').read_bytes()
            page.locator('[data-layer="cortical_gt.json"] input[type=color]').fill('#00ff66')
            page.locator('[data-layer="cortical_gt.json"] button[aria-label^="Toggle"]').click()
            assert files.layer_path(sid,'cortical_gt.json').read_bytes()==before
            page.locator('[data-layer="cortical_gt.json"] button[aria-label^="Toggle"]').click()
            page.locator('[data-layer="cortical_gt.json"] .region-row button:text("Edit")').click()
            page.locator('#vertexEditor button:text("Insert vertex")').click()
            page.locator('#vertexEditor button:text("Remove vertex")').click()
            page.locator('#vertexEditor input[aria-label="Vertex X"]').fill('310')
            page.locator('#vertexEditor input[aria-label="Vertex X"]').dispatch_event('change')
            page.locator('#vertexEditor input[aria-label="Region name"]').fill('Edited region')
            page.locator('#vertexEditor input[aria-label="Region name"]').dispatch_event('change')
            page.locator('#vertexDone').click()
            page.locator('#focusViewer').click();assert page.locator('#inspectorPanel').evaluate('p=>p.inert')
            assert page.evaluate('annotations.loaded.get("cortical_gt.json").dirty')
            page.locator('#fileSave').click();page.wait_for_function('!annotations.busy()')
            assert files.read(sid,'cortical_gt.json')['document']['metadata']=={'retain':True}
            assert files.read(sid,'cortical_gt.json')['document']['features'][0]['properties']['name']=='Edited region'
            assert client.get(f'/api/slides/{sid}/history').json()['events']==[]
            assert page.evaluate("async()=>{const pending=annotations.read('cortical_gt.json');annotations.modify('cortical_gt.json',{action:'rename',id:'stable',name:'Retained draft'});await pending;return annotations.loaded.get('cortical_gt.json').dirty;}")
            page.locator('#fileUndo').click()
            page.locator('#focusViewer').click()
            page.evaluate("annotations.pref('cortical_gt.json').expanded=true;annotations.render()")
            page.locator('[data-layer="cortical_gt.json"] .region-row button:text("Delete")').click()
            assert page.evaluate('annotations.loaded.get("cortical_gt.json").regions.length')==0
            page.locator('#fileUndo').click();assert page.evaluate('annotations.loaded.get("cortical_gt.json").regions.length')==1
            page.locator('#fileRedo').click();assert page.evaluate('annotations.loaded.get("cortical_gt.json").regions.length')==0
            page.locator('#fileUndo').click()
            page.locator('#annotationTarget').select_option('cortical_gt.json')
            page.locator('[data-tool=rectangle]').click()
            coords=page.evaluate("()=>{const r=$('viewerWrap').getBoundingClientRect(),i=S.viewer.world.getItemAt(0);return [[20,20],[100,100]].map(p=>{const q=i.imageToViewerElementCoordinates(new OpenSeadragon.Point(...p));return [q.x+r.left,q.y+r.top]});}")
            page.mouse.move(*coords[0]);page.mouse.down();page.mouse.move(*coords[1],steps=4);page.mouse.up()
            page.wait_for_function('annotations.loaded.get("cortical_gt.json").dirty')
            # Failed save keeps draft and explicit retry.
            page.route('**/annotations/cortical_gt.json',lambda route:route.fulfill(status=503,json={'detail':'simulated outage'}) if route.request.method=='PUT' else route.continue_())
            page.locator('#fileSave').click();page.wait_for_function('annotations.loaded.get("cortical_gt.json").status==="failed"')
            assert page.evaluate('annotations.loaded.get("cortical_gt.json").dirty')
            page.unroute('**/annotations/cortical_gt.json')
            page.locator('#fileSave').click();page.wait_for_function('!annotations.busy()')
            assert len(files.read(sid,'cortical_gt.json')['regions'])==2
            page.locator('#regenerateAnnotations').click();page.wait_for_function('annotations.generation.status==="current"')
            page.locator('[aria-controls=panel-overlap]').click()
            assert page.locator('#aggregationMethod').count()==0
            page.evaluate("async()=>{$('predictionView').value='raw-label';await updateMasks();$('predictionView').value='aggregate-score';const first=updateMasks();S.liveThreshold+=.01;await updateMasks();await first;}")
            assert page.evaluate('S.fieldInfo.score.length===S.fieldInfo.width*S.fieldInfo.height')
            page.locator('#liveThresholdNumber').fill('5');page.locator('#liveThresholdNumber').dispatch_event('input')
            page.wait_for_function('S.fieldInfo&&Array.from(S.fieldInfo.labels).every(x=>x!==1)')
            assert files.read(sid,'cortical_gt.json')['document']['metadata']['retain']
            assert client.get(f'/api/slides/{sid}').json()['manual_tiles']==0
            # Worker and main-thread fallback use the same medulla-priority overlap labels.
            parity=page.evaluate("async()=>{S.thresholdPreview=false;S.liveThreshold=.1;S.maskKey=null;await updateMasks();const worker=Array.from(S.fieldInfo.labels);S.engine.close();S.maskKey=null;await updateMasks();const fallback=Array.from(S.fieldInfo.labels),mismatch=[];for(let i=0;i<worker.length&&mismatch.length<8;i++)if(worker[i]!==fallback[i])mismatch.push([i,worker[i],fallback[i]]);return {equal:JSON.stringify(worker)===JSON.stringify(fallback),mismatch};}")
            assert parity['equal'],parity
            page.evaluate("async()=>{S.engine.startWorker();await S.engine.initialize(S.data,S.detail);S.engine.tissue(S.tissueGeometry,S.tissueRevision);S.maskKey=null;await updateMasks();}")
            # Generation cancellation must not publish a previous slide’s field.
            page.evaluate("$('predictionView').value='raw-label';updateMasks();loadSlide(S.slides.find(s=>s.id!==S.detail.id).id)")
            page.wait_for_function('(old)=>S.detail.id!==old&&!S.loading',arg=sid)
            page.wait_for_function('annotations.layers.length&&annotations.loaded.size===0')
            page.locator('[data-layer="cortical_gt.json"] button:text("Create")').click()
            page.wait_for_function('annotations.loaded.has("cortical_gt.json")')
            assert page.evaluate('annotations.loaded.get("cortical_gt.json").regions.length')==0
            page.locator('#annotationTarget').select_option('corrections');page.locator('[data-tool=pan]').click()
            page.locator('[aria-controls=panel-annotations]').press('Enter');assert page.locator('#panel-annotations').is_hidden()
            page.locator('[aria-controls=panel-annotations]').press('Enter')
            handle=page.locator('[data-resize=inspector]');handle.focus();handle.press('ArrowLeft')
            page.wait_for_timeout(800);page.reload();page.wait_for_function('S.detail&&!S.loading')
            assert page.evaluate('S.layout.expanded.annotations') is True
            for width in [1600,1024,768,390]:
                page.set_viewport_size({'width':width,'height':950});page.wait_for_timeout(150)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                assert page.locator('#viewerWrap').evaluate('e=>e.clientHeight')>200
            page.locator('#queueToggle').click();assert page.locator('#queuePanel').evaluate('p=>!p.inert')
            page.locator('#inspectorToggle').click();assert page.locator('#inspectorPanel').evaluate('p=>!p.inert')
            assert page.locator('#queuePanel').evaluate('p=>p.inert')
            assert not errors,errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(10);sock.close()
