"""Working-file contracts, schema preservation, crash recovery and vector Booleans."""
import copy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from shapely.ops import unary_union
from SVC_GPU.human_loop.annotation_files import AnnotationFiles, Document, checksum
from SVC_GPU.human_loop.migrate_annotations import plan, execute
from SVC_GPU.human_loop.storage import atomic_json, Conflict


def fc(*geometries):
    return {'type':'FeatureCollection', 'custom': {'retain': 123}, 'features':[
        {'type':'Feature','id':str(i),'properties':{'class':'original','nested':{'a':1}},'geometry':mapping(g)} for i,g in enumerate(geometries)]}


def hist():
    return {'_id':'keep','_elementCount':2,'annotation':{'name':'cortex','metadata':{'keep':True},'elements':[
        {'type':'polyline','closed':True,'points':[[0,0,0],[10,0,0],[10,10,0],[0,10,0]],'lineColor':'rgb(1,2,3)'},
        {'id':'other','type':'polyline','closed':True,'points':[[20,0,0],[30,0,0],[30,10,0],[20,10,0]],'user':{'x':1}}]}}


def files(setup):
    cfg,app,client,slides=setup
    return app.state.annotation_files,slides['A']['id'],client


def test_schema_metadata_holes_multipolygons_roundtrip():
    hole=Polygon([(0,0),(40,0),(40,40),(0,40)],holes=[[(10,10),(20,10),(20,20),(10,20)]])
    value=fc(MultiPolygon([hole,box(50,0,60,10)]))
    doc=Document(value)
    assert doc.value==value
    edited=doc.edit([{'action':'rename','id':'0','name':'renamed'}])
    assert edited['features'][0]['geometry']==value['features'][0]['geometry']
    assert edited['features'][0]['properties']['nested']=={'a':1}
    assert Document(edited).union().equals(Document(value).union())
    value['features'][0]['id']=0
    numeric=Document(value).edit([{'action':'rename','id':'0','name':'zero'}])
    assert numeric['features'][0]['id']==0 and '_histoloop_id' not in numeric['features'][0]


def test_histomics_edits_preserve_unrelated_metadata_and_reject_conversion():
    original=hist();doc=Document(original);rid=doc.regions()[0]['id']
    edited=doc.edit([{'action':'replace','id':rid,'geometry':mapping(box(1,1,9,9))}])
    assert edited['_id']=='keep' and edited['annotation']['metadata']==original['annotation']['metadata']
    assert edited['annotation']['elements'][1]==original['annotation']['elements'][1]
    assert edited['annotation']['elements'][0]['lineColor']=='rgb(1,2,3)'
    assert Document(edited).regions()[0]['id']==rid
    original['annotation']['elements'][0]['_histoloop_id']=rid
    original['annotation']['elements'][0]['label']={'value':'cortex','metadata':{'keep':True}}
    renamed=Document(original).edit([{'action':'rename','id':rid,'name':'Region name'}])
    assert renamed['annotation']['elements'][0]['label']==original['annotation']['elements'][0]['label']
    assert Document(renamed).regions()[0]['name']=='Region name'
    with pytest.raises(ValueError,match='cannot represent'):
        Document(original).edit([{'action':'replace','id':rid,'geometry':mapping(Polygon([(0,0),(40,0),(40,40),(0,40)],holes=[[(1,1),(2,1),(2,2),(1,2)]]))}])


def test_invalid_geometry_is_reported_without_repair():
    value=fc(Polygon([(0,0),(10,10),(10,0),(0,10)]))
    with pytest.raises(ValueError,match='Invalid geometry'):
        Document(value)


def test_save_correct_file_and_concurrent_external_revisions(setup):
    store,sid,client=files(setup)
    a=store.save(sid,'cortical_gt.json',{'revision':None,'document':fc(box(0,0,10,10))})
    b=store.save(sid,'medullary_gt.json',{'revision':None,'document':hist()})
    payload={'revision':a['revision'],'operations':[{'action':'delete','id':'0'}]}
    def save():
        try:return store.save(sid,'cortical_gt.json',payload)['status']
        except Conflict:return 'conflict'
    with ThreadPoolExecutor(2) as executor:
        result=list(executor.map(lambda _:save(),range(2)))
    assert sorted(result)==['conflict','saved']
    assert checksum(store.layer_path(sid,'medullary_gt.json'))==b['revision']
    assert store.read(sid,'cortical_gt.json')['regions']==[]
    assert list(store.root.glob('.history/*/cortical_gt.json/*.json'))
    assert store.list(sid)['layers'][0]['present']
    atomic_json(store.layer_path(sid,'medullary_gt.json'),fc())
    with pytest.raises(Conflict):store.save(sid,'medullary_gt.json',{'revision':b['revision'],'operations':[]})
    assert client.get(f'/api/slides/{sid}').json()['manual_tiles']==0


def test_atomic_failure_retains_original_and_reload(setup,monkeypatch):
    store,sid,_=files(setup);store.save(sid,'cortical_gt.json',{'revision':None,'document':hist()})
    path=store.layer_path(sid,'cortical_gt.json');before=path.read_bytes();revision=checksum(path)
    import SVC_GPU.human_loop.storage as storage
    replace=storage.os.replace
    def fail(source,target):
        if Path(target)==path:raise OSError('simulated interruption')
        return replace(source,target)
    monkeypatch.setattr(storage.os,'replace',fail)
    with pytest.raises(OSError):store.save(sid,'cortical_gt.json',{'revision':revision,'operations':[{'action':'delete','id':'other'}]})
    assert path.read_bytes()==before and not list(path.parent.glob('*.tmp'))
    cfg,app,_,_=setup
    recovered=AnnotationFiles(cfg,app.state.catalog)
    assert len(recovered.read(sid,'cortical_gt.json')['regions'])==2


def test_medulla_priority_holes_components_missing_and_empty(setup):
    store,sid,_=files(setup)
    c=Polygon([(0,0),(100,0),(100,100),(0,100)],holes=[[(5,5),(10,5),(10,10),(5,10)]])
    m=MultiPolygon([box(40,0,60,100),box(110,0,120,10)])
    tissue=MultiPolygon([box(0,0,90,90),box(110,0,125,20)])
    for name,value in [('cortical_gt.json',fc(c)),('medullary_gt.json',fc(m)),('Tissue_qc.json',fc(tissue))]:
        store.save(sid,name,{'revision':None,'document':value})
    result=store.regenerate(sid)
    cg=Document(store.read(sid,'cortical.json')['document']).union();mg=Document(store.read(sid,'medullary.json')['document']).union()
    assert result['status']=='current'
    assert cg.intersection(mg).area==0
    assert unary_union([cg,mg]).symmetric_difference(unary_union([c,m]).intersection(tissue)).area<1e-8
    assert store.read(sid,'cortical.json')['document']['histoloop']['id']==store.read(sid,'medullary.json')['document']['histoloop']['id']
    ids=[r['id'] for r in store.read(sid,'cortical.json')['regions']]
    store.regenerate(sid)
    assert ids==[r['id'] for r in store.read(sid,'cortical.json')['regions']]
    rev=checksum(store.layer_path(sid,'cortical_gt.json'))
    store.save(sid,'cortical_gt.json',{'revision':rev,'remove':True});store.regenerate(sid)
    assert not store.layer_path(sid,'cortical.json').exists()
    assert store.layer_path(sid,'medullary.json').exists()
    rev=checksum(store.layer_path(sid,'medullary_gt.json'))
    store.save(sid,'medullary_gt.json',{'revision':rev,'document':fc(box(300,300,400,400))});store.regenerate(sid)
    assert store.read(sid,'medullary.json')['regions']==[]
    assert next(l for l in store.list(sid)['layers'] if l['id']=='cortical_gt.json')['status']=='missing'


def test_generation_partial_publication_recovers_same_snapshot(setup,monkeypatch):
    store,sid,_=files(setup)
    for name in ['cortical_gt.json','medullary_gt.json','Tissue_qc.json']:
        store.save(sid,name,{'revision':None,'document':fc(box(0,0,100,100))})
    import SVC_GPU.human_loop.annotation_files as af
    atomic=af.atomic_json
    def fail(path,value):
        if Path(path).name=='medullary.json' and Path(path).parent==store.folder(sid):raise OSError('publication interrupted')
        return atomic(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(af,'atomic_json',fail)
        with pytest.raises(OSError):store.regenerate(sid)
    assert store.list(sid)['generation']['status']=='pending'
    store.recover(sid)
    assert store.list(sid)['generation']['status']=='current'
    assert store.read(sid,'cortical.json')['document']['histoloop']['id']==store.read(sid,'medullary.json')['document']['histoloop']['id']


def test_missing_invalid_or_ambiguous_tissue(setup):
    store,sid,client=files(setup)
    store.save(sid,'cortical_gt.json',{'revision':None,'document':fc(box(0,0,100,100))})
    assert store.regenerate(sid)['status']=='unavailable'
    assert store.read(sid,'cortical_gt.json')['regions']
    contours=store.root.parent/'contours';contours.mkdir();store.contours=contours
    name=store.catalog.slide(sid)['name']
    atomic_json(contours/(name+'.json'),fc(box(0,0,50,50)))
    atomic_json(contours/(name+'.geojson'),fc(box(0,0,100,100)))
    assert 'Select' in store.regenerate(sid)['error']
    selected=next(iter(store.tissue_candidates(sid)))
    assert store.regenerate(sid,selected)['status']=='current'
    source_path=store.tissue_path(store.tissue_candidates(sid)[selected]);original=source_path.read_bytes()
    response=client.post(f'/api/slides/{sid}/annotation-tissue-copy',json={'reviewer':'test','tissue_id':selected})
    assert response.status_code==200,response.text
    working=store.layer_path(sid,'Tissue_qc.json')
    assert working.is_file() and store.tissue(sid)[1]==working and source_path.read_bytes()==original
    current=store.read(sid,'Tissue_qc.json')
    assert client.put(f'/api/slides/{sid}/annotations/Tissue_qc.json',json={'reviewer':'test','revision':current['revision'],
        'operations':[{'action':'replace','id':current['regions'][0]['id'],'geometry':mapping(box(1,1,40,40))}]}).status_code==200
    assert source_path.read_bytes()==original


def test_api_create_edit_delete_and_separation(setup):
    store,sid,client=files(setup);url=f'/api/slides/{sid}/annotations/cortical_gt.json'
    assert client.put(url,json={'reviewer':'test','revision':None,'create':True}).status_code==200
    layer=client.get(url).json();assert not layer['regions']
    response=client.put(url,json={'reviewer':'test','revision':layer['revision'],'operations':[{'action':'add','geometry':mapping(box(0,0,30,30))}]})
    assert response.status_code==200
    layer=client.get(url).json();rid=layer['regions'][0]['id']
    assert client.put(url,json={'reviewer':'test','revision':layer['revision'],'operations':[{'action':'replace','id':rid,'geometry':mapping(box(0,0,60,60))}]}).status_code==200
    layer=client.get(url).json();assert layer['regions'][0]['id']==rid
    assert client.put(url,json={'reviewer':'test','revision':layer['revision'],'operations':[{'action':'delete','id':rid}]}).status_code==200
    assert client.get(url).json()['regions']==[]
    assert client.get(f'/api/slides/{sid}/history').json()['events']==[]
    assert client.put(f'/api/slides/{sid}/annotations/cortical.json',json={'reviewer':'test','revision':None,'create':True}).status_code==400


def test_idempotent_migration_missing_alias_conflicts_and_edited_destinations(tmp_path):
    source=tmp_path/'KPMP_batch2'/'Labelled_files';source.mkdir(parents=True)
    for name in ['a.svs','empty.svs','alias.svs','conflict.svs']:(source/name).mkdir()
    atomic_json(source/'a.svs'/'cortical_gt.json',hist())
    atomic_json(source/'alias.svs'/'cortex_gt.json',hist())
    atomic_json(source/'conflict.svs'/'cortex_gt.json',hist())
    atomic_json(source/'conflict.svs'/'cortical_gt.json',hist())
    cfg=SimpleNamespace(annotation_root=str(tmp_path/'working'),state_dir=str(tmp_path/'state'),annotation_sources=json.dumps([str(source),str(tmp_path/'missing')]))
    catalog=SimpleNamespace(slides={'a':{'id':'a','name':'a','wsi_path':str(tmp_path/'KPMP_batch2'/'a.svs'),'dataset_id':'d'}})
    dry=plan(cfg,catalog);assert not Path(cfg.annotation_root).exists();assert dry['conflicts'] and dry['missing_roots'];assert len(dry['samples'])==4
    assert execute(dry)['copied']==2
    a=Path(cfg.annotation_root)/'a'/'cortical_gt.json';assert a.read_bytes()==(source/'a.svs'/'cortical_gt.json').read_bytes()
    atomic_json(a,fc());assert execute(plan(cfg,catalog))['copied']==0 and json.loads(a.read_text())==fc()
    assert not list((Path(cfg.annotation_root)/'empty').glob('*.json'))
    assert not (Path(cfg.annotation_root)/'conflict'/'cortical_gt.json').exists()


def test_duplicate_names_do_not_merge_samples(tmp_path):
    roots=[]
    slides={}
    for cohort in ['one','two']:
        root=tmp_path/('KPMP_'+cohort)/'Labelled_files';(root/'same.svs').mkdir(parents=True);roots.append(str(root))
        atomic_json(root/'same.svs'/'cortical_gt.json',hist())
        slides[cohort]={'id':cohort,'name':'same','wsi_path':str(root.parent/'same.svs'),'dataset_id':cohort}
    cfg=SimpleNamespace(annotation_root=str(tmp_path/'working'),state_dir=str(tmp_path/'state'),annotation_sources=json.dumps(roots))
    dry=plan(cfg,SimpleNamespace(slides=slides));assert dry['duplicate_names']==['same']
    assert len({s['folder'] for s in dry['samples']})==2
    execute(dry);assert len(list(Path(cfg.annotation_root).glob('same--*/cortical_gt.json')))==2


def test_saved_version_restore_and_symlink_rejection(setup):
    store,sid,client=files(setup)
    first=store.save(sid,'cortical_gt.json',{'revision':None,'document':fc(box(0,0,30,30))})
    second=store.save(sid,'cortical_gt.json',{'revision':first['revision'],'operations':[{'action':'delete','id':'0'}]})
    versions=store.versions(sid,'cortical_gt.json');assert versions
    restored=store.restore(sid,'cortical_gt.json',{'revision':second['revision'],'version':versions[0]['hash']})
    assert restored['revision']==first['revision'] and len(restored['regions'])==1
    path=store.folder(sid)/'escape.json';path.symlink_to(store.layer_path(sid,'cortical_gt.json'))
    with pytest.raises(ValueError,match='Symlinks'):store.layer_path(sid,'escape.json')
    with pytest.raises(ValueError):store.layer_path(sid,'../outside.json')
    assert next(l for l in store.list(sid)['layers'] if l['id']=='escape.json')['status']=='error'
    qc=store.folder(sid)/'Tissue_qc.json';qc.symlink_to(path)
    with pytest.raises(ValueError,match='Symlinks'):store.tissue(sid)
