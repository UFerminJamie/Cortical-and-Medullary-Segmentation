"use strict";
const annotations={layers:[],loaded:new Map(),preferences:{},selected:null,editing:null,sequence:0,
  busy(){return [...this.loaded.values()].some(l=>l.dirty||l.saving)||!!this.editing;},
  url(layer,slide=S.detail.id){return '/api/slides/'+slide+'/annotations/'+encodeURIComponent(layer);},
  async load(slide,signal){
    const sequence=++this.sequence;this.editing=null;this.selected=null;this.loaded.clear();
    const result=await api('/api/slides/'+slide+'/annotations',{signal});
    if(sequence!==this.sequence||S.detail?.id!==slide)return;
    this.layers=result.layers;this.generation=result.generation;this.tissues=result.tissues;this.selectedTissue=result.selected_tissue;
    for(const layer of this.layers){const pref=this.pref(layer.id);pref.visible=false;pref.expanded=false;}
    this.render();
    if(sequence===this.sequence){this.render();requestRender();}
  },
  pref(id){const key=S.detail.id+':'+id;return this.preferences[key]||(this.preferences[key]={visible:false,color:id.includes('medull')?'#f18582':id.includes('cort')?'#529cff':'#e0cd48',opacity:.2,expanded:false,page:0,hidden:{}});},
  async display(id,expand=false){const pref=this.pref(id);if(expand)pref.expanded=!pref.expanded;else pref.visible=!pref.visible;
    this.render();requestRender();schedulePreferences();
    if((pref.visible||pref.expanded)&&!this.loaded.has(id))try{await this.read(id,S.slideController?.signal);this.render();}catch(e){if(e.name!=='AbortError'){toast(e.message,true);pref.visible=false;this.render();}}
  },
  async all(show){const slide=S.detail?.id;if(!slide)return;const sequence=this.sequence;
    for(const layer of this.layers){const pref=this.pref(layer.id);pref.visible=show;pref.expanded=show;if(show)pref.hidden={};}
    if(!show)this.selected=null;this.render();requestRender();schedulePreferences();
    if(show)for(const layer of this.layers.filter(l=>l.present&&l.status!=='error')){
      if(sequence!==this.sequence||slide!==S.detail?.id)return;
      if(!this.pref(layer.id).visible)continue;
      if(!this.loaded.has(layer.id))try{await this.read(layer.id,S.slideController?.signal);this.render();}catch(e){if(e.name!=='AbortError')toast(layer.name+': '+e.message,true);}
    }
  },
  async read(id,signal){
    const slide=S.detail.id,sequence=this.sequence;
    const value=await api(this.url(id),{signal});
    if(slide!==S.detail?.id||sequence!==this.sequence)return;
    if(this.loaded.get(id)?.dirty||this.loaded.get(id)?.saving||this.editing?.layer===id)return;
    this.loaded.set(id,{...value,operations:[],dirty:false,saving:false,status:'saved',undo:[],redo:[]});requestRender();return value;
  },
  checkpoint(id){const l=this.loaded.get(id);l.undo.push({regions:structuredClone(l.regions),operations:structuredClone(l.operations)});if(l.undo.length>40)l.undo.shift();l.redo=[];},
  modify(id,op){const l=this.loaded.get(id);if(!l||l.saving)return;this.checkpoint(id);
    if(op.action==='replace'||op.action==='delete')l.operations=l.operations.filter(old=>old.id!==op.id||!(op.action==='delete'?['replace','rename']:['replace']).includes(old.action));
    l.operations.push(op);l.dirty=true;l.status='dirty';
    if(op.action==='delete')l.regions=l.regions.filter(r=>r.id!==op.id);
    else if(op.action==='replace'){const r=l.regions.find(r=>r.id===op.id);r.geometry=structuredClone(op.geometry);if('name'in op)r.name=op.name;}
    else if(op.action==='rename')l.regions.find(r=>r.id===op.id).name=op.name;
    else if(op.action==='add')l.regions.push({id:'draft-'+l.operations.length,name:op.name||'',geometry:structuredClone(op.geometry)});
    this.render();requestRender();savedStatus('Annotation file has unsaved changes');
  },
  undo(id,redo=false){const l=this.loaded.get(id);if(!l||l.saving)return;const from=redo?l.redo:l.undo,to=redo?l.undo:l.redo,snapshot=from.pop();if(!snapshot)return;
    to.push({regions:structuredClone(l.regions),operations:structuredClone(l.operations)});Object.assign(l,snapshot);l.dirty=!!l.operations.length;l.status=l.dirty?'dirty':'saved';this.editing=null;this.render();requestRender();},
  async save(id){if(this.editing?.layer===id)this.commitGeometry();const l=this.loaded.get(id);if(!l?.dirty||l.saving)return;const slide=S.detail.id;l.saving=true;l.status='saving';this.render();
    try{const result=await api(this.url(id),{method:'PUT',body:{...actorPayload(),revision:l.revision,operations:l.operations}});Object.assign(l,result,{dirty:false,saving:false,operations:[],undo:[],redo:[]});this.editing=null;this.render();requestRender();
      if(['cortical_gt.json','medullary_gt.json','Tissue_qc.json'].includes(id)){for(const layer of this.layers)if(layer.derived)layer.status='stale';this.generation={status:'pending'};this.render();}
      savedStatus('Annotation file saved');this.refresh().catch(e=>toast('File saved; layer refresh failed: '+e.message,true));setTimeout(()=>{if(S.detail?.id===slide&&!this.busy())this.refresh().catch(e=>toast('File saved; derived layers: '+e.message,true));},500);
    }catch(e){l.saving=false;l.status='failed';l.error=e.message;savedStatus('File save failed · draft retained',true);this.render();toast(e.message,true);}
  },
  async refresh(){if(!S.detail)return;const slide=S.detail.id,result=await api('/api/slides/'+slide+'/annotations');if(S.detail?.id!==slide)return;
    this.layers=result.layers;this.generation=result.generation;this.tissues=result.tissues;this.selectedTissue=result.selected_tissue;
    for(const layer of this.layers.filter(l=>l.present&&l.status!=='error'))if((this.loaded.has(layer.id)||this.pref(layer.id).visible||this.pref(layer.id).expanded)&&!this.loaded.get(layer.id)?.dirty)await this.read(layer.id);
    this.render();predictionTissue();requestRender();},
  async create(id){if(!S.detail)return;try{await api(this.url(id),{method:'PUT',body:{...actorPayload(),revision:null,create:true}});this.pref(id).visible=true;this.pref(id).expanded=true;await this.refresh();$('annotationTarget').value=id;this.render();}catch(e){toast(e.message,true);}},
  async downloadDraft(id){const l=this.loaded.get(id);try{const doc=await api('/api/annotation-preview',{method:'POST',body:{...actorPayload(),document:l.document,operations:l.operations}}),url=URL.createObjectURL(new Blob([JSON.stringify(doc)],{type:'application/json'})),link=node('a',{href:url,download:id});link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){toast('Draft download failed; draft is still retained: '+e.message,true);}},
  async versions(id){try{if(this.busy()){toast('Save or undo file drafts before restoring a version.');return;}const versions=await api('/api/slides/'+S.detail.id+'/annotation-versions/'+encodeURIComponent(id));$('annotationVersionList').replaceChildren(...versions.map(v=>node('button',{class:'button',text:new Date(v.time_ns/1e6).toLocaleString()+' · '+v.hash.slice(0,10),onclick:async()=>{try{await api('/api/slides/'+S.detail.id+'/annotation-versions/'+encodeURIComponent(id),{method:'POST',body:{...actorPayload(),revision:this.loaded.get(id)?.revision??null,version:v.hash}});$('annotationVersionsDialog').close();await this.refresh();}catch(e){toast(e.message,true);}}})));if(!versions.length)$('annotationVersionList').textContent='No saved backups yet.';$('annotationVersionsDialog').showModal();}catch(e){toast(e.message,true);}},
  async regenerate(){const tissue=$('tissueChoice').value||undefined;for(const layer of this.layers)if(layer.derived)layer.status='stale';this.generation={status:'pending'};this.render();try{const result=await api('/api/slides/'+S.detail.id+'/annotation-generation',{method:'POST',body:{...actorPayload(),tissue_id:tissue}});if(result.status!=='current')toast(result.error||result.status,true);await this.refresh();}catch(e){toast(e.message,true);}},
  rings(geometry){return geometry.type==='Polygon'?geometry.coordinates:geometry.coordinates.flat();},
  select(id,region,zoom=false){this.selected={layer:id,id:region.id};if(zoom){const points=this.rings(region.geometry).flat();focusRegion(points);}requestRender();},
  beginEdit(id,region){const l=this.loaded.get(id);if(region.id.startsWith('draft-')){toast('Save the new region before vertex editing.');return;}
    this.select(id,region);this.editing={layer:id,id:region.id,geometry:structuredClone(region.geometry),ring:0,vertex:0};$('annotationTarget').value=id;setMode('pan');$('drawSurface').classList.add('drawing');S.viewer.setMouseNavEnabled(false);this.render();this.vertexEditor.scrollIntoView({block:'nearest'});requestRender();},
  commitGeometry(){const e=this.editing;if(!e)return;this.editing=null;this.modify(e.layer,{action:'replace',id:e.id,geometry:e.geometry});setMode('pan');},
  add(points){const id=$('annotationTarget').value,l=this.loaded.get(id);if(!l){toast('Create the annotation file first.',true);return;}
    const closed=[...points,points[0]];this.modify(id,{action:'add',geometry:{type:'Polygon',coordinates:[closed]},name:$('regionNote').value});S.draft=[];S.pendingPolygon=null;S.cursor=null;updateButtons();requestRender();},
  pointer(event){const e=this.editing;if(!e)return false;const p=imagePoint(event),item=S.viewer.world.getItemAt(0),rings=this.rings(e.geometry);if(!p||!item)return true;
    if(event.type==='pointerdown'){
      let best=null,distance=12;const screen=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(...p));
      rings.forEach((ring,r)=>ring.slice(0,-1).forEach((v,i)=>{const q=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(v[0],v[1])),d=Math.hypot(q.x-screen.x,q.y-screen.y);if(d<distance){distance=d;best=[r,i];}}));
      if(best){[e.ring,e.vertex]=best;e.drag=true;$('drawSurface').setPointerCapture(event.pointerId);this.render();}
    }else if(event.type==='pointermove'&&e.drag){const ring=rings[e.ring];ring[e.vertex]=[p[0],p[1],...ring[e.vertex].slice(2)];if(e.vertex===0)ring[ring.length-1]=[...ring[0]];requestRender();}
    else if(event.type==='pointerup'||event.type==='pointercancel'){e.drag=false;this.render();}
    return true;
  },
  draw(ctx,p0,scale){
    for(const layer of this.layers){const pref=this.pref(layer.id),l=this.loaded.get(layer.id);if(!pref.visible||!l)continue;
      for(const region of l.regions){if(pref.hidden[region.id])continue;
        const b=region.bounds,rect=$('viewerWrap').getBoundingClientRect();
        if(!l.dirty&&this.editing?.id!==region.id&&b?.length===4&&(p0.x+b[2]*scale<0||p0.y+b[3]*scale<0||p0.x+b[0]*scale>rect.width||p0.y+b[1]*scale>rect.height))continue;
        const selected=this.selected?.layer===layer.id&&this.selected.id===region.id;
        const geom=this.editing?.layer===layer.id&&this.editing.id===region.id?this.editing.geometry:region.geometry;
        ctx.beginPath();for(const ring of this.rings(geom)){ring.forEach((p,i)=>i?ctx.lineTo(p0.x+p[0]*scale,p0.y+p[1]*scale):ctx.moveTo(p0.x+p[0]*scale,p0.y+p[1]*scale));ctx.closePath();}
        ctx.fillStyle=pref.color;ctx.globalAlpha=pref.opacity;ctx.fill('evenodd');ctx.globalAlpha=1;ctx.strokeStyle=selected?'#fff':pref.color;ctx.lineWidth=selected?3:1.5;ctx.stroke();
      }
    }
    if(this.editing){ctx.fillStyle='#fff';for(const ring of this.rings(this.editing.geometry))for(const p of ring.slice(0,-1)){ctx.beginPath();ctx.arc(p0.x+p[0]*scale,p0.y+p[1]*scale,4,0,Math.PI*2);ctx.fill();}}
  },
  render(){
    if(this.rendering){this.renderAgain=true;return;}
    this.rendering=true;try{
    if(!$('annotationFiles'))return;
    const target=$('annotationTarget').value;
    $('annotationTarget').replaceChildren(node('option',{value:'corrections',text:'Tile corrections'}),...this.layers.filter(l=>l.editable&&l.present&&l.status!=='error').map(l=>node('option',{value:l.id,text:'File: '+l.name})));
    if([...$('annotationTarget').options].some(o=>o.value===target))$('annotationTarget').value=target;
    $('fileSave').hidden=target==='corrections';$('fileUndo').hidden=target==='corrections';$('fileRedo').hidden=target==='corrections';
    $('fileSave').disabled=(!this.loaded.get(target)?.dirty&&this.editing?.layer!==target)||this.loaded.get(target)?.saving;
    $('vertexDone').hidden=!this.editing;
    if(!S.detail){$('annotationFiles').replaceChildren();return;}
    $('derivationStatus').textContent='Derived: '+(this.generation?.status||'unavailable')+(this.generation?.error?' · '+this.generation.error:'');
    $('tissueChoice').replaceChildren(node('option',{value:'',text:'Select tissue mask'}),...(this.tissues||[]).map(t=>node('option',{value:t.id,text:t.name})));$('tissueChoice').value=this.selectedTissue||'';
    if($('annotationFiles').closest('[hidden]')||$('inspectorPanel').inert||!$('tab-annotate').classList.contains('active'))return;
    const rows=[];
    for(const layer of this.layers){const pref=this.pref(layer.id),l=this.loaded.get(layer.id),box=node('div',{class:'annotation-file','data-layer':layer.id}),header=node('div',{class:'annotation-file-row'});
      header.append(node('button',{class:'square-button',text:pref.visible?'◉':'○','aria-label':'Toggle '+layer.name,'aria-pressed':String(pref.visible),onclick:()=>this.display(layer.id)}),
        node('button',{class:'text-button layer-expand',text:(pref.expanded?'▾ ':'▸ ')+layer.name,'aria-expanded':String(pref.expanded),'aria-controls':'regions-'+layer.id,onclick:()=>this.display(layer.id,true)}));
      const color=node('input',{type:'color',value:pref.color,'aria-label':layer.name+' color',oninput:e=>{pref.color=e.target.value;requestRender();schedulePreferences();}});
      const opacity=node('input',{type:'range',min:'0',max:'1',step:'.05',value:pref.opacity,'aria-label':layer.name+' opacity',oninput:e=>{pref.opacity=Number(e.target.value);requestRender();schedulePreferences();}});
      box.append(header,node('div',{class:'layer-summary'},[color,opacity,node('span',{text:(l?.regions.length??layer.count)+' regions · '+(layer.derived?layer.status:l?.status||(layer.present&&layer.status!=='error'?'not loaded':layer.status))})]));
      box.append(node('small',{text:layer.derived?'Derived · read-only; edit ground truth':layer.editable?'Working file · editable':'Tissue contour · read-only'}));
      if(!layer.present&&layer.editable)box.append(node('button',{class:'text-button',text:'Create '+layer.name,onclick:()=>this.create(layer.id)}));
      if(l?.error||layer.error)box.append(node('div',{class:'file-error',text:l?.error||layer.error}));
      if(l?.dirty)box.append(node('button',{class:'button',text:l.status==='failed'?'Retry save':'Save file',onclick:()=>this.save(layer.id)}),node('button',{class:'text-button',text:'Undo',onclick:()=>this.undo(layer.id)}),node('button',{class:'text-button',text:'Redo',onclick:()=>this.undo(layer.id,true)}));
      if(l?.status==='failed')box.append(node('button',{class:'text-button',text:'Download draft',onclick:()=>this.downloadDraft(layer.id)}));
      if(this.editing?.layer===layer.id)box.append(this.vertexEditor);
      if(pref.expanded&&l){const list=node('div',{id:'regions-'+layer.id,class:'region-list'}),pageSize=30,max=Math.max(0,Math.ceil(l.regions.length/pageSize)-1);pref.page=Math.min(pref.page||0,max);
        for(const region of l.regions.slice(pref.page*pageSize,(pref.page+1)*pageSize)){
          const row=node('div',{class:'region-row'});row.append(node('button',{class:'text-button',text:pref.hidden[region.id]?'○':'◉','aria-label':'Region visibility',onclick:()=>{pref.hidden[region.id]=!pref.hidden[region.id];requestRender();schedulePreferences();this.render();}}),
            node('button',{class:'text-button region-name',text:region.name||region.id.slice(0,16),onclick:()=>this.select(layer.id,region)}),node('button',{class:'text-button',text:'Zoom',onclick:()=>this.select(layer.id,region,true)}));
          if(layer.editable&&!region.id.startsWith('draft-'))row.append(node('button',{class:'text-button',text:'Edit',onclick:()=>this.beginEdit(layer.id,region)}),node('button',{class:'text-button',text:'Delete',onclick:()=>this.modify(layer.id,{action:'delete',id:region.id})}));
          list.append(row);
        }
        if(max)list.append(node('button',{text:'Previous',onclick:()=>{pref.page=Math.max(0,pref.page-1);this.render();}}),node('span',{text:' '+(pref.page+1)+' / '+(max+1)+' '}),node('button',{text:'Next',onclick:()=>{pref.page=Math.min(max,pref.page+1);this.render();}}));
        box.append(list);
        if(layer.editable)box.append(node('button',{class:'text-button',text:'Saved versions / restore…',onclick:()=>this.versions(layer.id)}));
      }
      rows.push(box);
    }
    $('annotationFiles').replaceChildren(...rows);this.renderVertices();
    }finally{this.rendering=false;if(this.renderAgain){this.renderAgain=false;requestAnimationFrame(()=>this.render());}}
  },
  renderVertices(){const container=this.vertexEditor;container.replaceChildren();const e=this.editing;if(!e)return;
    const rings=this.rings(e.geometry),ring=rings[e.ring],point=ring[e.vertex];
    const ringSelect=node('select',{'aria-label':'Polygon ring',onchange:event=>{e.ring=Number(event.target.value);e.vertex=0;this.renderVertices();requestRender();}},rings.map((r,i)=>node('option',{value:i,text:'Ring '+(i+1)+' · '+(r.length-1)+' vertices'})));ringSelect.value=e.ring;
    const select=node('input',{type:'number',min:0,max:ring.length-2,value:e.vertex,'aria-label':'Vertex index',onchange:event=>{e.vertex=Math.max(0,Math.min(ring.length-2,Number(event.target.value)));this.renderVertices();requestRender();}});
    const coords=[0,1].map(axis=>node('input',{type:'number',value:point[axis],step:1,'aria-label':axis?'Vertex Y':'Vertex X',onchange:event=>{point[axis]=Number(event.target.value);if(!e.vertex)ring[ring.length-1]=[...point];requestRender();}}));
    const name=node('input',{value:this.loaded.get(e.layer).regions.find(r=>r.id===e.id)?.name||'',placeholder:'Region name','aria-label':'Region name',onchange:event=>{if(this.loaded.get(e.layer).regions.find(r=>r.id===e.id)?.name!==event.target.value)this.modify(e.layer,{action:'rename',id:e.id,name:event.target.value});}});
    container.append(node('strong',{text:'Vertex editor'}),ringSelect,select,...coords,name,
      node('button',{text:'Insert vertex',onclick:()=>{const next=ring[(e.vertex+1)%(ring.length-1)];ring.splice(e.vertex+1,0,point.map((v,i)=>i<2?(v+next[i])/2:v));e.vertex++;this.renderVertices();requestRender();}}),
      node('button',{text:'Remove vertex',onclick:()=>{if(ring.length<=4){toast('A ring needs three vertices.');return;}ring.splice(e.vertex,1);ring[ring.length-1]=[...ring[0]];e.vertex=Math.min(e.vertex,ring.length-2);this.renderVertices();requestRender();}}),
      node('button',{text:'Apply geometry',onclick:()=>this.commitGeometry()}),node('button',{text:'Cancel geometry',onclick:()=>{this.editing=null;setMode('pan');this.render();requestRender();}}));
  },
  setup(){
    this.vertexEditor=$('vertexEditor');
    $('showAllAnnotations').onclick=()=>this.all(true);$('hideAllAnnotations').onclick=()=>this.all(false);
    $('fileSave').onclick=()=>this.save($('annotationTarget').value);$('fileUndo').onclick=()=>this.undo($('annotationTarget').value);$('fileRedo').onclick=()=>this.undo($('annotationTarget').value,true);$('vertexDone').onclick=()=>this.commitGeometry();
    $('annotationTarget').onchange=async()=>{if(this.editing)this.editing=null;setMode('pan');const id=$('annotationTarget').value;if(id!=='corrections'&&!this.loaded.has(id))try{await this.read(id,S.slideController?.signal);}catch(e){toast(e.message,true);}this.render();};$('regenerateAnnotations').onclick=()=>this.regenerate();
    $('importAnnotation').onclick=()=>{if(this.busy()){toast('Save your file draft before importing.');return;}
      $('importSlide').replaceChildren(...S.slides.map(s=>node('option',{value:s.id,text:s.name})));$('importSlide').value=S.detail?.id;$('annotationImportDialog').showModal();};
    $('annotationImportForm').onsubmit=async event=>{event.preventDefault();const file=$('annotationImportFile').files[0],slide=$('importSlide').value,layer=$('importLayer').value==='custom'?$('importLayerName').value:$('importLayer').value;if(!file)return;
      try{const listing=await api('/api/slides/'+slide+'/annotations'),old=listing.layers.find(l=>l.id===layer);if(old?.present){throw new Error('Import uses a new layer or missing file; existing edits are preserved.');}
        await api(this.url(layer,slide),{method:'PUT',body:{...actorPayload(),revision:null,document:JSON.parse(await file.text())}});$('annotationImportDialog').close();if(S.detail?.id===slide)await this.refresh();toast('Annotation imported into selected slide.');
      }catch(e){toast(e.message,true);}};
    for(const type of ['pointerdown','pointermove','pointerup','pointercancel'])$('drawSurface').addEventListener(type,event=>{if(this.pointer(event)){event.stopImmediatePropagation();event.preventDefault();}},{capture:true});
  }
};
