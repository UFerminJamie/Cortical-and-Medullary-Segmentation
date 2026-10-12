"use strict";
const $ = id => document.getElementById(id);
const fmt = value => Number(value || 0).toLocaleString();
const S = {slides:[], filtered:[], datasets:[], dataset:null, passes:[], models:[], jobs:[], history:[], detail:null, data:null,
  mask:null, grid:null, raster:null, mode:"pan", label:0, pass:"pass-1", saving:false, loading:false,
  draft:[], drag:null, cursor:null, pendingPolygon:null, highlighted:null, renderPending:false,
  loadSequence:0, maskSequence:0, knownJobs:new Map(), selectedJob:null, pollBusy:false,
  selectedSlides:new Set(), bulkJobs:[], knownBulk:new Map(), bulkPlan:null, viewPrediction:"active",
  queueSequence:0, datasetSequence:0, queueBusy:false, bulkBusy:false, datasetBusy:false, freehandScreen:null, freehandRawCount:0};
const colors = ["#529cff", "#f18582"];
const labelColor = label => label===-2?"#aab2b9":label===-1?"#e0edf4":colors[label];
const labelName = label => label===-2?"excluded":label===-1?"restored to the prediction":label===0?"cortex":"medulla";
let preferenceTimer, filterTimer, planTimer;

function node(tag, attrs={}, children=[]) {
  const element = document.createElement(tag);
  for (const [key,value] of Object.entries(attrs)) {
    if (key === "text") element.textContent = value;
    else if (key === "class") element.className = value;
    else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else element.setAttribute(key, value);
  }
  for (const child of children) element.append(typeof child === "string" ? document.createTextNode(child) : child);
  return element;
}
function toast(message, error=false) {
  const box=node("div", {class:"toast"+(error?" error":"")}, [node("span",{text:message})]);
  box.append(node("button",{text:"×",onclick:()=>box.remove(),"aria-label":"Dismiss notification"}));
  $("toasts").append(box); setTimeout(()=>box.remove(),error?12000:5500);
}
function savedStatus(text, error=false) {
  $("saveStatus").textContent=text; $("saveDot").className="status-dot "+(error?"error":"live");
}
async function api(path, options={}) {
  const init={credentials:"same-origin", ...options};
  if (init.body && typeof init.body !== "string") {
    init.body=JSON.stringify(init.body); init.headers={"Content-Type":"application/json",...init.headers};
  }
  let response;
  try { response=await fetch(path,init); }
  catch(error) { if(error.name==="AbortError")throw error; savedStatus("Connection interrupted · completed saves remain in persistent storage",true); throw new Error("Cannot reach the review server. Check the server session and port forwarding."); }
  if (!response.ok) {
    const body=await response.json().catch(()=>({detail:response.statusText}));
    const error=new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    error.status=response.status; throw error;
  }
  return response.headers.get("Content-Type")?.includes("application/json") ? response.json() : response;
}
function actorPayload() { return {reviewer:$("reviewer").value.trim(),pass_id:S.pass}; }
function filters() {
  return {cortex:{enabled:$("cortexEnabled").checked,comparison:$("cortexComparison").value,threshold:Number($("cortexThreshold").value)},
    medulla:{enabled:$("medullaEnabled").checked,comparison:$("medullaComparison").value,threshold:Number($("medullaThreshold").value)},
    keep_manual:$("keepManual").checked,source:$("overlaySource").value,excluded_enabled:$("excludedEnabled").checked};
}
function schedulePreferences() {
  clearTimeout(preferenceTimer);
  preferenceTimer=setTimeout(async()=>{
    const preference={dataset_id:S.dataset,pass_id:S.pass,slide_id:S.detail?.id,filters:filters(),opacity:Number($("opacity").value),
      overlay:$("overlayEnabled").checked,visible_only:$("visibleOnly").checked,
      model_id:$("modelSelect").value,prediction_threshold:Number($("predictionThreshold").value),
      view_prediction:S.viewPrediction,queue_model:$("queueModelSelect").value,
      executor:S.executor,compute_device:S.device,grid_mode:$("gridMode").value,source_overlap:$("sourceOverlap").value,
      layout:S.layout,annotation_display:annotations.preferences,prediction_display:predictionOptions(),
      queue_filters:Object.fromEntries(["cohortSelect","statusSelect","splitSelect","tissueSelect","sortSelect"].map(id=>[id,$(id).value]))};
    try { await api("/api/preferences",{method:"POST",body:{...actorPayload(),preferences:preference}}); }
    catch(error) { toast("Preferences could not be saved: "+error.message,true); }
  },650);
}
function applyPreferences(preference) {
  annotations.preferences=preference.annotation_display||{};S.panels?.restore(preference.layout);
  for(const [id,value] of Object.entries(preference.prediction_display?.controls||{}))if($(id))$(id).value=value;
  $('tileBoundaries').checked=preference.prediction_display?.boundaries===true;
  if(preference.filters) {
    for(const key of ["cortex","medulla"]) {
      const f=preference.filters[key]; if(!f)continue;
      $(key+"Enabled").checked=f.enabled!==false;
      $(key+"Comparison").value=["lte","gte","all"].includes(f.comparison)?f.comparison:"all";
      $(key+"Threshold").value=Number.isFinite(f.threshold)?f.threshold:0;
    }
    $("keepManual").checked=preference.filters.keep_manual!==false;
    $("excludedEnabled").checked=preference.filters.excluded_enabled!==false;
    if(["combined","prediction","human"].includes(preference.filters.source))$("overlaySource").value=preference.filters.source;
  }
  if(["native","0","50","75"].includes(preference.grid_mode))$("gridMode").value=preference.grid_mode;
  if(["auto","0","50","75"].includes(preference.source_overlap))$("sourceOverlap").value=preference.source_overlap;
  if(Number.isFinite(preference.opacity)) $("opacity").value=preference.opacity;
  $("opacityValue").textContent=$("opacity").value+"%";
  $("overlayEnabled").checked=preference.overlay!==false;
  $("visibleOnly").checked=preference.visible_only!==false;
  if(Number.isFinite(preference.prediction_threshold)) $("predictionThreshold").value=preference.prediction_threshold;
  for(const [id,value] of Object.entries(preference.queue_filters||{}))if($(id)&&[...$(id).options].some(option=>option.value===value))$(id).value=value;
}
function gridPayload() {return {grid_format:2,grid_mode:$("gridMode").value,source_overlap:$("sourceOverlap").value};}
function gridPass(index) {
  const info=S.detail.grid,mode=$("gridMode").value;if(mode==="native")return true;
  const step=info.profiles[$("sourceOverlap").value].steps[mode];
  return (S.data[index*5]-info.origin[0])%step===0&&(S.data[index*5+1]-info.origin[1])%step===0;
}
function configureGrid() {
  if(!S.detail)return;
  const info=S.detail.grid,source=$("sourceOverlap").value,profile=info.profiles[source];
  if(!profile){$("sourceOverlap").value="auto";toast("The selected source overlap does not match the tile coordinates.",true);configureGrid();return;}
  for(const option of $("gridMode").options)option.disabled=option.value!=="native"&&!(option.value in profile.steps);
  if($("gridMode").selectedOptions[0].disabled)$("gridMode").value="native";
  const spacing=profile.steps[$("gridMode").value];
  S.activeGrid={grid_mode:$("gridMode").value,source_overlap:$("sourceOverlap").value};
  $("gridStatus").textContent=(info.native_overlap===null?"Irregular source lattice":info.native_overlap+"% source overlap")+(spacing?" · "+spacing+" px grid spacing":"")+" · corrections affect only the selected grid";
}
function updatePredictionPlan() {
  const slides=bulkSlides($("predictionScope").value);
  $("predictionPlan").textContent=fmt(slides.length)+" slides · one GPU worker processes slides sequentially";
  $("predictButton").textContent="Run SVC on "+fmt(slides.length)+" slide"+(slides.length===1?"":"s");
}
function cohortName(name) { return name.startsWith("batch")?"Batch "+name.slice(5):name.charAt(0).toUpperCase()+name.slice(1); }
function fillPasses() {
  $("passSelect").replaceChildren(...S.passes.map(pass=>node("option",{value:pass.id,text:pass.name})));
  $("passSelect").value=S.pass;
  const current=$("historyPass").value;
  $("historyPass").replaceChildren(node("option",{value:"all",text:"All passes"}),...S.passes.map(pass=>node("option",{value:pass.id,text:pass.name})));
  $("historyPass").value=S.passes.some(p=>p.id===current)?current:"all";
}
function fillModels(selected) {
  const choice=selected||$("modelSelect").value||"baseline";
  $("modelSelect").replaceChildren(...S.models.map(model=>node("option",{value:model.id,text:model.name+" · "+model.id})));
  $("modelSelect").value=S.models.some(model=>model.id===choice)?choice:"baseline";
  const queueChoice=$("queueModelSelect").value||"active";
  $("queueModelSelect").replaceChildren(node("option",{value:"active",text:"Latest prediction per slide"}),
    ...S.models.map(model=>node("option",{value:model.id,text:model.name+(model.id==="baseline"?" · initial predictions":" · "+model.id)})));
  $("queueModelSelect").value=["active",...S.models.map(m=>m.id)].includes(queueChoice)?queueChoice:"active";
}
function updateProgress() {
  const reviewed=S.slides.filter(slide=>slide.reviewed).length;
  const percent=S.slides.length?Math.round(reviewed*100/S.slides.length):0;
  $("totalSlides").textContent=fmt(S.slides.length);
  $("progressText").textContent=fmt(reviewed)+" of "+fmt(S.slides.length)+" reviewed";
  $("progressPercent").textContent=percent+"%"; $("progressFill").style.width=percent+"%";
}
function filterQueue(reset=false) {
  const search=$("search").value.toLowerCase().trim(), cohort=$("cohortSelect").value, status=$("statusSelect").value,
    split=$("splitSelect").value,tissue=$("tissueSelect").value,sort=$("sortSelect").value;
  S.filtered=S.slides.filter(slide=>(!search||(slide.name+" "+slide.case_id).toLowerCase().includes(search))&&(!cohort||slide.cohort===cohort)&&
    (split==="all"||slide.original_split===split)&&
    (tissue==="all"||(tissue==="medulla"&&slide.prediction_available&&slide.medullary>0)||
     (tissue==="cortex"&&slide.prediction_available&&slide.cortical>0&&slide.medullary===0)||
     (tissue==="only_medulla"&&slide.prediction_available&&slide.medullary>0&&slide.cortical===0)||
     (tissue==="missing"&&!slide.prediction_available))&&
    (status==="all"||(status==="remaining"&&!slide.reviewed)||(status==="reviewed"&&slide.reviewed)||
     (status==="corrected"&&(slide.manual_tiles>0||slide.excluded_tiles>0))));
  S.filtered.sort((a,b)=>{
    if(sort!=="name") {
      if(a.medulla_fraction===null&&b.medulla_fraction!==null)return 1;
      if(b.medulla_fraction===null&&a.medulla_fraction!==null)return -1;
      const difference=(a.medulla_fraction??0)-(b.medulla_fraction??0);
      if(difference)return sort==="medulla_asc"?difference:-difference;
    }
    return a.name.localeCompare(b.name);
  });
  $("queueCount").textContent=fmt(S.filtered.length)+" slides · "+(cohort?cohortName(cohort):"all cohorts");
  $("slideListInner").style.height=(S.filtered.length*68)+"px";
  if(reset)$("slideList").scrollTop=0;
  renderQueue(); updatePosition();updateSelection();
}
function renderQueue() {
  const list=$("slideList"),start=Math.max(0,Math.floor(list.scrollTop/68)-3),end=Math.min(S.filtered.length,start+Math.ceil(list.clientHeight/68)+8);
  const rows=[];
  for(let index=start;index<end;index++) {
    const slide=S.filtered[index],state=slide.reviewed?"reviewed":(slide.manual_tiles||slide.excluded_tiles)?"corrected":"";
    const meta=node("div",{class:"slide-row-meta"},[
      node("span",{text:cohortName(slide.cohort)}),node("span",{text:"·"}),
      node("span",{text:(slide.manual_tiles||slide.excluded_tiles)?fmt(slide.manual_tiles)+" labelled · "+fmt(slide.excluded_tiles)+" excluded":fmt(slide.tiles)+" tiles"}),
      node("span",{class:"split-badge "+slide.original_split,text:slide.original_split})]);
    const fraction=slide.medulla_fraction===null?"No prediction":(slide.medulla_fraction*100).toFixed(1)+"% medulla";
    const ratio=slide.medulla_cortex_ratio===null?(slide.medullary>0?"∞":"—"):slide.medulla_cortex_ratio.toFixed(3);
    const check=node("input",{type:"checkbox","aria-label":"Select "+slide.name,onchange:event=>{
      event.target.checked?S.selectedSlides.add(slide.id):S.selectedSlides.delete(slide.id);updateSelection();}});
    check.checked=S.selectedSlides.has(slide.id);
    const row=node("div",{class:"slide-row"+(S.detail?.id===slide.id?" selected":""),title:slide.name+" · "+fraction+" · M/C "+ratio},[
      check,node("button",{class:"slide-open","aria-current":S.detail?.id===slide.id?"true":"false",
        onclick:()=>loadSlide(slide.id).catch(error=>toast(error.message,true))},[
        node("span",{class:"slide-state "+state}),node("div",{class:"slide-row-body"},[
          node("div",{class:"slide-row-name",text:slide.name}),meta,
          node("div",{class:"slide-row-fraction",text:fraction+" · M/C "+ratio})])])]);
    row.style.top=(index*68)+"px"; rows.push(row);
  }
  if(!S.filtered.length)rows.push(node("div",{class:"empty-state",text:"No slides match these filters."}));
  $("slideListInner").replaceChildren(...rows);
}
function updatePosition() {
  const index=S.filtered.findIndex(slide=>slide.id===S.detail?.id);
  $("slidePosition").textContent=index<0?"—":(index+1)+" / "+S.filtered.length;
  $("prevSlide").disabled=index<=0||S.saving; $("nextSlide").disabled=index<0||index>=S.filtered.length-1||S.saving;
}
async function refreshQueue() {
  const seq=++S.queueSequence;S.queueBusy=true;$("bulkButton").disabled=true;
  S.queueController?.abort();const controller=S.queueController=new AbortController();
  try {
    const bytes=await navigationCache.read("/api/slides?compact=true&pass_id="+encodeURIComponent(S.pass)+"&model_id="+encodeURIComponent($("queueModelSelect").value||"active")+
      (S.dataset?"&dataset_id="+encodeURIComponent(S.dataset):""),controller.signal);
    const slides=JSON.parse(new TextDecoder().decode(bytes));
    if(seq!==S.queueSequence)return;
    S.slides=slides;updateProgress();filterQueue();updateDatasetStatus();
  }catch(error){if(error.name!=="AbortError")throw error;
  }finally{if(seq===S.queueSequence){S.queueBusy=false;updateButtons();}}
}
function fillDatasets(preferred) {
  S.dataset=S.datasets.find(item=>item.id===preferred&&item.available)?.id||S.datasets.find(item=>item.available)?.id||null;
  $("datasetSelect").replaceChildren(...S.datasets.map(item=>{
    const option=node("option",{value:item.id,text:item.name+(item.available?"":" · "+item.reason)});
    option.disabled=!item.available;return option;
  }));
  $("datasetSelect").value=S.dataset||"";updateDatasetStatus();
}
function updateDatasetStatus() {
  const dataset=S.datasets.find(item=>item.id===S.dataset);
  $("workspaceLabel").textContent=dataset?dataset.name+" workspace":"Review workspace";
  const missing=S.slides.filter(slide=>!slide.prediction_available).length;
  $("datasetStatus").textContent=dataset?(missing?fmt(missing)+" slides need predictions before review.":""):"No dataset workbooks are available.";
}
function fillCohorts(preferred="") {
  const cohorts=[...new Set(S.slides.map(slide=>slide.cohort))].sort();
  $("cohortSelect").replaceChildren(node("option",{value:"",text:"All cohorts"}),
    ...cohorts.map(cohort=>node("option",{value:cohort,text:cohortName(cohort)})));
  $("cohortSelect").value=cohorts.includes(preferred)?preferred:"";
}
function clearSlide() {
  annotations.sequence++;annotations.layers=[];annotations.loaded.clear();annotations.editing=null;annotations.render();S.fieldInfo=null;
  S.raster?.close?.();S.histogramBins=null;S.detail=null;S.data=null;S.mask=null;S.grid=null;S.raster=null;S.history=[];S.historyNext=null;S.highlighted=null;S.cursor=null;
  S.viewPrediction="active";S.viewer.close();
  for(const id of ["overlayCanvas","drawSurface","histogram"]){const canvas=$(id);canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);}
  $("slideCohort").textContent="SELECT A SLIDE";$("slideTitle").textContent="Choose a slide to review";
  $("slideSubtitle").textContent="WSI image + predicted tiles + human corrections";
  for(const id of ["cortexCount","medullaCount","manualCount","excludedCount","cortexVisible","medullaVisible","excludedVisible","scoreMin","scoreMax"])$(id).textContent="—";
  $("revisionLabel").textContent="revision —";$("currentModelInfo").replaceChildren();$("displayPredictionSelect").replaceChildren();
  $("displayModelBadge").textContent="Select a slide";$("hoverInfo").textContent="Hover over a tile to inspect its score";
  $("loading").hidden=true;renderHistory();updateButtons();
}
async function switchDataset() {
  const next=$("datasetSelect").value;
  if(next===S.dataset)return;
  if(annotations.busy()){ $('datasetSelect').value=S.dataset;toast('Save the annotation file draft before changing datasets.',true);return;}
  if(S.saving||S.bulkBusy||S.datasetBusy||S.pendingPolygon||S.draft.length){
    $("datasetSelect").value=S.dataset;toast("Finish loading, save, or cancel the current region before changing datasets.");return;
  }
  const sequence=++S.datasetSequence;cancelSlideLoad();
  S.dataset=next;S.selectedSlides.clear();S.bulkPlan=null;$("bulkDialog").close();
  clearSlide();S.slides=[];S.filtered=[];updateProgress();filterQueue(true);updateSelection();
  $("search").value="";for(const id of ["statusSelect","splitSelect","tissueSelect"])$(id).value="all";
  fillCohorts();updateDatasetStatus();
  await refreshQueue();if(sequence!==S.datasetSequence)return;
  fillCohorts();filterQueue(true);schedulePreferences();
  const first=S.filtered.find(slide=>slide.prediction_available);
  if(first)await loadSlide(first.id);
  else savedStatus("Dataset selected · predictions are needed before review");
}
function canManageDatasets() {
  if(S.saving||S.loading||S.queueBusy||S.bulkBusy||S.datasetBusy||S.pendingPolygon||S.draft.length){
    toast("Finish loading, save, or cancel the current region before managing cohorts.");return false;
  }
  return true;
}
async function submitDataset(event) {
  event.preventDefault();if(!canManageDatasets())return;
  S.datasetBusy=true;updateButtons();$("submitDataset").disabled=true;
  $("datasetFormStatus").textContent="Checking the workbook, slide images, and feature files…";
  try {
    const result=await api("/api/datasets",{method:"POST",body:{...actorPayload(),name:$("datasetName").value,
      manifest:$("datasetManifest").value,predictions:$("datasetPredictions").value}});
    S.datasets=result.datasets;
    // Keep the old selection until switchDataset resets its viewer and queue.
    const previous=S.dataset;fillDatasets(previous);$("datasetSelect").value=result.dataset.id;
    $("datasetDialog").close();$("datasetForm").reset();S.datasetBusy=false;
    await switchDataset();toast("Cohort added. It will remain available after restarting the GUI.");
  }catch(error){$("datasetFormStatus").textContent=error.message;toast(error.message,true);}
  finally{S.datasetBusy=false;$("submitDataset").disabled=false;updateButtons();}
}
async function refreshDatasets() {
  if(!canManageDatasets())return;
  const previous=S.dataset;S.datasetBusy=true;updateButtons();
  try {
    const result=await api("/api/datasets/refresh",{method:"POST"});navigationCache.clear();S.datasets=result.datasets;fillDatasets(previous);
    const selected=S.dataset;S.dataset=previous;
    if(selected!==previous){S.datasetBusy=false;$("datasetSelect").value=selected;await switchDataset();}
    else {
      const cohort=$("cohortSelect").value;
      await refreshQueue();fillCohorts(cohort);filterQueue();
      const ids=new Set(S.slides.map(slide=>slide.id));S.selectedSlides=new Set([...S.selectedSlides].filter(id=>ids.has(id)));updateSelection();
      if(S.detail&&!ids.has(S.detail.id))clearSlide();
      S.datasetBusy=false;
      if(!S.detail){const first=S.filtered.find(slide=>slide.prediction_available);if(first)await loadSlide(first.id);}
    }
    toast("Cohorts refreshed.");
  }catch(error){toast(error.message,true);}
  finally{S.datasetBusy=false;updateButtons();}
}
function navigate(delta) {
  const index=S.filtered.findIndex(slide=>slide.id===(S.targetSlide||S.detail?.id)),next=S.filtered[index+delta];
  if(next)loadSlide(next.id).catch(error=>toast(error.message,true));
}
function viewerDrawers() {
  // Software WebGL competes with decoding and UI work on CPU-only desktops.
  let gl;
  try {
    gl=document.createElement("canvas").getContext("webgl2");
    if(!gl)return ["canvas","html"];
    const extension=gl.getExtension("WEBGL_debug_renderer_info");
    const renderer=extension?gl.getParameter(extension.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
    return /swiftshader|llvmpipe|softpipe|software/i.test(renderer)?["canvas","html"]:["webgl","canvas","html"];
  }catch(error){return ["canvas","html"];}
  finally{gl?.getExtension("WEBGL_lose_context")?.loseContext();}
}
function initializeViewer() {
  S.engine=new TileEngine();
  S.viewer=OpenSeadragon({id:"viewer",drawer:viewerDrawers(),showNavigationControl:false,showNavigator:true,navigatorPosition:"BOTTOM_RIGHT",
    navigatorWidth:138,navigatorHeight:94,navigatorAutoFade:true,minZoomImageRatio:.7,visibilityRatio:.5,
    constrainDuringPan:true,maxZoomPixelRatio:3,imageLoaderLimit:4,timeout:30000,
    gestureSettingsMouse:{clickToZoom:false,dblClickToZoom:true},animationTime:.25,blendTime:.1});
  for(const event of ["animation","viewport-change","resize","open"])S.viewer.addHandler(event,requestRender);
  S.viewer.addHandler("open",()=>{requestRender();});
  S.viewer.addHandler("open-failed",event=>{$("loading").hidden=true;toast("WSI could not open: "+event.message,true);});
  S.viewer.addHandler("tiled-image-drawn",event=>{
    if(event.tiles?.length&&event.tiledImage===S.viewer.world.getItemAt(0))$("loading").hidden=true;
  });
  window.addEventListener("resize",()=>{renderQueue();requestRender();});
}
function cancelSlideLoad() {
  S.loadSequence++;S.maskSequence++;S.slideController?.abort();S.cancelViewerOpen?.();
  S.loading=false;S.targetSlide=null;navigationCache.cancel();
}
function openSlideImage(detail,signal) {
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);S.viewer.removeHandler("open",opened);S.viewer.removeHandler("open-failed",failed);
      signal.removeEventListener("abort",cancel);if(S.cancelViewerOpen===cancel)S.cancelViewerOpen=null;};
    const opened=()=>{cleanup();resolve();};
    const failed=event=>{cleanup();reject(new Error(event.message||"WSI could not open"));};
    const cancel=()=>{cleanup();reject(new DOMException("Superseded slide","AbortError"));};
    const timer=setTimeout(()=>{cleanup();reject(new Error("WSI opening timed out"));},60000);
    S.cancelViewerOpen=cancel;signal.addEventListener("abort",cancel,{once:true});
    S.viewer.addHandler("open",opened);S.viewer.addHandler("open-failed",failed);
    // Metadata is already in /view; skip another DZI round trip.
    S.viewer.open({width:detail.width,height:detail.height,tileSize:detail.tile_size,tileOverlap:0,
      minLevel:Math.min(8,detail.max_level),maxLevel:detail.max_level,
      getTileUrl:(level,x,y)=>`/api/wsi/${detail.id}_files/${level}/${x}_${y}.jpg`});
  });
}
async function loadSlide(id,preserve=false,view=null) {
  if(annotations.busy()){toast('Save or undo the annotation file draft before changing slides.',true);return;}
  annotations.sequence++;S.tissueRevision='';S.fieldInfo=null;S.transferredField=null;S.mainGuide?.close();S.mainGuide=null;S.fallbackField=null;
  if(S.datasetBusy){toast("Wait for the cohort update to finish.");return;}
  if(S.saving){toast("Wait for the current correction to save.");return;}
  if(S.pendingPolygon){toast("Retry the unsaved region or press Escape before changing slides.",true);return;}
  if(S.draft.length){toast("Finish the region or press Escape before changing slides.");return;}
  const selection=view||(preserve&&S.detail?.id===id?S.viewPrediction:"active");
  const query=selection==="active"?"":selection.startsWith("model:")?"?model_id="+encodeURIComponent(selection.slice(6)):"?prediction_id="+encodeURIComponent(selection);
  const old=S.detail?.id;cancelSlideLoad();
  const sequence=S.loadSequence,controller=S.slideController=new AbortController();
  S.targetSlide=id;S.loading=true;$("loading").hidden=false;updateButtons();
  // History never delays usable images or annotation controls.
  const historyRequest=api("/api/slides/"+id+"/history?limit=100",{signal:controller.signal})
    .then(value=>({value}),error=>({error}));
  const tissueRequest=api('/api/slides/'+id+'/prediction-tissue',{signal:controller.signal})
    .then(value=>({value}),error=>({error}));
  try {
    const bytes=await navigationCache.read("/api/slides/"+id+"/view"+query,controller.signal);
    const length=new DataView(bytes).getUint32(0,true),offset=4+length;
    const detail=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,length)));
    if(sequence!==S.loadSequence)return;
    if(bytes.byteLength-offset!==detail.tiles*20)throw new Error("Prediction data is incomplete");
    S.raster?.close?.();S.raster=null;S.mask=null;S.histogramBins=null;S.history=[];S.historyNext=null;renderHistory();
    S.maskKey=null;S.detail=detail;S.viewPrediction=selection;S.data=new Float32Array(bytes,offset);S.draft=[];S.highlighted=null;S.cursor=null;
    const imageRequest=(old!==id||!preserve||!S.viewer.world.getItemAt(0))?
      openSlideImage(detail,controller.signal):Promise.resolve();
    // Observe rejection immediately while the worker initializes.
    const imageResult=imageRequest.then(()=>({}),error=>({error}));
    const initialized=await S.engine.initialize(S.data,detail);
    if(sequence!==S.loadSequence)return;
    if(initialized)S.grid=new Map(initialized.grid);else buildGrid();
    const tissue=await tissueRequest;
    if(sequence!==S.loadSequence)return;
    S.tissueGeometry=tissue.value?.geometry||null;S.tissueRevision=tissue.value?.revision||'';S.tissueStatus=tissue.value?.status||'unavailable';
    S.engine.tissue(S.tissueGeometry,S.tissueRevision);S.guideSlide=null;
    configureGrid();initializeThreshold();await updateMasks();
    annotations.load(id,controller.signal).catch(error=>{if(error.name!=='AbortError'&&sequence===S.loadSequence)toast('Annotations: '+error.message,true);});
    if($('aggregationMethod').value==='crf')await predictionGuide(controller.signal);
    if(sequence!==S.loadSequence)return;
    $("slideCohort").textContent=cohortName(detail.cohort).toUpperCase()+" · "+detail.original_split.toUpperCase();
    $("slideTitle").textContent=detail.name;
    $("slideSubtitle").textContent=fmt(detail.width)+" × "+fmt(detail.height)+" px · "+detail.tile_size_level0+" px patches · "+fmt(detail.tiles)+" tiles";
    $("revisionLabel").textContent="revision "+detail.revision;
    $("currentModelInfo").replaceChildren(node("div",{text:"Displayed model: "+(S.models.find(model=>model.id===detail.model_id)?.name||detail.model_id)}),
      node("div",{text:"Model ID: "+detail.model_id}),node("div",{text:"Prediction run: "+detail.prediction_id}),
      node("div",{text:"Prediction threshold: "+detail.prediction_threshold}),node("div",{text:"Human corrections are applied over the model's labels."}));
    fillDisplayedPredictions();
    $("displayModelBadge").textContent=(S.models.find(model=>model.id===detail.model_id)?.name||detail.model_id)+" · threshold "+detail.prediction_threshold;
    $("displayModelBadge").title="Model "+detail.model_id+" · prediction "+detail.prediction_id;
    const image=await imageResult;if(image.error)throw image.error;
    if(sequence!==S.loadSequence)return;
    $("loading").hidden=true;requestRender();
    historyRequest.then(result=>{
      if(sequence!==S.loadSequence)return;
      if(result.error){if(result.error.name!=="AbortError")toast("Slide loaded; history refresh failed: "+result.error.message,true);return;}
      // A local save may have completed before the history request returned.
      if(result.value.revision===S.detail.revision){S.history=result.value.events;S.historyNext=result.value.next_before;renderHistory();updateButtons();}
    });
    savedStatus("All completed corrections are saved");
    renderQueue();updatePosition();schedulePreferences();updateButtons();
  } catch(error) {
    if(error.name==="AbortError"||sequence!==S.loadSequence)return;
    if(sequence===S.loadSequence){$("loading").hidden=true;savedStatus("Slide could not be loaded",true);}
    throw error;
  } finally { if(sequence===S.loadSequence){S.loading=false;S.targetSlide=null;updateButtons();
    if(S.detail?.id===id)navigationCache.schedule(S.filtered,id);} }
}
function buildGrid() {
  S.grid=new Map();const size=S.detail.tile_size_level0;
  for(let index=0;index<S.detail.tiles;index++) {
    const offset=index*5,key=Math.floor((S.data[offset]+size/2)/2048)+","+Math.floor((S.data[offset+1]+size/2)/2048);
    if(!S.grid.has(key))S.grid.set(key,[]);S.grid.get(key).push(index);
  }
}
function displayedLabel(index) {
  const off=index*5,human=S.data[off+4],source=S.overlaySource||"combined";
  const prediction=Number(S.data[off+2]>=(S.liveThreshold??S.detail.prediction_threshold));
  return source==="prediction"?prediction:source==="human"?human:human!==-1?human:prediction;
}
function displayedHuman(index) {return S.overlaySource!=="prediction"&&S.data[index*5+4]!==-1;}
function revealHumanLabel(label) {
  if(label===-1)return;
  if($("overlaySource").value==="prediction")$("overlaySource").value="combined";
  $(label===-2?"excludedEnabled":label===0?"cortexEnabled":"medullaEnabled").checked=true;
  $("keepManual").checked=true;$("overlayEnabled").checked=true;
  if(Number($("opacity").value)===0){$("opacity").value=45;$("opacityValue").textContent="45%";}
}
function tilePass(index,f) {
  const off=index*5,human=f.source==="prediction"?-1:S.data[off+4],label=displayedLabel(index),config=label===0?f.cortex:f.medulla,score=S.data[off+2];
  if(label===-2)return f.excluded_enabled&&f.source!=="prediction";
  if(label<0)return false;
  if(!config.enabled)return false;
  if(human>=0&&f.keep_manual)return true;
  return config.comparison==="all"||(config.comparison==="lte"&&score<=config.threshold)||(config.comparison==="gte"&&score>=config.threshold);
}
async function updateMasks() {
  if(!S.data)return;
  S.overlaySource=$("overlaySource").value;
  const key=JSON.stringify({...filters(),...gridPayload(),...predictionOptions(),tissue:S.tissueRevision,preview:S.thresholdPreview,slide:S.detail.id,prediction:S.detail.prediction_id,revision:S.detail.revision});
  if(S.mask&&S.maskKey===key){requestRender();return;}
  if(S.engine?.worker) {
    const sequence=++S.maskSequence,id=S.detail.id;
    const result=await S.engine.filter({...filters(),...gridPayload(),...predictionOptions(),preview:S.thresholdPreview});
    if(result.fieldInfo?.score&&S.detail?.id===id)S.transferredField={key:result.fieldInfo.key,score:result.fieldInfo.score,covered:result.fieldInfo.covered};
    if(result.cancelled&&sequence===S.maskSequence&&S.detail?.id===id&&!S.engine.worker)return updateMasks();
    if(result.cancelled||sequence!==S.maskSequence||S.detail?.id!==id){result.raster?.close();return;}
    if(!result.fieldInfo&&predictionOptions().view.startsWith('aggregate')){S.engine.close();S.maskKey=null;return updateMasks();}
    S.fieldInfo=result.fieldInfo?{...(S.transferredField?.key===result.fieldInfo.key?S.transferredField:{}),...result.fieldInfo}:null;S.lastComputeMs=result.compute_ms;S.mask=result.mask;S.maskKey=key;S.detail.manual_tiles=result.manual;S.detail.excluded_tiles=result.excluded;
    S.raster?.close?.();S.raster=result.raster;S.histogramBins=result.bins;
    $("excludedCount").textContent=fmt(result.excluded);$("excludedVisible").textContent=fmt(result.excludedVisible)+" / "+fmt(result.excluded);
    $("cortexCount").textContent=fmt(result.totals[0]);$("medullaCount").textContent=fmt(result.totals[1]);$("manualCount").textContent=fmt(result.manual);
    $("cortexVisible").textContent=fmt(result.visible[0])+" / "+fmt(result.totals[0]);$("medullaVisible").textContent=fmt(result.visible[1])+" / "+fmt(result.totals[1]);
    for(const key of ["cortex","medulla"])$(key+"Threshold").disabled=$(key+"Comparison").value==="all";
    $("keepManual").disabled=S.overlaySource==="prediction";
    updateFieldStatus();requestRender();drawHistogram();return;
  }
  const f=filters(),totals=[0,0],visible=[0,0];let manual=0,excluded=0,excludedVisible=0;
  S.mask=new Uint8Array(S.detail.tiles);
  for(let index=0;index<S.detail.tiles;index++) {
    const offset=index*5,human=S.data[offset+4],label=displayedLabel(index);
    if(human>=0)manual++;
    if(human===-2)excluded++;
    if(!gridPass(index))continue;
    if(label>=0)totals[label]++;
    if(tilePass(index,f)){S.mask[index]=1;if(label>=0)visible[label]++;else if(label===-2)excludedVisible++;}
  }
  S.detail.manual_tiles=manual;
  S.detail.excluded_tiles=excluded;$("excludedCount").textContent=fmt(excluded);$("excludedVisible").textContent=fmt(excludedVisible)+" / "+fmt(excluded);
  $("cortexCount").textContent=fmt(totals[0]);$("medullaCount").textContent=fmt(totals[1]);$("manualCount").textContent=fmt(manual);
  $("cortexVisible").textContent=fmt(visible[0])+" / "+fmt(totals[0]);$("medullaVisible").textContent=fmt(visible[1])+" / "+fmt(totals[1]);
  for(const key of ["cortex","medulla"])$(key+"Threshold").disabled=$(key+"Comparison").value==="all";
  $("keepManual").disabled=$("overlaySource").value==="prediction";
  S.maskKey=key;S.raster=null;S.fieldInfo=null;
  const options={...filters(),...gridPayload(),...predictionOptions()};
  if(options.view.startsWith('aggregate')){
    const fieldKey=JSON.stringify([S.detail.fingerprint,S.detail.prediction_id,S.detail.revision,S.tissueRevision,options.method,options.sigma,options.resolution,options.grid_mode,options.source_overlap]);
    if(S.fallbackField?.key!==fieldKey)S.fallbackField={key:fieldKey,field:PredictionField.create(S.data,S.detail,options,mainTissueMask(options))};
    const field=S.fallbackField.field,canvas=document.createElement('canvas');canvas.width=field.width;canvas.height=field.height;
    const ctx=canvas.getContext('2d');let guide=null;
    if(options.method==='crf'&&S.mainGuide){ctx.drawImage(S.mainGuide,0,0,field.width,field.height);guide=ctx.getImageData(0,0,field.width,field.height).data;}
    const labels=PredictionField.classify(field,S.liveThreshold,S.thresholdPreview&&options.method==='crf'?'mean':options.method,options,guide),result=PredictionField.pixels(field,labels,options);
    ctx.putImageData(new ImageData(result.rgba,field.width,field.height),0,0);S.raster=canvas;S.fieldInfo={...field,area:result.area,labels:result.displayed,guide:!!guide};
  }
  updateFieldStatus();requestRender();drawHistogram();
}
function candidates(x0,y0,x1,y1) {
  const found=[],padding=S.detail.tile_size_level0;
  for(let x=Math.floor((x0-padding)/2048);x<=Math.floor((x1+padding)/2048);x++)for(let y=Math.floor((y0-padding)/2048);y<=Math.floor((y1+padding)/2048);y++) {
    const bin=S.grid.get(x+","+y);if(bin)found.push(...bin);
  }
  return found;
}
function requestRender() {
  if(S.renderPending)return;S.renderPending=true;
  requestAnimationFrame(()=>{S.renderPending=false;renderOverlay();drawAnnotations();});
}
function requestDraw() {
  if(S.drawPending)return;S.drawPending=true;
  requestAnimationFrame(()=>{S.drawPending=false;drawAnnotations();});
}
function renderOverlay() {
  const item=S.viewer?.world.getItemAt(0);if(!item||!S.detail||!S.data)return;
  const canvas=$("overlayCanvas"),rect=$("viewerWrap").getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)) {
    canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
    $("drawSurface").width=canvas.width;$("drawSurface").height=canvas.height;
  }
  const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  const p0=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(0,0));
  const p1=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(1,0));const scale=p1.x-p0.x;
  const size=S.detail.tile_size_level0*scale;
  canvas.style.opacity=Number($("opacity").value)/100;
  if($("overlayEnabled").checked&&S.mask) {
    const view=$('predictionView').value,aggregate=view.startsWith('aggregate');
    const ids=candidates(-p0.x/scale,-p0.y/scale,(rect.width-p0.x)/scale,(rect.height-p0.y)/scale);
    if(aggregate&&S.raster){ctx.imageSmoothingEnabled=false;const f=S.fieldInfo;ctx.drawImage(S.raster,p0.x,p0.y,(f?.width*f?.step||S.detail.width)*scale,(f?.height*f?.step||S.detail.height)*scale);}
    ctx.save();
    if(S.tissueGeometry&&(!aggregate||$('tileBoundaries').checked)){ctx.beginPath();const polygons=S.tissueGeometry.type==='Polygon'?[S.tissueGeometry.coordinates]:S.tissueGeometry.type==='MultiPolygon'?S.tissueGeometry.coordinates:[];for(const rings of polygons)for(const ring of rings){ring.forEach((p,i)=>i?ctx.lineTo(p0.x+p[0]*scale,p0.y+p[1]*scale):ctx.moveTo(p0.x+p[0]*scale,p0.y+p[1]*scale));ctx.closePath();}ctx.clip('evenodd');}
    if(!aggregate||$('tileBoundaries').checked)for(const index of ids) {
      if(!S.mask[index])continue;const off=index*5,x=p0.x+S.data[off]*scale,y=p0.y+S.data[off+1]*scale;
      if(x+size<0||y+size<0||x>rect.width||y>rect.height)continue;
      const label=displayedLabel(index),human=displayedHuman(index),rgb=PredictionField.color(S.data[off+2],S.detail.score_range);
      ctx.strokeStyle=human?'#eefff5':labelColor(label);ctx.lineWidth=human?1.5:1;
      if(!aggregate&&view==='raw-score'){ctx.globalAlpha=.12;ctx.fillStyle=human?labelColor(label):'rgb('+rgb.join(',')+')';ctx.fillRect(x,y,size,size);ctx.globalAlpha=1;ctx.strokeStyle='rgb('+rgb.join(',')+')';}
      ctx.strokeRect(x,y,size,size);
    }
    ctx.restore();
  }
}
function drawAnnotations() {
  const canvas=$("drawSurface"),ctx=canvas.getContext("2d"),rect=$("viewerWrap").getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  const item=S.viewer?.world.getItemAt(0);
  if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);if(!item||!S.detail)return;
  const p0=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(0,0));
  const p1=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(1,0)),scale=p1.x-p0.x;
  function polygon(points,color,dash=false) {
    if(!points.length)return;
    ctx.beginPath();points.forEach((point,index)=>{const x=p0.x+point[0]*scale,y=p0.y+point[1]*scale;index?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    if(points.length>2)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash?[6,4]:[]);ctx.stroke();
    if(points.length>2){ctx.fillStyle=color+"18";ctx.fill();}ctx.setLineDash([]);
  }
  annotations.draw(ctx,p0,scale);
  if(S.highlighted)polygon(S.highlighted,"#ffffff",true);
  if(S.pendingPolygon)polygon(S.pendingPolygon,"#ffc478",true);
  if(S.draft.length) {
    polygon(S.mode==="polygon"&&S.cursor?[...S.draft,S.cursor]:S.draft,labelColor(S.label));
    if(S.mode!=="freehand")for(const point of S.draft){ctx.beginPath();ctx.arc(p0.x+point[0]*scale,p0.y+point[1]*scale,3,0,2*Math.PI);ctx.fillStyle="#fff";ctx.fill();}
  }
}
function drawHistogram() {
  if(!S.data||$('histogram').closest('[hidden]'))return;
  const canvas=$("histogram"),ctx=canvas.getContext("2d"),width=canvas.width,height=canvas.height;
  ctx.clearRect(0,0,width,height);const min=S.detail.score_range[0],max=S.detail.score_range[1],span=Math.max(max-min,.01),bins=[new Uint32Array(60),new Uint32Array(60)];
  if(S.histogramBins){bins[0]=S.histogramBins[0];bins[1]=S.histogramBins[1];}
  else for(let index=0;index<S.detail.tiles;index++) {const off=index*5,label=displayedLabel(index);if(label>=0)bins[label][Math.min(59,Math.max(0,Math.floor((S.data[off+2]-min)/span*60)))]++;}
  const maxHeight=Math.max(...bins[0],...bins[1],1);
  for(let label=0;label<2;label++)for(let index=0;index<60;index++) {
    const bar=Math.log1p(bins[label][index])/Math.log1p(maxHeight)*(height-12);
    ctx.fillStyle=colors[label];ctx.globalAlpha=.75;ctx.fillRect(index*width/60,height-bar,width/60-1,bar);
  }
  ctx.globalAlpha=1;const f=filters();
  for(const [label,key] of [[0,"cortex"],[1,"medulla"]])if(f[key].comparison!=="all") {
    const x=(f[key].threshold-min)/span*width;ctx.beginPath();ctx.moveTo(x,5);ctx.lineTo(x,height);ctx.strokeStyle=colors[label];ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
  }
  $("scoreMin").textContent=min.toFixed(2);$("scoreMax").textContent=max.toFixed(2);
}
function imagePoint(event) {
  const item=S.viewer?.world.getItemAt(0);if(!item||!S.detail)return null;
  const rect=$("viewerWrap").getBoundingClientRect();
  const point=item.viewerElementToImageCoordinates(new OpenSeadragon.Point(event.clientX-rect.left,event.clientY-rect.top));
  return [Math.max(0,Math.min(S.detail.width,point.x)),Math.max(0,Math.min(S.detail.height,point.y))];
}
function setMode(mode) {
  if(annotations.editing&&mode!=='pan'){toast('Apply or cancel vertex editing first.');return;}
  if(S.saving||S.pendingPolygon){toast("Finish saving the current region first.");return;}
  S.mode=mode;S.draft=[];S.drag=null;S.cursor=null;
  document.querySelectorAll("[data-tool]").forEach(button=>button.classList.toggle("active",button.dataset.tool===mode));
  $("drawSurface").classList.toggle("drawing",mode!=="pan");S.viewer.setMouseNavEnabled(mode==="pan");updateModeChip();requestRender();
}
function setLabel(label) {
  S.label=label;document.querySelectorAll("[data-label]").forEach(button=>button.classList.toggle("selected",Number(button.dataset.label)===label));
  updateModeChip();requestRender();
}
function updateModeChip() {
  const chip=$("modeChip");chip.className="viewer-chip"+(S.mode==="pan"?"":S.label===0?" label-cortex":S.label===1?" label-medulla":"");
  chip.replaceChildren(node("span",{class:"status-dot"}),document.createTextNode(S.mode==="pan"?"Pan · scroll to zoom":
    ({rectangle:"Rectangle",polygon:"Polygon",freehand:"Freehand"}[S.mode])+" · "+(S.label===-2?"exclude tiles":S.label===0?"label cortex":S.label===1?"label medulla":"restore prediction")+(S.mode==="polygon"?" · Enter to finish":S.mode==="freehand"?" · release to save":"")));
}
function rectangle(a,b) {return [[Math.min(a[0],b[0]),Math.min(a[1],b[1])],[Math.max(a[0],b[0]),Math.min(a[1],b[1])],[Math.max(a[0],b[0]),Math.max(a[1],b[1])],[Math.min(a[0],b[0]),Math.max(a[1],b[1])]];}
async function finishPolygon() {
  if(S.mode!=="polygon"||S.draft.length<3)return;
  const points=S.draft.filter((point,index,list)=>index===0||Math.hypot(point[0]-list[index-1][0],point[1]-list[index-1][1])>.1);
  if(points.length<3)return;
  await saveRegion(points);
}
async function saveRegion(points) {
  if($('annotationTarget').value!=='corrections'){annotations.add(points);return;}
  if(S.saving||S.loading||!S.detail)return;
  if(!$("reviewer").value.trim()){toast("Enter your reviewer name before annotating.",true);return;}
  S.saving=true;S.pendingPolygon=points;S.draft=[];S.cursor=null;$("retrySave").hidden=true;
  savedStatus("Saving annotation changes…");updateButtons();requestRender();
  try {
    const result=await api("/api/slides/"+S.detail.id+"/corrections",{method:"POST",body:{...actorPayload(),
      ...gridPayload(),action:S.label===-1?"clear":"label",label:S.label,polygon:points,revision:S.detail.revision,
      fingerprint:S.detail.fingerprint,prediction_id:S.detail.prediction_id,
      view_prediction_id:S.detail.prediction_id,shape:S.mode,
      display_threshold:S.liveThreshold,
      visible_only:$("visibleOnly").checked,filters:filters(),note:$("regionNote").value}});
    navigationCache.invalidateSlide(S.detail.id);
    for(const index of result.tile_ids)S.data[index*5+4]=result.label;
    S.engine?.labels(result.tile_ids,result.label);
    S.detail.revision=result.revision;S.pendingPolygon=null;S.highlighted=points;revealHumanLabel(result.label);await updateMasks();schedulePreferences();
    if(S.detail.reviewed_passes)delete S.detail.reviewed_passes[S.pass];
    const slide=S.slides.find(row=>row.id===S.detail.id);if(slide){slide.manual_tiles=S.detail.manual_tiles;slide.excluded_tiles=S.detail.excluded_tiles;slide.reviewed=false;}
    $("revisionLabel").textContent="revision "+result.revision;
    savedStatus(fmt(result.tile_ids.length)+" tiles saved · "+new Date(result.saved_at).toLocaleTimeString());
    toast(fmt(result.tile_ids.length)+" tiles "+(result.label===-2?"excluded from retraining":result.label===-1?"restored to the prediction":"labelled "+labelName(result.label)));
    const savedId=S.detail.id,savedRevision=result.revision;S.saving=false;updateButtons();updateProgress();filterQueue();
    api("/api/slides/"+savedId+"/history?limit=100").then(history=>{if(S.detail?.id===savedId&&S.detail.revision===savedRevision){S.history=history.events;S.historyNext=history.next_before;renderHistory();updateButtons();}}).catch(error=>toast("Labels saved; history refresh failed: "+error.message,true));
    refreshTrainingSummary().catch(()=>{});
  } catch(error) {
    if(error.status===409) {
      S.pendingPolygon=null;S.saving=false;
      await loadSlide(S.detail.id,true).catch(()=>{});toast(error.message+" Your region was not applied; draw it again after reload.",true);
    } else {$("retrySave").hidden=false;savedStatus("Region not saved · retry or press Escape",true);toast(error.message,true);}
  } finally {S.saving=false;updateButtons();requestRender();}
}
function updateButtons() {
  const blocked=S.saving||S.loading||S.datasetBusy||!S.detail;
  $('undoButton').title=$('annotationTarget').value==='corrections'?'Undo last tile correction':'Undo file edit';
  const datasetBlocked=S.saving||S.bulkBusy||S.datasetBusy||Boolean(S.pendingPolygon)||S.draft.length>0;
  $("datasetSelect").disabled=datasetBlocked;$("addDatasetButton").disabled=datasetBlocked||S.loading||S.queueBusy;$("refreshDatasetsButton").disabled=datasetBlocked||S.loading||S.queueBusy;
  $("predictButton").disabled=blocked||S.jobs.some(job=>job.kind==="predict"&&job.slide_id===S.detail?.id&&["preparing","submitting","pending","running","submitted","unknown"].includes(job.status));
  $("markReviewed").disabled=blocked||Boolean(S.pendingPolygon)||S.draft.length>0;$("undoButton").disabled=blocked||!S.history.some(event=>["label","clear"].includes(event.action)&&!event.undone);
  $("doneReview").disabled=blocked||Boolean(S.pendingPolygon)||S.draft.length>0;
  $("doneReview").textContent=S.detail?.reviewed_passes?.[S.pass]?"Reviewed ✓":"Done reviewing";
  $("bulkButton").disabled=blocked||S.queueBusy||Boolean(S.pendingPolygon)||S.draft.length>0||S.bulkBusy;
  $("allCortex").disabled=$("bulkButton").disabled;$("allMedulla").disabled=$("bulkButton").disabled;
  $("displayPredictionSelect").disabled=blocked||Boolean(S.pendingPolygon)||S.draft.length>0;
  for(const id of ["gridMode","sourceOverlap"])$(id).disabled=blocked||Boolean(S.pendingPolygon)||S.draft.length>0;
  const targets=bulkSlides($("predictionScope").value).map(slide=>slide.id);
  $("predictButton").disabled=S.jobSubmitting||S.saving||S.loading||S.datasetBusy||!targets.length||S.jobs.some(job=>job.kind==="predict"&&["preparing","submitting","pending","running","submitted","unknown"].includes(job.status)&&targets.some(id=>job.slide_id===id||job.slide_ids?.includes(id)));
  document.querySelectorAll("[data-tool],[data-label]").forEach(button=>button.disabled=S.saving||S.loading||S.datasetBusy);
  if(S.computeEnabled===false){for(const id of ["predictButton","trainButton","trainFromTab","submitTraining","executorSelect","trainExecutorSelect"])$(id).disabled=true;}
  updatePosition();
}
async function undoEvent(id) {
  if(S.saving||S.loading||!S.detail)return;
  if(S.pendingPolygon){toast("Retry or cancel the unsaved region before undoing.");return;}
  S.saving=true;updateButtons();
  try {await api("/api/slides/"+S.detail.id+"/undo",{method:"POST",body:{...actorPayload(),revision:S.detail.revision,target_id:id}});
    navigationCache.invalidateSlide(S.detail.id);
    S.saving=false;await loadSlide(S.detail.id,true);await refreshQueue();toast("Correction undone. Its history is retained.");}
  catch(error){toast(error.message,true);}finally{S.saving=false;updateButtons();}
}
function undoLatest() {if($('annotationTarget').value!=='corrections'){annotations.undo($('annotationTarget').value);return;}const event=S.history.find(item=>["label","clear"].includes(item.action)&&!item.undone);if(event)undoEvent(event.id);}
async function reviewedNext(advance=true) {
  if(S.saving||S.loading||S.pendingPolygon||S.draft.length||!S.detail)return;
  const oldId=S.detail.id,index=S.filtered.findIndex(row=>row.id===oldId),next=S.filtered[index+1];
  S.saving=true;updateButtons();
  try {const result=await api("/api/slides/"+oldId+"/review",{method:"POST",body:{...actorPayload(),revision:S.detail.revision,view_prediction_id:S.detail.prediction_id}});
    navigationCache.invalidateSlide(oldId);
    S.detail.revision=result.revision;S.saving=false;await refreshQueue();
    if(!advance){await loadSlide(oldId,true);toast("Slide marked reviewed for "+S.passes.find(p=>p.id===S.pass)?.name);return;}
    const remaining=next||S.filtered.find(row=>!row.reviewed&&row.id!==oldId);
    if(remaining)await loadSlide(remaining.id);else{toast("Review saved. No further slides in this queue.");await loadSlide(oldId,true);}
  }catch(error){toast(error.message,true);}finally{S.saving=false;updateButtons();}
}
function renderHistory() {
  if($('historyList').closest('[hidden]'))return;
  const pass=$("historyPass").value,events=S.history.filter(event=>pass==="all"||event.pass_id===pass);
  if(!events.length){$("historyList").replaceChildren(node("div",{class:"empty-state",text:"No corrections in this view yet. Completed regions are saved here automatically."}));return;}
  const rows=events.map(event=>{
    const name=event.action==="label"?(event.label===-2?"Excluded":event.label===0?"Cortex":"Medulla"):event.action==="clear"?"Restored prediction":event.action==="undo"?"Undo recorded":"Slide reviewed";
    const row=node("div",{class:"history-item"+(event.undone?" undone":""),onclick:()=>focusRegion(event.polygon)});
    const title=node("div",{class:"history-title"},[node("span",{class:"legend-dot "+(event.label===-2?"excluded":event.label===0?"cortex":event.label===1?"medulla":"corrected")}),
      node("span",{text:name+(event.tile_count?" · "+fmt(event.tile_count)+" tiles":"")})]);
    row.append(title,node("div",{class:"history-meta",text:event.reviewer+" · "+new Date(event.created_at).toLocaleString()}));
    if(event.shape)row.append(node("div",{class:"history-meta",text:event.shape.replaceAll("_"," ")+" · "+(S.models.find(m=>m.id===event.model_id)?.name||event.model_id||"")}));
    if(event.note)row.append(node("div",{class:"history-note",text:event.note}));
    const footer=node("div",{class:"history-footer"},[node("span",{class:"history-pass-tag",text:S.passes.find(item=>item.id===event.pass_id)?.name||event.pass_id})]);
    if(event.undone)footer.append(node("span",{class:"history-meta",text:"Undone"}));
    else if(["label","clear"].includes(event.action))footer.append(node("button",{class:"history-undo",text:"Undo",onclick:click=>{click.stopPropagation();undoEvent(event.id);}}));
    row.append(footer);return row;
  });$("historyList").replaceChildren(...rows);
  if(S.historyNext)$("historyList").append(node("button",{class:"button",text:"Load older history",onclick:async event=>{
    const id=S.detail.id,revision=S.detail.revision;event.target.disabled=true;
    try {
      const page=await api(`/api/slides/${id}/history?limit=100&before=${S.historyNext}`);
      if(S.detail?.id!==id||S.detail.revision!==revision)return;
      if(page.revision!==revision){toast("History changed. Reload the slide to see the latest events.");return;}
      S.history.push(...page.events);S.historyNext=page.next_before;renderHistory();updateButtons();
    }catch(error){toast(error.message,true);}
    finally{event.target.disabled=false;}
  }}));
}
function focusRegion(points) {
  if(!points?.length)return;S.highlighted=points;
  const xs=points.map(point=>point[0]),ys=points.map(point=>point[1]),x=Math.min(...xs),y=Math.min(...ys);
  const width=Math.max(...xs)-x,height=Math.max(...ys)-y,padding=Math.max(width,height)*.15;
  const item=S.viewer.world.getItemAt(0);if(item)S.viewer.viewport.fitBounds(item.imageToViewportRectangle(x-padding,y-padding,width+2*padding,height+2*padding),false);requestRender();
}
function selectTab(tab) {
  document.querySelectorAll("[data-tab]").forEach(button=>{button.classList.toggle("active",button.dataset.tab===tab);button.setAttribute("aria-selected",button.dataset.tab===tab?"true":"false");});
  document.querySelectorAll(".tab-content").forEach(panel=>panel.classList.toggle("active",panel.id==="tab-"+tab));
  if(tab==='history')renderHistory();if(tab==='annotate')annotations.render();
  if(tab==="model")refreshTrainingSummary().catch(error=>toast(error.message,true));
}
async function refreshTrainingSummary() {
  const plan=await api("/api/training-plan?budget="+Number($("tileBudget").value));
  $("trainingSummary").replaceChildren(node("div",{text:fmt(plan.corrected_tiles)+" confirmed tiles across "+fmt(plan.corrected_slides)+" slides"}),
    node("div",{text:fmt(plan.excluded_tiles)+" excluded tiles omitted from retraining"}),node("div",{text:fmt(plan.original_train_slides)+" original training slides included"}));return plan;
}
async function openTraining() {$("trainExecutorSelect").value=S.executor;$("trainDialog").showModal();await updateTrainPlan();}
async function updateTrainPlan() {
  $("submitTraining").disabled=true;
  try {
    const plan=await refreshTrainingSummary();
    const pairs=[["Original training slides",fmt(plan.original_train_slides)],
      ["Original training tiles",plan.original_train_tiles?fmt(plan.original_train_tiles):"Read from original labels"],
      ["Human labelled slides",fmt(plan.corrected_slides)],["Confirmed human tiles",fmt(plan.corrected_tiles)],
      ["Excluded human tiles",fmt(plan.excluded_tiles)],
      ["Additional human tiles selected",fmt(plan.additional_tiles_selected)],["Untouched test slides",fmt(plan.untouched_test_slides)],
      ["Test slides excluded from evaluation",fmt(plan.excluded_test_slides)]];
    $("trainPlan").replaceChildren(...pairs.map(([label,value])=>node("div",{class:"plan-row"},[node("span",{text:label}),node("strong",{text:value})])));
    $("submitTraining").disabled=(plan.corrected_tiles===0&&plan.excluded_tiles===0)||S.jobs.some(job=>job.kind==="train"&&["preparing","submitting","pending","running","submitted","unknown"].includes(job.status));
  }catch(error){$("trainPlan").textContent=error.message;toast(error.message,true);}
}
async function submitJob(kind) {
  if(!$("reviewer").value.trim()){toast("Enter a reviewer name first.",true);return;}
  const threshold=Number($("predictionThreshold").value);if(!Number.isFinite(threshold)){toast("Enter a finite prediction threshold.",true);return;}
  if(S.jobSubmitting)return;S.jobSubmitting=true;
  const button=$(kind==="train"?"submitTraining":"predictButton"),old=button.textContent;button.disabled=true;button.textContent=kind==="train"?"Freezing correction snapshot…":"Submitting prediction…";
  try {const job=await api("/api/jobs",{method:"POST",body:{...actorPayload(),kind,slide_ids:kind==="predict"?bulkSlides($("predictionScope").value).map(slide=>slide.id):undefined,
    model_id:$("modelSelect").value,threshold,budget:Number($("tileBudget").value),executor:S.executor}});
    S.jobs.unshift(job);S.knownJobs.set(job.id,job.status);renderJobs();updateButtons();
    if(kind==="train")$("trainDialog").close();toast((kind==="train"?"Retraining":"Prediction")+" submitted"+(job.slurm_id?" · GPU job "+job.slurm_id:""));
  }catch(error){toast(error.message,true);}finally{S.jobSubmitting=false;button.textContent=old;updateButtons();}
}
function renderJobs() {
  const active=S.jobs.filter(job=>["preparing","submitting","pending","running","submitted","unknown"].includes(job.status));$("jobsCount").textContent=active.length;
  const latest=active[0]||S.jobs[0];$("jobStatusLink").textContent=latest?(latest.kind==="train"?"Retraining":"Prediction")+" · "+latest.status+(latest.slurm_id?" · job "+latest.slurm_id:""):S.computeEnabled?'GPU jobs run in the background':'CPU review · file drafts need Save';
  if(!S.jobs.length){$("jobList").replaceChildren(node("div",{class:"empty-state",text:"No review jobs yet. Predict a slide or retrain after annotating."}));return;}
  $("jobList").replaceChildren(...S.jobs.map(job=>node("div",{class:"job-row",onclick:()=>showLogs(job.id)},[
    node("div",{class:"job-row-title"},[node("span",{text:job.kind==="train"?"Retrain from human review":job.slide_name}),node("span",{class:"job-badge "+job.status,text:job.status+(job.total?" · "+(job.completed||0)+"/"+job.total+" · "+(job.failed||0)+" failed":"")})]),
    node("div",{class:"job-row-meta",text:"Threshold "+job.threshold+" · "+new Date(job.created_at).toLocaleString()+(job.slurm_id?" · Slurm "+job.slurm_id:"")}),
    ...(job.error?[node("div",{class:"job-row-meta",text:job.error})]:[])])));
}
async function showLogs(id) {S.selectedJob=id;const logs=await api("/api/jobs/"+id+"/logs");$("jobLogs").hidden=false;$("jobLogs").textContent=(logs.stdout||"Waiting for worker output…")+(logs.stderr?"\n\nErrors:\n"+logs.stderr:"");}
async function pollJobs() {
  if(S.pollBusy||document.hidden)return;S.pollBusy=true;
  try {const jobs=await api("/api/jobs");
    for(const job of jobs) {
      const previous=S.knownJobs.get(job.id);S.knownJobs.set(job.id,job.status);
      if(previous&&previous!==job.status&&["completed","completed_with_errors"].includes(job.status)) {
        const boot=await api("/api/bootstrap");S.models=boot.models;fillModels(job.kind==="train"?job.id:null);
        toast(job.kind==="train"?"New reviewed SVC is ready. Select it to predict a slide.":"New predictions are ready"+(job.failed?" · "+job.failed+" slides failed; inspect Jobs":"."),Boolean(job.failed));
        if(job.kind==="predict")S.refreshQueue=true;
        if(job.kind==="predict"&&(job.slide_id===S.detail?.id||job.slide_ids?.includes(S.detail?.id)))S.refreshPrediction=true;
      }
      if(previous&&previous!==job.status&&job.status==="failed")toast("GPU job failed: "+(job.error||"Open Jobs to inspect the logs."),true);
    }
    S.jobs=jobs;renderJobs();updateButtons();
    if(S.refreshQueue&&!S.queueBusy&&!S.datasetBusy){S.refreshQueue=false;await refreshQueue();}
    await pollBulk();
    if(S.refreshPrediction&&!S.saving&&!S.pendingPolygon&&!S.draft.length&&!S.loading&&!S.datasetBusy&&S.detail){S.refreshPrediction=false;await loadSlide(S.detail.id,true);}
    if($("jobsDialog").open&&S.selectedJob)await showLogs(S.selectedJob);
  }catch(error){console.warn("Job polling",error.message);}finally{S.pollBusy=false;}
}
function fillDisplayedPredictions() {
  const options=[node("option",{value:"active",text:"Latest prediction for this slide"})];
  for(const version of S.detail.prediction_versions||[]) {
    const model=(S.models.find(m=>m.id===version.model_id)?.name||version.model_id)+" · "+version.model_id;
    const when=version.created_at?new Date(version.created_at).toLocaleString():"Initial predictions";
    options.push(node("option",{value:version.job_id,text:model+" · "+when+" · threshold "+version.threshold+" · "+version.job_id}));
  }
  for(const model of S.models.filter(m=>m.id!=="baseline")) {
    const available=S.detail.prediction_versions.some(v=>v.model_id===model.id);
    const option=node("option",{value:"model:"+model.id,text:model.name+" · latest saved"+(available?"":" · run this model first")});
    option.disabled=!available;options.push(option);
  }
  $("displayPredictionSelect").replaceChildren(...options);$("displayPredictionSelect").value=S.viewPrediction;
}
function setupExecutors(data) {
  S.device=data.device;S.localGpu=data.local_gpu_available;S.computeEnabled=data.compute_enabled!==false;
  $("jobStatusLink").textContent=S.computeEnabled?"GPU jobs run in the background":"CPU review · autosave enabled";
  for(const id of ["executorSelect","trainExecutorSelect"]) {
    const local=node("option",{value:"local",text:"Use this session's allocated GPU"});local.disabled=!S.localGpu;
    $(id).replaceChildren(local,node("option",{value:"slurm",text:"Submit a separate Slurm GPU job"}));
  }
  let choice=data.executor;
  if(data.preferences.compute_device===S.device&&["local","slurm"].includes(data.preferences.executor))choice=data.preferences.executor;
  if(choice==="local"&&!S.localGpu)choice="slurm";
  setExecutor(choice,false);
}
function setExecutor(choice,persist=true) {
  if(choice==="local"&&!S.localGpu){toast("Start a GPU allocation to use the session GPU.",true);choice="slurm";}
  S.executor=choice;$("executorSelect").value=choice;$("trainExecutorSelect").value=choice;
  $("executorHint").textContent=choice==="local"?"Uses the GPU and memory already allocated to this session. No new Slurm job is submitted.":"Submits a separate GPU job under the configured Slurm account. The CPU viewer stays available while it queues and runs.";
  if(!S.computeEnabled){
    for(const id of ['trainButton','jobsButton'])$(id).hidden=true;document.querySelector('[data-tab="model"]').hidden=true;
    $("executorHint").textContent="GPU prediction and retraining are disabled in this deployment. Saved predictions and annotation tools remain available.";
    $("jobStatusLink").textContent="CPU review · annotations autosave";
  }
  if(persist)schedulePreferences();
}
function updateSelection() {$("selectedCount").textContent=fmt(S.selectedSlides.size)+" selected";updatePredictionPlan();updateButtons();}
function bulkSlides(scope) {
  if(scope==="current")return S.detail?[S.slides.find(s=>s.id===S.detail.id)||S.detail]:[];
  return scope==="selected"?S.slides.filter(s=>S.selectedSlides.has(s.id)):[...S.filtered];
}
async function updateBulkPlan() {
  const generation=S.bulkPlanGeneration=(S.bulkPlanGeneration||0)+1;
  $("submitBulk").disabled=true;S.bulkPlan=null;$("bulkPlan").textContent="Reading selected-grid counts…";
  const slides=bulkSlides($("bulkScope").value);
  try {
    const plan=await api("/api/bulk-labels/plan",{method:"POST",body:{...gridPayload(),slides:slides.map(s=>({id:s.id,prediction_id:s.id===S.detail?.id?S.detail.prediction_id:undefined}))}});
    if(generation!==S.bulkPlanGeneration)return;
    S.bulkPlan=plan.slides;
    $("bulkPlan").replaceChildren(...[["Slides",fmt(plan.slides.length)],["Selected grid tiles",fmt(plan.tiles)],["Grid",$("gridMode").selectedOptions[0].textContent],["Label",$("bulkLabel").value==="0"?"Cortex":"Medulla"]].map(([a,b])=>node("div",{class:"plan-row"},[node("span",{text:a}),node("strong",{text:b})])));
    $("submitBulk").disabled=!plan.slides.length||S.bulkBusy;
    $("submitBulk").textContent="Label "+fmt(plan.slides.length)+" slides";
  }catch(error){if(generation===S.bulkPlanGeneration){$("bulkPlan").textContent=error.message;toast(error.message,true);}}
}
function openBulk(scope="current",label=S.label) {
  if(S.saving||S.loading||S.pendingPolygon||S.draft.length||S.queueBusy)return;
  $("bulkScope").value=scope;$("bulkLabel").value=String(label<0?0:label);
  $("bulkNote").value=$("regionNote").value;$("bulkStatus").textContent="";
  updateBulkPlan();$("bulkDialog").showModal();
}
async function submitBulk() {
  if(!S.bulkPlan?.length||S.bulkBusy)return;
  if(!$("reviewer").value.trim()){toast("Enter your reviewer name first.",true);return;}
  $("submitBulk").disabled=true;
  try {
    const batch=await api("/api/bulk-labels",{method:"POST",body:{...actorPayload(),...gridPayload(),slides:S.bulkPlan,
      label:Number($("bulkLabel").value),note:$("bulkNote").value,model_id:"active",scope:$("bulkScope").value}});
    S.bulkBusy=true;S.knownBulk.set(batch.id,batch.status);$("bulkStatus").textContent="Saving annotations for "+fmt(batch.total)+" slides…";
    toast("Whole-slide labels are saving in the background. Each slide can be undone from History.");
    await pollBulk();updateButtons();
  }catch(error){toast(error.message,true);$("bulkStatus").textContent=error.message;$("submitBulk").disabled=false;}
}
async function pollBulk() {
  const batches=await api("/api/bulk-labels");S.bulkJobs=batches;
  S.bulkBusy=batches.some(b=>["queued","running"].includes(b.status));
  let changed=false;
  for(const batch of batches) {
    const previous=S.knownBulk.get(batch.id);S.knownBulk.set(batch.id,batch.status);
    if(previous&&previous!==batch.status&&!["queued","running"].includes(batch.status)) {
      changed=true;
      toast(fmt(batch.completed)+" slides labelled · "+fmt(batch.tiles)+" tiles"+(batch.failed?" · "+fmt(batch.failed)+" skipped":""),batch.failed>0||batch.status==="interrupted");
      if(batch.results.some(r=>r.slide_id===S.detail?.id&&r.status==="saved")){revealHumanLabel(batch.label);S.refreshPrediction=true;schedulePreferences();}
    }
  }
  const latest=[...batches].sort((a,b)=>b.created_at.localeCompare(a.created_at))[0];
  if(latest)$("bulkStatus").textContent=latest.status+" · "+latest.completed+" / "+latest.total+" slides · "+latest.failed+" skipped"+
    (latest.failed?" · "+latest.results.filter(r=>r.error).map(r=>r.error).join("; "):"");
  if(changed){await refreshQueue();refreshTrainingSummary().catch(()=>{});}
  if($("bulkDialog").open)$("submitBulk").disabled=S.bulkBusy||!S.bulkPlan?.length;
  updateButtons();
}
function simplifyFreehand(points,tolerance) {
  // Iterative Douglas–Peucker keeps a bounded audit polygon and avoids recursion on long strokes.
  const ring=[...points,points[0]],keep=new Uint8Array(ring.length),stack=[[0,ring.length-1]];
  keep[0]=keep[ring.length-1]=1;
  while(stack.length) {
    const [first,last]=stack.pop(),a=ring[first],b=ring[last],dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
    let max=tolerance*tolerance,index=-1;
    for(let i=first+1;i<last;i++) {
      const p=ring[i],t=length?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)):0;
      const distance=(p[0]-a[0]-t*dx)**2+(p[1]-a[1]-t*dy)**2;
      if(distance>max){max=distance;index=i;}
    }
    if(index>=0){keep[index]=1;stack.push([first,index],[index,last]);}
  }
  return ring.filter((p,i)=>keep[i]&&i<ring.length-1);
}
function appendFreehand(event,force=false) {
  const point=imagePoint(event);if(!point)return;
  if(!force&&S.freehandScreen&&Math.hypot(event.clientX-S.freehandScreen[0],event.clientY-S.freehandScreen[1])<2)return;
  S.freehandScreen=[event.clientX,event.clientY];S.freehandRawCount++;
  if(S.draft.length>=8192)S.draft=S.draft.filter((_,i)=>i%2===0);
  S.draft.push(point);
}
function finishFreehand() {
  if(S.draft.length<3){S.draft=[];requestRender();return;}
  const item=S.viewer.world.getItemAt(0),a=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(0,0)),
    b=item.imageToViewerElementCoordinates(new OpenSeadragon.Point(1,0));
  let tolerance=1/Math.max(Math.abs(b.x-a.x),.000001),points=simplifyFreehand(S.draft,tolerance);
  while(points.length>512){tolerance*=1.5;points=simplifyFreehand(S.draft,tolerance);}
  if(points.length<3){S.draft=[];toast("Draw a larger closed freehand region.");requestRender();return;}
  saveRegion(points);
}
function inspectHover(event) {
  if(!S.detail||!S.data)return;const point=imagePoint(event);if(!point)return;
  const size=S.detail.tile_size_level0,ids=candidates(point[0]-size,point[1]-size,point[0]+size,point[1]+size).filter(i=>{const o=i*5;return gridPass(i)&&point[0]>=S.data[o]&&point[0]<S.data[o]+size&&point[1]>=S.data[o+1]&&point[1]<S.data[o+1]+size;});
  const f=S.fieldInfo,x=f?Math.floor(point[0]/f.step):-1,y=f?Math.floor(point[1]/f.step):-1,p=f&&x>=0&&y>=0&&x<f.width&&y<f.height?y*f.width+x:-1;
  if(!ids.length||(f&&(p<0||!f.covered?.[p]&& !f.count?.[p]))){$('hoverInfo').textContent='Uncovered / outside selected tissue';return;}
  let text=ids.length+' contributing tiles · '+ids.slice(0,12).map(i=>i+': '+S.data[i*5+2].toFixed(3)).join(', ')+(ids.length>12?'…':'');
  if(f&&p>=0)text+=' · aggregate '+f.score[p].toFixed(3)+' · displayed '+({1:'medulla',0:'cortex','-2':'excluded','-1':'hidden'}[f.labels[p]]);
  else text+=' · ≥ '+S.liveThreshold.toFixed(2)+' → medulla';$('hoverInfo').textContent=text;
}
function predictionOptions(){
  const ids=['predictionView','aggregationMethod','gaussianSigma','rasterResolution','crfStrength','crfRadius','crfIterations'];
  return {view:$('predictionView').value,method:$('aggregationMethod').value,sigma:Number($('gaussianSigma').value),resolution:Number($('rasterResolution').value),strength:Number($('crfStrength').value),radius:Number($('crfRadius').value),iterations:Number($('crfIterations').value),classification_threshold:S.liveThreshold??S.detail?.prediction_threshold??0,boundaries:$('tileBoundaries').checked,controls:Object.fromEntries(ids.map(id=>[id,$(id).value]))};
}
function initializeThreshold(){S.liveThreshold=S.detail.prediction_threshold;S.thresholdPreview=false;
  const [lo,hi]=S.detail.score_range;for(const id of ['liveThreshold','liveThresholdNumber']){$(id).min=Math.min(lo,S.liveThreshold)-.5;$(id).max=Math.max(hi,S.liveThreshold)+.5;$(id).value=S.liveThreshold;}}
let thresholdFrame=false,refineTimer;
function liveThreshold(value,preview=true){if(!Number.isFinite(value))return;S.liveThreshold=value;S.thresholdPreview=preview;
  if(preview&&$('aggregationMethod').value==='crf')$('thresholdStatus').textContent='Preview only · ≥ threshold: medulla · refinement pending';
  $('liveThreshold').value=value;$('liveThresholdNumber').value=value;
  if(!thresholdFrame){thresholdFrame=true;requestAnimationFrame(()=>{thresholdFrame=false;updateMasks().catch(e=>toast(e.message,true));});}
  clearTimeout(refineTimer);if(preview&&$('aggregationMethod').value==='crf')refineTimer=setTimeout(()=>{S.thresholdPreview=false;updateMasks().catch(e=>toast(e.message,true));},220);
}
function updateFieldStatus(){const options=predictionOptions(),f=S.fieldInfo;
  $('heatmapLegend').hidden=!options.view.includes('score');$('heatmapMin').textContent=S.detail.score_range[0].toFixed(2);$('heatmapMax').textContent=S.detail.score_range[1].toFixed(2)+' · decision score';
  $('crfControls').hidden=options.method!=='crf';
  $('thresholdStatus').textContent='Preview only · ≥ threshold: medulla'+(options.method==='crf'&&S.thresholdPreview?' · refinement pending':'');
  $('fieldStatus').textContent=f?f.width+' × '+f.height+' · '+f.step+' px / cell · transparent: uncovered'+(!S.tissueRevision?' · tissue unavailable':'')+(options.method==='crf'?(f.guide?' · image-guided':' · spatial only; image pending'):''):'Tile counts · overlaps inspectable';
  if(f){$('cortexCount').textContent='≈ '+(f.area[0]/1e6).toFixed(2)+' M px²';$('medullaCount').textContent='≈ '+(f.area[1]/1e6).toFixed(2)+' M px²';}
  else {for(const id of ['cortexCount','medullaCount'])$(id).textContent+=' tiles';}
}
async function predictionTissue(signal){const slide=S.detail?.id,sequence=S.loadSequence;if(!slide)return;
  try{const result=await api('/api/slides/'+slide+'/prediction-tissue',{signal});if(sequence!==S.loadSequence||slide!==S.detail?.id)return;S.tissueGeometry=result.geometry||null;S.tissueRevision=result.revision||'';S.engine.tissue(S.tissueGeometry,S.tissueRevision);S.maskKey=null;await updateMasks();}
  catch(e){if(e.name!=='AbortError')toast('Tissue clipping: '+e.message,true);}
}
async function predictionGuide(signal){const slide=S.detail?.id,sequence=S.loadSequence;if(!slide)return;
  if(S.guideSlide===slide)return;
  try{const response=await api('/api/wsi/'+slide+'/thumbnail',{signal}),image=await createImageBitmap(await response.blob());if(sequence!==S.loadSequence||slide!==S.detail?.id){image.close();return;}S.guideSlide=slide;S.mainGuide?.close();S.mainGuide=image;if(S.engine.worker)S.engine.guide(await createImageBitmap(image));if($('aggregationMethod').value==='crf'){S.maskKey=null;await updateMasks();}}
  catch(e){if(e.name!=='AbortError')console.warn('CRF image guide unavailable',e.message);}
}
function mainTissueMask(options){if(!S.tissueGeometry)return null;const step=Math.max(options.resolution,Math.ceil(Math.max(S.detail.width,S.detail.height)/1536)),canvas=document.createElement('canvas');canvas.width=Math.ceil(S.detail.width/step);canvas.height=Math.ceil(S.detail.height/step);return PredictionField.tissueMask(S.tissueGeometry,canvas.width,canvas.height,step,canvas);}
function exportPredictionView(metadata=false){if(!S.raster||!S.fieldInfo){toast('Choose an aggregated view first.');return;}const f=S.fieldInfo,stamp=new Date().toISOString().replaceAll(':','-'),name='histoloop-preview-'+stamp;
  const download=(blob,extension)=>{const url=URL.createObjectURL(blob),link=node('a',{href:url,download:name+extension});link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  if(metadata)download(new Blob([JSON.stringify({version:1,created_at:new Date().toISOString(),slide_id:S.detail.id,prediction_id:S.detail.prediction_id,model_id:S.detail.model_id,fingerprint:S.detail.fingerprint,tissue_revision:S.tissueRevision||null,algorithm:'bounded-field-v1',...predictionOptions(),output:{width:f.width,height:f.height,step_level0:f.step,origin_level0:[0,0]},score_orientation:'score >= threshold means medulla',uncalibrated:true,uncovered:'transparent',refinement_pending:!!S.thresholdPreview},null,2)],{type:'application/json'}),'.json');
  else {const canvas=document.createElement('canvas');canvas.width=f.width;canvas.height=f.height;canvas.getContext('2d').drawImage(S.raster,0,0);canvas.toBlob(blob=>download(blob,'.png'));}
}
function wireEvents() {
  window.addEventListener("beforeunload",event=>{
    if(S.saving||S.pendingPolygon||S.draft.length||annotations.busy()){event.preventDefault();event.returnValue="";}
  });
  window.addEventListener("pagehide",event=>{if(!event.persisted){cancelSlideLoad();S.queueController?.abort();S.engine?.close();}});
  document.addEventListener("visibilitychange",()=>{
    if(document.hidden)navigationCache.cancel();else if(S.detail&&!S.loading)navigationCache.schedule(S.filtered,S.detail.id);
  });
  annotations.setup();setupPanels();
  $('liveThreshold').oninput=()=>liveThreshold(Number($('liveThreshold').value),true);
  $('liveThresholdNumber').oninput=()=>liveThreshold(Number($('liveThresholdNumber').value),true);
  $('liveThreshold').onchange=()=>liveThreshold(Number($('liveThreshold').value),false);
  $('thresholdReset').onclick=()=>liveThreshold(S.detail.prediction_threshold,false);
  for(const id of ['predictionView','aggregationMethod','gaussianSigma','rasterResolution','crfStrength','crfRadius','crfIterations'])$(id).onchange=()=>{S.thresholdPreview=false;updateMasks().catch(e=>toast(e.message,true));if($('aggregationMethod').value==='crf')predictionGuide(S.slideController?.signal);schedulePreferences();};
  $('exportMap').onclick=()=>exportPredictionView();$('exportMapMetadata').onclick=()=>exportPredictionView(true);
  $('tileBoundaries').onchange=()=>{requestRender();schedulePreferences();};
  for(const id of ["gridMode","sourceOverlap"])$(id).onchange=()=>{if(S.saving||S.pendingPolygon||S.draft.length){$("gridMode").value=S.activeGrid.grid_mode;$("sourceOverlap").value=S.activeGrid.source_overlap;toast("Save or cancel the region before changing grids.");return;}configureGrid();updateMasks().catch(error=>toast(error.message,true));schedulePreferences();};
  $("predictionScope").onchange=()=>{updatePredictionPlan();updateButtons();};
  $("datasetSelect").onchange=()=>switchDataset().catch(error=>toast(error.message,true));
  $("addDatasetButton").onclick=()=>{if(canManageDatasets()){$("datasetFormStatus").textContent="";$("datasetDialog").showModal();}};
  $("datasetForm").onsubmit=submitDataset;$("refreshDatasetsButton").onclick=refreshDatasets;
  let queueFrame=false;$("slideList").addEventListener("scroll",()=>{if(!queueFrame){queueFrame=true;requestAnimationFrame(()=>{queueFrame=false;renderQueue();});}},{passive:true});
  for(const id of ["search","cohortSelect","statusSelect","splitSelect","tissueSelect","sortSelect"])$(id).addEventListener(id==="search"?"input":"change",()=>{filterQueue(true);schedulePreferences();});
  $("queueModelSelect").onchange=()=>{refreshQueue().then(schedulePreferences).catch(error=>toast(error.message,true));};
  $("selectFiltered").onclick=()=>{S.filtered.forEach(s=>S.selectedSlides.add(s.id));renderQueue();updateSelection();};
  $("clearSelected").onclick=()=>{S.selectedSlides.clear();renderQueue();updateSelection();};
  $("bulkButton").onclick=()=>openBulk(S.selectedSlides.size?"selected":"filtered");
  $("allCortex").onclick=()=>openBulk("current",0);$("allMedulla").onclick=()=>openBulk("current",1);
  $("bulkScope").onchange=updateBulkPlan;$("bulkLabel").onchange=updateBulkPlan;$("submitBulk").onclick=submitBulk;
  $("displayPredictionSelect").onchange=()=>{
    const choice=$("displayPredictionSelect").value;
    loadSlide(S.detail.id,true,choice).catch(error=>{fillDisplayedPredictions();toast(error.message,true);});
  };
  $("prevSlide").onclick=()=>navigate(-1);$("nextSlide").onclick=()=>navigate(1);
  $("passSelect").onchange=async()=>{if(S.saving||S.pendingPolygon||S.draft.length){$("passSelect").value=S.pass;toast("Save or cancel the current region first.");return;}S.pass=$("passSelect").value;await refreshQueue();fillPasses();updateButtons();schedulePreferences();};
  $("newPass").onclick=()=>{$("passName").value="Pass "+(S.passes.length+1);$("passDialog").showModal();$("passName").focus();};
  $("passForm").onsubmit=async event=>{event.preventDefault();try{const pass=await api("/api/passes",{method:"POST",body:{name:$("passName").value}});S.passes.push(pass);S.pass=pass.id;fillPasses();$("passDialog").close();await refreshQueue();schedulePreferences();}catch(error){toast(error.message,true);}};
  document.querySelectorAll("[data-close]").forEach(button=>button.onclick=()=>$(button.dataset.close).close());
  document.querySelectorAll("[data-tool]").forEach(button=>button.onclick=()=>setMode(button.dataset.tool));
  document.querySelectorAll("[data-label]").forEach(button=>button.onclick=()=>setLabel(Number(button.dataset.label)));
  document.querySelectorAll("[data-tab]").forEach(button=>button.onclick=()=>selectTab(button.dataset.tab));
  $("fitButton").onclick=()=>S.viewer.viewport.goHome();$("undoButton").onclick=undoLatest;$("markReviewed").onclick=()=>reviewedNext(true);
  $("doneReview").onclick=()=>reviewedNext(false);
  $("historyPass").onchange=renderHistory;
  $("opacity").oninput=()=>{$("opacityValue").textContent=$("opacity").value+"%";$("overlayCanvas").style.opacity=Number($("opacity").value)/100;schedulePreferences();};
  $("overlayEnabled").onchange=()=>{requestRender();schedulePreferences();};
  $("overlaySource").onchange=()=>{updateMasks();schedulePreferences();};
  for(const id of ["cortexEnabled","medullaEnabled","cortexComparison","medullaComparison","cortexThreshold","medullaThreshold","keepManual","excludedEnabled"])$(id).addEventListener("input",()=>{clearTimeout(filterTimer);filterTimer=setTimeout(()=>{updateMasks();schedulePreferences();},100);});
  $("showAll").onclick=()=>{for(const key of ["cortex","medulla"]){$(key+"Comparison").value="all";$(key+"Enabled").checked=true;}updateMasks();schedulePreferences();};
  $("visibleOnly").onchange=schedulePreferences;$("reviewer").onchange=schedulePreferences;
  $("modelSelect").onchange=schedulePreferences;$("predictionThreshold").onchange=schedulePreferences;
  $("executorSelect").onchange=()=>setExecutor($("executorSelect").value);
  $("trainExecutorSelect").onchange=()=>setExecutor($("trainExecutorSelect").value);
  $("trainButton").onclick=()=>openTraining().catch(error=>toast(error.message,true));$("trainFromTab").onclick=$("trainButton").onclick;
  $("tileBudget").oninput=()=>{clearTimeout(planTimer);planTimer=setTimeout(updateTrainPlan,350);};
  $("submitTraining").onclick=()=>submitJob("train");$("predictButton").onclick=()=>submitJob("predict");
  $("jobsButton").onclick=()=>{$("jobsDialog").showModal();pollJobs();};$("jobStatusLink").onclick=$("jobsButton").onclick;
  $("helpButton").onclick=()=>$("helpDialog").showModal();$("storageButton").onclick=()=>toast("Correction history: "+S.stateDir);
  $("retrySave").onclick=()=>{if(S.pendingPolygon)saveRegion(S.pendingPolygon);};
  $("viewerWrap").addEventListener("pointermove",inspectHover);
  const surface=$("drawSurface");
  surface.addEventListener("pointerdown",event=>{
    if(S.saving||S.loading||S.datasetBusy||S.pendingPolygon||event.button!==0)return;
    const point=imagePoint(event);if(!point)return;S.highlighted=null;
    if(S.mode==="rectangle"){S.drag=point;S.draft=rectangle(point,point);surface.setPointerCapture(event.pointerId);}
    else if(S.mode==="polygon"){S.draft.push(point);S.cursor=point;}
    else if(S.mode==="freehand"){S.drag=point;S.draft=[point];S.freehandScreen=[event.clientX,event.clientY];S.freehandRawCount=1;surface.setPointerCapture(event.pointerId);}
    updateButtons();
    requestRender();
  });
  surface.addEventListener("pointermove",event=>{if(S.saving)return;const point=imagePoint(event);if(!point)return;S.cursor=point;if(S.drag){if(S.mode==="freehand")appendFreehand(event);else S.draft=rectangle(S.drag,point);}requestDraw();});
  surface.addEventListener("pointerup",event=>{if(!S.drag)return;const point=imagePoint(event),start=S.drag;S.drag=null;
    if(S.mode==="freehand"){appendFreehand(event,true);finishFreehand();updateButtons();return;}
    if(!point)return;const rect=rectangle(start,point);if(Math.abs(point[0]-start[0])*Math.abs(point[1]-start[1])<1){S.draft=[];updateButtons();requestRender();return;}saveRegion(rect);});
  surface.addEventListener("pointercancel",()=>{S.drag=null;S.draft=[];updateButtons();requestRender();});
  surface.addEventListener("dblclick",event=>{event.preventDefault();finishPolygon();});
  document.addEventListener("keydown",event=>{
    if(event.target.matches("input,select,textarea")||document.querySelector("dialog[open]"))return;
    if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"){event.preventDefault();undoLatest();return;}
    if(event.ctrlKey||event.metaKey||event.altKey)return;
    const key=event.key.toLowerCase();
    if(key==="h")setMode("pan");else if(key==="r")setMode("rectangle");else if(key==="p")setMode("polygon");else if(key==="b")setMode("freehand");
    else if(key==="c")setLabel(0);else if(key==="m")setLabel(1);else if(key==="x")setLabel(-2);else if(key==="f")S.viewer.viewport.goHome();
    else if(key==="n")navigate(1);else if(key==="v")reviewedNext();else if(key==="enter")finishPolygon();
    else if(key==='escape'&&annotations.editing){annotations.editing=null;setMode('pan');annotations.render();requestRender();}
    else if(key==="escape"&&!S.saving){S.pendingPolygon=null;S.draft=[];S.drag=null;S.highlighted=null;$("retrySave").hidden=true;updateButtons();requestRender();savedStatus("All completed corrections are saved");}
    else if(key==="/"){event.preventDefault();$("search").focus();}
  });
  window.addEventListener("beforeunload",event=>{if(S.saving||S.pendingPolygon){event.preventDefault();event.returnValue="";}});
}
async function boot() {
  initializeViewer();wireEvents();
  try {
    const data=await api("/api/bootstrap");S.passes=data.passes;S.models=data.models;S.stateDir=data.state_dir;
    S.datasets=data.datasets||[];fillDatasets(data.preferences.dataset_id);
    $("reviewer").value=data.reviewer;S.pass=S.passes.some(pass=>pass.id===data.preferences.pass_id)?data.preferences.pass_id:S.passes[0].id;
    fillPasses();fillModels(data.preferences.model_id);applyPreferences(data.preferences);
    setupExecutors(data);
    if(["active",...S.models.map(m=>m.id)].includes(data.preferences.queue_model))$("queueModelSelect").value=data.preferences.queue_model;
    if(S.dataset)await refreshQueue();fillCohorts(data.preferences.queue_filters?.cohortSelect);
    filterQueue();
    const restored=S.filtered.find(slide=>slide.id===data.preferences.slide_id&&slide.prediction_available);
    const first=restored||S.filtered.find(slide=>slide.prediction_available);
    if(first)await loadSlide(first.id,false,restored?data.preferences.view_prediction||"active":"active");
    else {clearSlide();savedStatus(S.dataset?"Dataset selected · predictions are needed before review":"No dataset workbooks are available");}
    await pollJobs();setInterval(pollJobs,5000);
  }catch(error){$("loading").hidden=true;savedStatus("Workspace unavailable",true);toast(error.message,true);}
}
boot();
