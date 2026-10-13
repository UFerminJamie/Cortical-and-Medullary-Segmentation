"use strict";
importScripts('/static/prediction-field.js?v=2');
let data, detail, grid;
let fieldCache=null,tissue=null,tissueRevision='',guide=null,lastSentField=null;
const color = label => label===-2?"#aab2b9":label===0?"#529cff":"#f18582";
function labelAt(i, source, threshold=detail.prediction_threshold) {
  const human=data[i*5+4];
  const prediction=Number(data[i*5+2]>=threshold);
  return source==="prediction"?prediction:source==="human"?human:human!==-1?human:prediction;
}
onmessage = event => {
  const message=event.data;
  if(message.type==="init") {
    data=new Float32Array(message.data);detail=message.detail;grid=new Map();
    fieldCache=null;tissue=null;guide?.close();guide=null;tissueRevision='';lastSentField=null;
    for(let i=0;i<detail.tiles;i++) {
      const key=Math.floor((data[i*5]+detail.tile_size_level0/2)/2048)+","+Math.floor((data[i*5+1]+detail.tile_size_level0/2)/2048);
      let bin=grid.get(key);if(!bin){bin=[];grid.set(key,bin);}bin.push(i);
    }
    postMessage({type:"ready",generation:message.generation,grid:[...grid.entries()]});
    return;
  }
  if(message.type==='tissue') {tissue=message.geometry;tissueRevision=message.revision||'';fieldCache=null;return;}
  if(message.type==='guide') {guide?.close();guide=message.image;return;}
  if(message.type==="labels") {for(const i of message.ids)data[i*5+4]=message.label;fieldCache=null;return;}
  if(message.type!=="filter"||!data)return;
  const start=performance.now(),f=message.filters,mask=new Uint8Array(detail.tiles),totals=[0,0],visible=[0,0];
  const bins=[new Uint32Array(60),new Uint32Array(60)],min=detail.score_range[0],span=Math.max(detail.score_range[1]-min,.01);
  let manual=0,excluded=0,excludedVisible=0;
  for(let i=0;i<detail.tiles;i++) {
    const off=i*5,human=data[off+4],label=labelAt(i,f.source,f.classification_threshold),score=data[off+2];
    if(human>=0)manual++;if(human===-2)excluded++;
    if(f.grid_mode&&f.grid_mode!=="native") {
      const step=detail.grid.profiles[f.source_overlap||"auto"].steps[f.grid_mode],origin=detail.grid.origin;
      if((data[off]-origin[0])%step||(data[off+1]-origin[1])%step)continue;
    }
    if(label>=0){totals[label]++;bins[label][Math.min(59,Math.max(0,Math.floor((score-min)/span*60)))]++;}
    let keep=false;
    if(label===-2)keep=f.excluded_enabled&&f.source!=="prediction";
    else if(label>=0) {
      const config=label===0?f.cortex:f.medulla;
      keep=config.enabled&&((f.source!=="prediction"&&human>=0&&f.keep_manual)||config.comparison==="all"||
        (config.comparison==="lte"&&score<=config.threshold)||(config.comparison==="gte"&&score>=config.threshold));
    }
    if(keep){mask[i]=1;if(label>=0)visible[label]++;else excludedVisible++;}
  }
  let raster=null,fieldInfo=null;
  if(typeof OffscreenCanvas!=="undefined"&&f.view?.startsWith('aggregate')) {
    const key=JSON.stringify([detail.fingerprint,detail.prediction_id,tissueRevision,f.method,f.sigma,f.resolution,f.grid_mode,f.source_overlap]);
    if(fieldCache?.key!==key) {
      let tissueMask=null;
      if(tissue) {
        const step=Math.max(f.resolution||32,Math.ceil(Math.max(detail.width,detail.height)/1536)),w=Math.ceil(detail.width/step),h=Math.ceil(detail.height/step);
        tissueMask=PredictionField.tissueMask(tissue,w,h,step,new OffscreenCanvas(w,h));
      }
      fieldCache={key,field:PredictionField.create(data,detail,f,tissueMask)};
    }
    const field=fieldCache.field;
    let image=null;
    if(f.method==='crf'&&guide){const c=new OffscreenCanvas(field.width,field.height),ctx=c.getContext('2d');ctx.drawImage(guide,0,0,field.width,field.height);image=ctx.getImageData(0,0,field.width,field.height).data;}
    const labels=PredictionField.classify(field,f.classification_threshold,f.preview&&f.method==='crf'?'mean':f.method,f,image);
    const {rgba,area,displayed}=PredictionField.pixels(field,labels,f),canvas=new OffscreenCanvas(field.width,field.height);
    canvas.getContext('2d').putImageData(new ImageData(rgba,field.width,field.height),0,0);raster=canvas.transferToImageBitmap();
    const changed=lastSentField!==key;
    fieldInfo={key,width:field.width,height:field.height,step:field.step,area,labels:displayed,guide:!!image};
    if(changed){fieldInfo.score=field.score.slice();fieldInfo.covered=field.count.slice();lastSentField=key;}
  }
  const result={type:"filter",id:message.id,generation:message.generation,mask,totals,visible,manual,excluded,excludedVisible,bins,raster,
    compute_ms:performance.now()-start,fieldInfo};
  postMessage(result,[mask.buffer,bins[0].buffer,bins[1].buffer,...(raster?[raster]:[]),...(fieldInfo?[fieldInfo.labels.buffer,...(fieldInfo.score?[fieldInfo.score.buffer,fieldInfo.covered.buffer]:[])]:[])]);
};
