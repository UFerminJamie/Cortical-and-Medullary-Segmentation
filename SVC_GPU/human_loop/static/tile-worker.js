"use strict";
let data, detail, grid;
const color = label => label===-2?"#aab2b9":label===0?"#529cff":"#f18582";
function labelAt(i, source) {
  const human=data[i*5+4];
  return source==="prediction"?data[i*5+3]:source==="human"?human:human!==-1?human:data[i*5+3];
}
onmessage = event => {
  const message=event.data;
  if(message.type==="init") {
    data=new Float32Array(message.data);detail=message.detail;grid=new Map();
    for(let i=0;i<detail.tiles;i++) {
      const key=Math.floor((data[i*5]+detail.tile_size_level0/2)/2048)+","+Math.floor((data[i*5+1]+detail.tile_size_level0/2)/2048);
      let bin=grid.get(key);if(!bin){bin=[];grid.set(key,bin);}bin.push(i);
    }
    postMessage({type:"ready",generation:message.generation,grid:[...grid.entries()]});
    return;
  }
  if(message.type==="labels") {for(const i of message.ids)data[i*5+4]=message.label;return;}
  if(message.type!=="filter"||!data)return;
  const start=performance.now(),f=message.filters,mask=new Uint8Array(detail.tiles),totals=[0,0],visible=[0,0];
  const bins=[new Uint32Array(60),new Uint32Array(60)],min=detail.score_range[0],span=Math.max(detail.score_range[1]-min,.01);
  let manual=0,excluded=0,excludedVisible=0;
  for(let i=0;i<detail.tiles;i++) {
    const off=i*5,human=data[off+4],label=labelAt(i,f.source),score=data[off+2];
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
  let raster=null;
  if(typeof OffscreenCanvas!=="undefined") {
    const divisor=Math.max(1,Math.ceil(Math.max(detail.width,detail.height)/1024));
    const width=Math.ceil(detail.width/divisor),height=Math.ceil(detail.height/divisor);
    const canvas=new OffscreenCanvas(width,height),ctx=canvas.getContext("2d"),pixels=new Uint32Array(width*height);
    const size=detail.tile_size_level0/divisor;
    for(let pass=0;pass<2;pass++)for(let i=0;i<detail.tiles;i++) {
      if(!mask[i]||(f.source!=="prediction"&&data[i*5+4]!==-1)!==Boolean(pass))continue;
      const x=Math.max(0,Math.floor(data[i*5]/divisor)),y=Math.max(0,Math.floor(data[i*5+1]/divisor));
      const right=Math.min(width,Math.ceil(data[i*5]/divisor+size)),bottom=Math.min(height,Math.ceil(data[i*5+1]/divisor+size));
      const label=labelAt(i,f.source),rgba=label===-2?0xffb9b2aa:label===0?0xffff9c52:0xff8285f1;
      for(let row=y;row<bottom;row++)pixels.fill(rgba,row*width+x,row*width+right);
    }
    ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer),width,height),0,0);
    raster=canvas.transferToImageBitmap();
  }
  const result={type:"filter",id:message.id,generation:message.generation,mask,totals,visible,manual,excluded,excludedVisible,bins,raster,
    compute_ms:performance.now()-start};
  postMessage(result,[mask.buffer,bins[0].buffer,bins[1].buffer,...(raster?[raster]:[])]);
};
