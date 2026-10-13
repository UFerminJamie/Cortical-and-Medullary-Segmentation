"use strict";
class TileEngine {
  constructor() {
    this.pending=new Map();this.sequence=0;this.generation=0;
    this.startWorker();
  }
  startWorker() {
    this.worker=typeof Worker!=="undefined"?new Worker("/static/tile-worker.js?v=8"):null;
    if(this.worker) {
      const worker=this.worker;
      this.worker.onmessage=event=>{
        const message=event.data;
        const key=message.type==="ready"?"init:"+message.generation:message.id;
        const pending=this.pending.get(key);if(!pending){message.raster?.close();return;}
        this.pending.delete(key);pending.resolve(message);
        if(message.type==='filter') {this.filterBusy=false;const next=this.nextFilter;this.nextFilter=null;if(next)this.sendFilter(next);}
      };
      this.worker.onerror=error=>{
        if(this.worker!==worker)return;
        console.warn("Overlay worker unavailable:",error.message);worker.terminate();this.worker=null;
        for(const [key,pending] of this.pending)pending.resolve(typeof key==="string"?null:{cancelled:true});
        this.pending.clear();
        this.filterBusy=false;this.nextFilter?.resolve({cancelled:true});this.nextFilter=null;
      };
    }
  }
  initialize(data,detail) {
    if(!this.worker)return Promise.resolve(null);
    this.nextFilter?.resolve({cancelled:true});this.nextFilter=null;this.filterBusy=false;
    if(this.pending.size) {
      // Superseded heavy computation cannot delay the newest slide.
      this.worker.terminate();
      for(const [key,pending] of this.pending)pending.resolve(typeof key==="string"?null:{cancelled:true});
      this.pending.clear();this.startWorker();
    }
    this.generation++;
    const generation=this.generation,copy=data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);
    return new Promise((resolve,reject)=>{
      this.pending.set("init:"+generation,{resolve,reject});
      this.worker.postMessage({type:"init",data:copy,detail,generation},[copy]);
    });
  }
  labels(ids,label){this.worker?.postMessage({type:"labels",ids,label});}
  tissue(geometry,revision){this.worker?.postMessage({type:'tissue',geometry,revision});}
  guide(image){this.worker?.postMessage({type:'guide',image},[image]);}
  filter(filters) {
    const id=++this.sequence,generation=this.generation;
    return new Promise((resolve,reject)=>{
      const request={id,generation,filters,resolve,reject};
      if(this.filterBusy){this.nextFilter?.resolve({cancelled:true});this.nextFilter=request;}else this.sendFilter(request);
    });
  }
  sendFilter(request){const {id,generation,filters,resolve,reject}=request;if(!this.worker){resolve({cancelled:true});return;}this.filterBusy=true;this.pending.set(id,{resolve,reject});this.worker.postMessage({type:'filter',id,generation,filters});}
  close() {
    this.worker?.terminate();this.worker=null;
    for(const [key,pending] of this.pending)pending.resolve(typeof key==="string"?null:{cancelled:true});
    this.pending.clear();
    this.filterBusy=false;this.nextFilter?.resolve({cancelled:true});this.nextFilter=null;
  }
}
