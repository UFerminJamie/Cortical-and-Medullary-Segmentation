"use strict";
class TileEngine {
  constructor() {
    this.pending=new Map();this.sequence=0;this.generation=0;
    this.startWorker();
  }
  startWorker() {
    this.worker=typeof Worker!=="undefined"?new Worker("/static/tile-worker.js?v=6"):null;
    if(this.worker) {
      const worker=this.worker;
      this.worker.onmessage=event=>{
        const message=event.data;
        const key=message.type==="ready"?"init:"+message.generation:message.id;
        const pending=this.pending.get(key);if(!pending){message.raster?.close();return;}
        this.pending.delete(key);pending.resolve(message);
      };
      this.worker.onerror=error=>{
        if(this.worker!==worker)return;
        console.warn("Overlay worker unavailable:",error.message);worker.terminate();this.worker=null;
        for(const [key,pending] of this.pending)pending.resolve(typeof key==="string"?null:{cancelled:true});
        this.pending.clear();
      };
    }
  }
  initialize(data,detail) {
    if(!this.worker)return Promise.resolve(null);
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
  filter(filters) {
    const id=++this.sequence,generation=this.generation;
    return new Promise((resolve,reject)=>{
      this.pending.set(id,{resolve,reject});this.worker.postMessage({type:"filter",id,generation,filters});
    });
  }
  close() {
    this.worker?.terminate();this.worker=null;
    for(const [key,pending] of this.pending)pending.resolve(typeof key==="string"?null:{cancelled:true});
    this.pending.clear();
  }
}
