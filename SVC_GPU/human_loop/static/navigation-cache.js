"use strict";
// Immutable response bytes. Always validate with the server before using a view.
class NavigationCache {
  constructor(maxBytes=48*1024*1024,capacity=8) {
    this.maxBytes=maxBytes;this.capacity=capacity;this.items=new Map();this.bytes=0;
    this.hits=0;this.misses=0;this.notModified=0;this.timer=null;this.background=null;this.generation=0;
  }
  async read(path,signal) {
    const old=this.items.get(path),headers=old?.etag?{"If-None-Match":old.etag}:{};
    const response=await fetch(path,{credentials:"same-origin",headers,signal,cache:"no-store"});
    if(response.status===304&&old) {
      this.hits++;this.notModified++;
      // The response may have been evicted by another request while validating.
      if(this.items.has(path)){this.items.delete(path);this.items.set(path,old);}
      return old.bytes.slice(0);
    }
    if(!response.ok) {
      const body=await response.json().catch(()=>({detail:response.statusText}));
      const error=new Error(body.detail||response.statusText);error.status=response.status;throw error;
    }
    this.misses++;
    const bytes=await response.arrayBuffer(),etag=response.headers.get("ETag");
    const existing=this.items.get(path);
    if(existing){this.items.delete(path);this.bytes-=existing.bytes.byteLength;}
    if(bytes.byteLength<=this.maxBytes) {
      this.items.set(path,{bytes,etag});this.bytes+=bytes.byteLength;
      while(this.items.size>this.capacity||this.bytes>this.maxBytes) {
        const key=this.items.keys().next().value;this.bytes-=this.items.get(key).bytes.byteLength;this.items.delete(key);
      }
    }
    return bytes.slice(0);
  }
  invalidateSlide(id) {
    for(const [key,value] of this.items)if(key.startsWith("/api/slides/"+id+"/")) {
      this.bytes-=value.bytes.byteLength;this.items.delete(key);
    }
  }
  cancel() {
    this.generation++;clearTimeout(this.timer);this.timer=null;
    this.background?.abort();this.background=null;
  }
  clear() {this.cancel();this.items.clear();this.bytes=0;}
  schedule(slides,currentId) {
    this.cancel();const generation=this.generation;
    const index=slides.findIndex(row=>row.id===currentId);
    if(index<0||navigator.connection?.saveData)return;
    const neighbors=[slides[index+1],slides[index-1]].filter(row=>row?.prediction_available);
    this.timer=setTimeout(async()=>{
      const controller=this.background=new AbortController();
      try {
        // One background request at a time; foreground requests cancel this lane.
        for(const slide of neighbors) {
          if(generation!==this.generation)return;
          const path="/api/slides/"+slide.id+"/view";
          if(!this.items.has(path)) {
            const bytes=await this.read(path,controller.signal);
            const length=new DataView(bytes).getUint32(0,true);
            const detail=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,length)));
            // One 256 px overview, never full-resolution WSI pixels.
            await fetch(`/api/wsi/${slide.id}_files/${Math.min(8,detail.max_level)}/0_0.jpg`,
              {credentials:"same-origin",signal:controller.signal}).then(r=>r.arrayBuffer());
          }
        }
      }catch(error){if(error.name!=="AbortError")console.debug("Adjacent slide prefetch:",error.message);}
      finally{if(this.background===controller)this.background=null;}
    },200);
  }
  metrics() {return {entries:this.items.size,bytes:this.bytes,maxBytes:this.maxBytes,hits:this.hits,
    misses:this.misses,notModified:this.notModified,background:Boolean(this.background)};}
}
const navigationCache=new NavigationCache();
