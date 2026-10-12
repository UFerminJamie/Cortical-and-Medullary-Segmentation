"use strict";
function setupPanels() {
  const state={queue:innerWidth>1100,inspector:innerWidth>850};
  function apply() {for(const key of ["queue","inspector"]){document.body.dataset[key+"Open"]=state[key];$(key+"Toggle").setAttribute("aria-expanded",String(state[key]));$(key+"Panel").inert=!state[key];}S.viewer?.forceResize();requestRender();renderQueue();}
  for(const key of ["queue","inspector"])$(key+"Toggle").onclick=()=>{state[key]=!state[key];if(innerWidth<=850&&state[key])state[key==="queue"?"inspector":"queue"]=false;apply();};
  $("panelBackdrop").onclick=()=>{if(innerWidth<=1100)state.queue=false;if(innerWidth<=850)state.inspector=false;apply();};
  document.addEventListener("keydown",e=>{if(e.key==="Escape"&&!document.querySelector("dialog[open]"))$("panelBackdrop").click();});
  for(const key of ["queue","inspector"]) {
    const handle=document.querySelector('[data-resize="'+key+'"]');let start,width;
    const resize=value=>document.documentElement.style.setProperty("--"+key+"-width",Math.max(key==="queue"?220:260,Math.min(key==="queue"?400:440,value))+"px");
    handle.onpointerdown=e=>{start=e.clientX;width=$(key+"Panel").getBoundingClientRect().width;handle.setPointerCapture(e.pointerId);};
    handle.onpointermove=e=>{if(handle.hasPointerCapture(e.pointerId))resize(width+(e.clientX-start)*(key==="queue"?1:-1));};
    handle.onpointerup=e=>handle.releasePointerCapture(e.pointerId);
    handle.onkeydown=e=>{if(["ArrowLeft","ArrowRight"].includes(e.key)){e.preventDefault();resize($(key+"Panel").getBoundingClientRect().width+(e.key==="ArrowRight"?10:-10)*(key==="queue"?1:-1));}};
  }
  for(const [key,limit] of [["queue",1100],["inspector",850]])matchMedia('(min-width: '+(limit+1)+'px)').addEventListener("change",e=>{state[key]=e.matches;apply();});
  new ResizeObserver(()=>{S.viewer?.forceResize();requestRender();renderQueue();}).observe($("viewerWrap"));
  new ResizeObserver(entries=>document.documentElement.style.setProperty("--header-height",entries[0].target.getBoundingClientRect().height+"px")).observe(document.querySelector(".topbar"));apply();
}
