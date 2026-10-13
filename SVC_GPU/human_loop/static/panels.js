"use strict";
function setupPanels() {
  S.layout={queue:innerWidth>1100,inspector:innerWidth>850,widths:{queue:260,inspector:310},expanded:{},focus:false};
  const sections=new Map();
  function accordion(key,title,content,open=false){
    const shell=node('section',{class:'collapsible-panel'}),button=node('button',{class:'panel-header','aria-controls':'panel-'+key,text:title}),body=node('div',{id:'panel-'+key});
    content[0].before(shell);for(const element of content)body.append(element);shell.append(button,body);sections.set(key,{button,body,open});
    button.onclick=()=>{S.layout.expanded[key]=body.hidden;applySections();schedulePreferences();};
  }
  accordion('sample-filters','Sample filters',[document.querySelector('.queue-filters')]);
  let index=0;
  for(const section of [...document.querySelectorAll('#tab-annotate > .inspector-section')]){
    if(section.id)continue;
    accordion(index?'prediction-display':'tile-corrections',index?'Prediction display':'Tile-label corrections',[section]);index++;
  }
  const histogramShell=node('div');$('panel-prediction-display').parentNode.after(histogramShell);histogramShell.append($('histogram'),document.querySelector('.histogram-labels'));
  accordion('score-distribution','Score distribution',[histogramShell]);
  accordion('annotations','Annotations',[$('annotationPanel')],true);
  accordion('overlap','Prediction display',[$('overlapPanel')]);
  accordion('history','History',[$('historyList')]);
  const compute=[...document.querySelectorAll('#tab-model > .inspector-section')];if(compute.length)accordion('compute','Compute / jobs',compute);
  function applySections(){for(const [key,s]of sections){const expanded=S.layout.expanded[key]??s.open;s.body.hidden=!expanded;s.button.setAttribute('aria-expanded',String(expanded));s.button.dataset.expanded=expanded;}
    if(!sections.get('score-distribution').body.hidden)drawHistogram();if(!sections.get('annotations').body.hidden)annotations.render();if(!sections.get('history').body.hidden)renderHistory();}
  function widths(){
    const minViewer=460,other=S.layout.inspector&&!S.layout.focus?S.layout.widths.inspector:0;
    S.layout.widths.queue=Math.max(220,Math.min(380,innerWidth>1100?innerWidth-other-minViewer:380,S.layout.widths.queue));
    S.layout.widths.inspector=Math.max(260,Math.min(420,innerWidth>1100?innerWidth-(S.layout.queue&&!S.layout.focus?S.layout.widths.queue:0)-minViewer:420,S.layout.widths.inspector));
    for(const key of ['queue','inspector'])document.documentElement.style.setProperty('--'+key+'-width',S.layout.widths[key]+'px');
  }
  function apply(){widths();for(const key of ['queue','inspector']){const visible=S.layout[key]&&!S.layout.focus;document.body.dataset[key+'Open']=visible;$(key+'Toggle').setAttribute('aria-expanded',String(visible));$(key+'Panel').inert=!visible;}
    document.body.dataset.focus=S.layout.focus;$('focusViewer').textContent=S.layout.focus?'Exit focus':'Focus viewer';$('focusViewer').setAttribute('aria-pressed',String(S.layout.focus));
    applySections();requestAnimationFrame(()=>{S.viewer?.forceResize();requestRender();renderQueue();});}
  S.panels={restore(value){if(value)S.layout={...S.layout,...value,widths:{...S.layout.widths,...value.widths},expanded:{...value.expanded}};if(innerWidth<=850&&S.layout.queue&&S.layout.inspector)S.layout.inspector=false;apply();},apply};
  for(const key of ['queue','inspector'])$(key+'Toggle').onclick=()=>{S.layout.focus=false;S.layout[key]=!S.layout[key];if(innerWidth<=850&&S.layout[key])S.layout[key==='queue'?'inspector':'queue']=false;apply();schedulePreferences();};
  $('focusViewer').onclick=()=>{S.layout.focus=!S.layout.focus;apply();schedulePreferences();};
  $('panelBackdrop').onclick=()=>{if(innerWidth<=1100)S.layout.queue=false;if(innerWidth<=850)S.layout.inspector=false;apply();schedulePreferences();};
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!document.querySelector('dialog[open]')){if(S.layout.focus){S.layout.focus=false;apply();schedulePreferences();}else $('panelBackdrop').click();}});
  for(const key of ['queue','inspector']) {
    const handle=document.querySelector('[data-resize="'+key+'"]');let start,width;handle.tabIndex=0;
    const resize=value=>{S.layout.widths[key]=value;widths();requestRender();};
    handle.onpointerdown=e=>{start=e.clientX;width=$(key+'Panel').getBoundingClientRect().width;handle.setPointerCapture(e.pointerId);};
    handle.onpointermove=e=>{if(handle.hasPointerCapture(e.pointerId))resize(width+(e.clientX-start)*(key==='queue'?1:-1));};
    handle.onpointerup=e=>{handle.releasePointerCapture(e.pointerId);apply();schedulePreferences();};
    handle.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();resize($(key+'Panel').getBoundingClientRect().width+(e.key==='ArrowRight'?10:-10)*(key==='queue'?1:-1));apply();schedulePreferences();}};
  }
  window.addEventListener('resize',()=>{widths();if(innerWidth<=850){S.layout.queue=false;S.layout.inspector=false;}apply();});
  new ResizeObserver(()=>{S.viewer?.forceResize();requestRender();}).observe($('viewerWrap'));
  new ResizeObserver(entries=>document.documentElement.style.setProperty('--header-height',entries[0].target.getBoundingClientRect().height+'px')).observe(document.querySelector('.topbar'));apply();
}
