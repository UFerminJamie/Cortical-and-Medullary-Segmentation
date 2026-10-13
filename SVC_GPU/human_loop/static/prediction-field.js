/* Shared worker/fallback math. Bounded level-zero sampling; never a full WSI raster. */
"use strict";
const PredictionField = (() => {
  const sigmoid=x=>1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));
  function create(data, detail, options, tissue=null) {
    const step=Math.max(Number(options.resolution)||32,Math.ceil(Math.max(detail.width,detail.height)/1536));
    const width=Math.ceil(detail.width/step),height=Math.ceil(detail.height/step),length=width*height;
    const sum=new Float64Array(length),weight=new Float64Array(length),count=new Uint32Array(length),
      human0=new Uint16Array(length),human1=new Uint16Array(length),excluded=new Uint16Array(length);
    const ids=Array.from({length:detail.tiles},(_,i)=>i).sort((a,b)=>data[a*5]-data[b*5]||data[a*5+1]-data[b*5+1]||data[a*5+2]-data[b*5+2]);
    const gaussian=options.method==='gaussian',sigma=Math.max(.05,Math.min(.75,options.sigma||.25))*detail.tile_size_level0;
    const f={width,height,step,count,human0,human1,excluded,ids,data,detail,options,tissue};
    for(const i of ids)visit(f,i,(p,x,y)=>{
      const o=i*5,s=detail.tile_size_level0,
        w=gaussian?Math.exp(-((x-data[o]-s/2)**2+(y-data[o+1]-s/2)**2)/(2*sigma*sigma)):1;
      sum[p]+=w*data[o+2];weight[p]+=w;count[p]++;
      if(data[o+4]===0)human0[p]++;else if(data[o+4]===1)human1[p]++;else if(data[o+4]===-2)excluded[p]++;
    });
    const score=new Float32Array(length);score.fill(NaN);
    const active=[];
    for(let p=0;p<length;p++)if(weight[p]){score[p]=sum[p]/weight[p];active.push(p);}
    f.active=Uint32Array.from(active);f.score=score;return f;
  }
  function visit(f,i,callback) {
    const o=i*5,{data,detail,options,width,height,step}=f,size=detail.tile_size_level0;
    if(options.grid_mode&&options.grid_mode!=='native') {
      const spacing=detail.grid.profiles[options.source_overlap||'auto'].steps[options.grid_mode],origin=detail.grid.origin;
      if((data[o]-origin[0])%spacing||(data[o+1]-origin[1])%spacing)return;
    }
    const left=Math.max(0,Math.ceil(data[o]/step-.5)),right=Math.min(width,Math.ceil((data[o]+size)/step-.5)),
      top=Math.max(0,Math.ceil(data[o+1]/step-.5)),bottom=Math.min(height,Math.ceil((data[o+1]+size)/step-.5));
    for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){
      const p=y*width+x;if(f.tissue&&!f.tissue[p])continue;callback(p,(x+.5)*step,(y+.5)*step);
    }
  }
  function classify(f,threshold,method,params={},guide=null) {
    const labels=new Int8Array(f.count.length);labels.fill(-1);
    let votes=null;
    if(method==='majority') {
      votes=new Uint32Array(labels.length);
      for(const i of f.ids)if(f.data[i*5+2]>=threshold)visit(f,i,p=>votes[p]++);
    }
    for(const p of f.active)labels[p]=votes?
      (votes[p]*2===f.count[p]?Number(f.score[p]>=threshold):Number(votes[p]*2>f.count[p])):Number(f.score[p]>=threshold);
    if(method==='crf')refine(f,threshold,labels,params,guide);
    return labels;
  }
  function refine(f,threshold,labels,params,guide) {
    const strength=Math.max(0,Math.min(5,Number(params.strength)||0)),radius=Math.max(1,Math.min(3,Number(params.radius)||1)),
      iterations=Math.max(1,Math.min(12,Number(params.iterations)||5)),scale=Math.max(.1,(f.detail.score_range[1]-f.detail.score_range[0])/6);
    let q=new Float32Array(labels.length),next=new Float32Array(labels.length);
    for(const p of f.active)q[p]=sigmoid((f.score[p]-threshold)/scale);
    const directions=[[1,0],[-1,0],[0,1],[0,-1]];
    for(let iter=0;iter<iterations;iter++) {
      for(const p of f.active) {
        const x=p%f.width,y=Math.floor(p/f.width);let sum=0,weights=0;
        for(const [dx,dy] of directions)for(let k=1;k<=radius;k++) {
          const xx=x+dx*k,yy=y+dy*k;if(xx<0||xx>=f.width||yy<0||yy>=f.height)break;
          const n=yy*f.width+xx;if(!f.count[n])break; // No edges across uncovered/tissue gaps.
          if(f.tissue&&f.tissue[p]!==f.tissue[n])break; // Separate vector tissue components stay separate even at coarse resolutions.
          let w=Math.exp(-k*k/(2*radius*radius));
          if(guide){let color=0;for(let c=0;c<3;c++)color+=(guide[p*4+c]-guide[n*4+c])**2;w*=Math.exp(-color/(2*35*35));}
          sum+=w*(2*q[n]-1);weights+=w;
        }
        next[p]=sigmoid((f.score[p]-threshold)/scale+strength*(weights?sum/weights:0));
      }
      [q,next]=[next,q];
    }
    for(const p of f.active)labels[p]=Number(q[p]>=.5);
  }
  // Palette changes affect display only, never scores or classification.
  function palette(options={}) {
    const presets={cividis:[[0,32,77],[65,82,109],[128,124,116],[190,174,108],[255,233,69]],
      'blue-red':[[30,80,210],[245,245,245],[210,35,45]],viridis:[[68,1,84],[33,145,140],[253,231,37]]};
    if(options.heatmap_palette==='custom')return [options.heatmap_low,options.heatmap_mid,options.heatmap_high].map((hex,i)=>{
      const value=/^#[0-9a-f]{6}$/i.test(hex||'')?hex:['#2455d6','#f5f5f5','#d2232d'][i];
      return [1,3,5].map(start=>parseInt(value.slice(start,start+2),16));});
    return presets[options.heatmap_palette]||presets.cividis;
  }
  function color(score,range,options={}) {
    const t=Math.max(0,Math.min(1,(score-range[0])/Math.max(.001,range[1]-range[0]))),
      stops=palette(options),u=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(u));
    return stops[i].map((v,c)=>Math.round(v+(stops[i+1][c]-v)*(u-i)));
  }
  function pixels(f,labels,options) {
    const rgba=new Uint8ClampedArray(labels.length*4),area=[0,0],range=f.detail.score_range,displayed=new Int8Array(labels.length);displayed.fill(-1);
    const paletteKey=JSON.stringify(palette(options));
    if(f.paletteKey!==paletteKey){f.paletteKey=paletteKey;f.palette=new Uint8Array(1024*3);for(let i=0;i<1024;i++)f.palette.set(color(range[0]+i/1023*(range[1]-range[0]),range,options),i*3);}
    const span=Math.max(.001,range[1]-range[0]);
    for(const p of f.active) {
      let label=labels[p],manual=false;
      if(options.source!=='prediction') {
        if(f.excluded[p]){label=-2;manual=true;}
        else if(f.human0[p]||f.human1[p]){label=f.human0[p]===f.human1[p]?labels[p]:Number(f.human1[p]>f.human0[p]);manual=true;}
      }
      if(options.source==='human'&&!manual)continue;
      displayed[p]=label;
      const config=label===0?options.cortex:options.medulla,score=f.score[p];
      if(label===-2){if(!options.excluded_enabled)continue;}
      else {
        area[label]+=f.step*f.step;
        if(!config.enabled)continue;
        if(!(manual&&options.keep_manual)&&config.comparison!=='all'&&!((config.comparison==='lte'&&score<=config.threshold)||(config.comparison==='gte'&&score>=config.threshold)))continue;
      }
      const o=p*4;
      if(options.view==='aggregate-score'&&!manual){const i=Math.max(0,Math.min(1023,Math.round((score-range[0])/span*1023)))*3;rgba[o]=f.palette[i];rgba[o+1]=f.palette[i+1];rgba[o+2]=f.palette[i+2];}
      else if(label===-2){rgba[o]=170;rgba[o+1]=178;rgba[o+2]=185;}
      else if(label===0){rgba[o]=82;rgba[o+1]=156;rgba[o+2]=255;}
      else {rgba[o]=241;rgba[o+1]=133;rgba[o+2]=130;}
      rgba[o+3]=255;
    }
    return {rgba,area,displayed};
  }
  function tissueMask(geometry,width,height,step,canvas) {
    const ctx=canvas.getContext('2d');ctx.scale(1/step,1/step);
    const polygons=geometry.type==='Polygon'?[geometry.coordinates]:geometry.type==='MultiPolygon'?geometry.coordinates:[];
    polygons.forEach((rings,i)=>{const id=i+1;ctx.fillStyle='rgb('+(id&255)+','+((id>>8)&255)+','+((id>>16)&255)+')';ctx.beginPath();for(const ring of rings){ring.forEach((p,j)=>j?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();}ctx.fill('evenodd');});
    const pixels=ctx.getImageData(0,0,width,height).data,mask=new Uint32Array(width*height);
    for(let p=0;p<mask.length;p++)if(pixels[p*4+3]>=128)mask[p]=pixels[p*4]+(pixels[p*4+1]<<8)+(pixels[p*4+2]<<16);
    return mask;
  }
  return {create,classify,color,palette,pixels,visit,tissueMask};
})();
if(typeof module!=='undefined')module.exports=PredictionField;
