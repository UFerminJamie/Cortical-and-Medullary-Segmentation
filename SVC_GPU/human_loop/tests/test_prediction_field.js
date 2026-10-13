const assert=require('node:assert/strict'),Field=require('../static/prediction-field.js');
const detail={tiles:3,width:110,height:40,tile_size_level0:40,score_range:[-2,2]};
const data=new Float32Array([0,0,-2,0,-1,20,0,2,1,-1,70,0,1,1,-1]);
const options={resolution:10,method:'mean',source:'prediction',cortex:{enabled:true,comparison:'all'},medulla:{enabled:true,comparison:'all'}};
for(const method of ['mean','majority','gaussian','crf']){
  const f=Field.create(data,detail,{...options,method,sigma:.25}),labels=Field.classify(f,0,method,{strength:1,radius:1,iterations:3});
  const shuffled=new Float32Array([...data.slice(10),...data.slice(0,5),...data.slice(5,10)]),g=Field.create(shuffled,detail,{...options,method,sigma:.25});
  assert.deepEqual(Field.classify(g,0,method,{strength:1,radius:1,iterations:3}),labels);
  assert.deepEqual(g.score,f.score);
  assert.equal(f.count[6],0);assert.equal(labels[6],-1);
  const high=Field.classify(f,10,method,{strength:1,radius:1,iterations:3});assert(high.every(x=>x<1));
}
const f=Field.create(data,detail,options);assert.equal(Field.classify(f,0,'majority')[2],1); // tie -> mean >= threshold
assert.equal(Field.classify(f,.01,'majority')[2],0);
const tissue=new Uint8Array(44);tissue.fill(1);tissue[2]=0;
const masked=Field.create(data,detail,options,tissue);assert.equal(masked.count[2],0);assert.equal(Field.classify(masked,0,'crf',{strength:5,radius:3})[2],-1);
const components=new Uint32Array([1,2]),isolated=Field.create(new Float32Array([0,0,-.01,0,-1,10,0,2,1,-1]),
  {tiles:2,width:20,height:10,tile_size_level0:10,score_range:[-2,2]},options,components);
assert.deepEqual(Field.classify(isolated,0,'crf',{strength:5,radius:3,iterations:12}),new Int8Array([0,1]));
const manual=data.slice();manual[4]=0;const human=Field.create(manual,detail,options);
assert.equal(Field.pixels(human,Field.classify(human,-10,'mean'),{...options,source:'combined',keep_manual:true}).rgba[0],82);
assert.equal(Field.pixels(human,Field.classify(human,-10,'mean'),{...options,source:'combined',keep_manual:true}).displayed[0],0);
assert.deepEqual(data,new Float32Array([0,0,-2,0,-1,20,0,2,1,-1,70,0,1,1,-1]));
assert.deepEqual(Field.color(-2,[-2,2],{heatmap_palette:'custom',heatmap_low:'#010203',heatmap_mid:'#040506',heatmap_high:'#070809'}),[1,2,3]);
assert.deepEqual(Field.color(2,[-2,2],{heatmap_palette:'custom',heatmap_low:'#010203',heatmap_mid:'#040506',heatmap_high:'#070809'}),[7,8,9]);
const scoreLabels=Field.classify(f,0,'mean'),firstPixels=Field.pixels(f,scoreLabels,{...options,view:'aggregate-score'}).rgba;
const changedPixels=Field.pixels(f,scoreLabels,{...options,view:'aggregate-score',heatmap_palette:'blue-red'}).rgba;
assert.notDeepEqual(changedPixels,firstPixels);assert.deepEqual(Field.classify(f,0,'mean'),scoreLabels);
console.log('Prediction field: order independence, ties, thresholds, tissue gaps, overrides and palette cache invalidation passed');
