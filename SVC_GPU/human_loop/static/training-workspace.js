"use strict";
const trainingWorkspace={samples:[],train:new Map(),validation:new Map(),filtered:[],loading:false,submitting:false,
  async open(){
    $('trainDialog').showModal();
    this.results();
    if(!this.samples.length)await this.refresh();else this.filter();
    this.mode();
  },
  async refresh(){
    if(this.loading)return;this.loading=true;$('trainingPickerStatus').textContent='Loading samples and correction summaries…';
    try{this.samples=await api('/api/training/samples');
      for(const group of [this.train,this.validation])for(const [id] of group){const sample=this.samples.find(s=>s.id===id);if(sample?.prediction_available)group.set(id,sample);else group.delete(id);}
      const choices=new Map(this.samples.map(s=>[s.dataset_id+':'+s.cohort,(s.dataset_name||s.dataset_id)+' / '+s.cohort]));
      $('trainingCohort').replaceChildren(node('option',{value:'',text:'All cohorts'}),...Array.from(choices).sort((a,b)=>a[1].localeCompare(b[1])).map(([value,text])=>node('option',{value,text})));
      this.filter();
    }catch(e){toast(e.message,true);$('trainingPickerStatus').textContent=e.message;}
    finally{this.loading=false;this.counts();}
  },
  mode(){const legacy=$('trainingMode').value==='legacy';$('selectedTrainingPanel').hidden=legacy;$('legacyTrainingPanel').hidden=!legacy;
    $('submitTraining').hidden=!legacy;$('submitSelectedTraining').hidden=legacy;
    if(legacy)updateTrainPlan();else this.counts();
  },
  filter(){const search=$('trainingSearch').value.toLowerCase(),cohort=$('trainingCohort').value,split=$('trainingSplit').value;
    this.filtered=this.samples.filter(s=>(!search||s.name.toLowerCase().includes(search))&&(!cohort||s.dataset_id+':'+s.cohort===cohort)&&(!split||s.original_split===split));
    const key=$('trainingSort').value,direction=$('trainingSortDirection').value==='desc'?-1:1;
    this.filtered.sort((a,b)=>{let delta;
      if(key==='medulla_fraction'){if(a[key]==null||b[key]==null)return a[key]==null?(b[key]==null?a.name.localeCompare(b.name):1):-1;delta=a[key]-b[key];}
      else if(key==='cohort')delta=((a.dataset_name||a.dataset_id)+' '+a.cohort).localeCompare((b.dataset_name||b.dataset_id)+' '+b.cohort);
      else delta=String(a[key]).localeCompare(String(b[key]));
      return direction*delta||a.name.localeCompare(b.name);});
    $('trainingSampleViewport').scrollTop=0;this.render();this.counts();
  },
  select(sample,group,checked){const selected=group==='training'?this.train:this.validation,other=group==='training'?this.validation:this.train;
    if(checked){if(other.has(sample.id))return;selected.set(sample.id,sample);}else selected.delete(sample.id);
    this.render();this.counts();
  },
  counts(){
    $('trainingSelectionCounts').textContent=this.train.size+' training samples · '+this.validation.size+' validation samples';
    if(!this.loading)$('trainingPickerStatus').textContent=this.filtered.length+' samples shown · Medullary % uses predictions with human corrections; excluded tiles omitted.';
    const active=S.jobs.some(j=>j.kind==='train'&&['preparing','submitting','pending','running','submitted','unknown'].includes(j.status));
    $('submitSelectedTraining').disabled=this.loading||this.submitting||S.computeEnabled===false||!this.train.size||!this.validation.size||active;
  },
  render(){
    const viewport=$('trainingSampleViewport'),columns=Number($('trainingColumns').value),height=286,
      count=Math.ceil(this.filtered.length/columns),first=Math.max(0,Math.floor(viewport.scrollTop/height)-2),last=Math.min(count,Math.ceil((viewport.scrollTop+(viewport.clientHeight||500))/height)+2);
    const content=$('trainingSampleGrid');content.style.gridTemplateColumns='repeat('+columns+',minmax(0,1fr))';
    content.style.paddingTop=first*height+'px';content.style.paddingBottom=Math.max(0,count-last)*height+'px';
    const cards=[];
    for(const sample of this.filtered.slice(first*columns,last*columns)){
      const card=node('article',{class:'training-sample-card'+(!sample.prediction_available?' unavailable':'')});
      const image=node('img',{alt:'WSI, predictions, and human corrections for '+sample.name,loading:'lazy',width:480,height:300});
      if(sample.prediction_available)image.src='/api/training/samples/'+sample.id+'/thumbnail?'+new URLSearchParams({prediction_id:sample.prediction_id,revision:sample.revision});
      image.onerror=()=>{image.alt='Thumbnail unavailable; refresh samples if corrections changed.';};
      const checks=node('div',{class:'training-sample-checks'});
      for(const [role,group,other] of [['training',this.train,this.validation],['validation',this.validation,this.train]]){
        const input=node('input',{type:'checkbox','aria-label':sample.name+' '+role,onchange:e=>this.select(sample,role,e.target.checked)});
        input.checked=group.has(sample.id);input.disabled=!sample.prediction_available||other.has(sample.id)||this.submitting;
        checks.append(node('label',{},[input,role==='training'?'Training':'Validation']));
      }
      card.append(image,node('strong',{text:sample.name,title:sample.name}),node('small',{text:(sample.dataset_name||sample.dataset_id)+' / '+sample.cohort+' · '+sample.original_split}),
        node('small',{text:'Medullary '+(sample.medulla_fraction==null?'—':(100*sample.medulla_fraction).toFixed(1)+'%')+' · '+fmt(sample.manual_tiles)+' corrected tiles'}),checks);
      if(!sample.prediction_available)card.append(node('small',{text:sample.selection_error||'Run a prediction before selecting this sample.'}));
      cards.push(card);
    }
    content.replaceChildren(...cards);
  },
  async submit(){
    if(this.submitting)return;
    if(annotations.busy()||S.saving||S.pendingPolygon||S.draft.length){toast('Save or cancel the current draft before submitting training.',true);return;}
    const model_name=$('trainingModelName').value.trim(),reviewer=$('reviewer').value.trim();
    const text=$('trainingCValues').value.trim(),tokens=text.split(/[\s,]+/),c_values=tokens.map(Number),threshold=Number($('trainingThreshold').value);
    if(!reviewer||!model_name||model_name.length>120){toast('Enter a reviewer and model name (up to 120 characters).',true);return;}
    if(!text||tokens.length>20||c_values.some(c=>!Number.isFinite(c)||c<=0)||!Number.isFinite(threshold)){toast('Enter 1–20 positive C values and a finite threshold.',true);return;}
    if(!this.train.size||!this.validation.size){toast('Select training and validation samples.',true);return;}
    this.submitting=true;this.counts();$('submitSelectedTraining').textContent='Submitting training…';
    const entries=group=>Array.from(group.values(),s=>({id:s.id,prediction_id:s.prediction_id,revision:s.revision}));
    try{const job=await api('/api/jobs',{method:'POST',body:{...actorPayload(),kind:'train',model_name,c_values,threshold,executor:'slurm',
        model_id:$('modelSelect').value,training_samples:entries(this.train),validation_samples:entries(this.validation)}});
      S.jobs.unshift(job);S.knownJobs.set(job.id,job.status);renderJobs();updateButtons();
      toast('Training submitted. Labels are being frozen before the Slurm GPU job queues.');
      $('trainingSubmissionStatus').textContent=model_name+' · '+job.id+' · '+job.status+' — check GPU Jobs for progress and errors.';
    }catch(e){toast(e.message,true);$('trainingSubmissionStatus').textContent=e.message;}
    finally{this.submitting=false;$('submitSelectedTraining').textContent='Submit GPU training';this.counts();}
  },
  results(){
    const models=S.models.filter(m=>m.validation_metrics),container=$('trainingResults');
    const metric=value=>value==null?'—':(100*value).toFixed(2)+'%';
    const table=node('table',{class:'training-metrics'}),head=node('tr');
    for(const label of ['Model / C','Accuracy','Balanced accuracy','AUROC','Class','Class accuracy','Class balanced accuracy','Class AUROC','Sensitivity','Specificity','Use model'])head.append(node('th',{text:label}));
    table.append(node('thead',{},[head]));const body=node('tbody');
    for(const model of models){const m=model.validation_metrics;
      for(const [index,key] of ['cortical','medullary'].entries()){
        const per=m.per_class[key],row=node('tr');
        if(index===0)for(const value of [model.name,metric(m.accuracy),metric(m.balanced_accuracy),metric(m.auroc??m.per_class.medullary.auc)])row.append(node('td',{rowspan:2,text:value}));
        row.append(...[key,metric(per.accuracy),metric(per.sensitivity==null||per.specificity==null?null:(per.sensitivity+per.specificity)/2),metric(per.auc),metric(per.sensitivity),metric(per.specificity)].map(text=>node('td',{text})));
        if(index===0)row.append(node('td',{rowspan:2},[node('button',{class:'button secondary',text:'Use model',onclick:()=>{
          $('modelSelect').value=model.id;schedulePreferences();toast(model.name+' selected for prediction.');this.results();}})]));
        body.append(row);
      }
    }
    table.append(body);container.replaceChildren();
    container.append(node('p',{class:'muted',text:'Validation reference: saved predictions + human corrections. AUROC is unavailable when the validation set contains only one class. Selecting a model makes it available for the next prediction run.'}));
    if(!models.length){container.append(node('p',{text:'Completed candidates and validation results will appear here.'}));return;}
    container.append(node('div',{class:'training-table-scroll'},[table]));
    for(const model of models){const details=node('details',{class:'training-model-record'}),m=model.validation_metrics,md=model.metadata;
      details.append(node('summary',{text:model.name+' · Confusion matrix and sample record'}));
      details.append(node('p',{text:'Model '+model.id+' · threshold '+md.decision_threshold+' · '+fmt(md.training_tiles)+' training tiles · '+fmt(md.validation_tiles)+' validation tiles'}));
      const matrix=node('table',{class:'training-metrics confusion-matrix'}),top=node('tr');
      for(const text of ['Reference ↓ / prediction →','Cortex','Medulla'])top.append(node('th',{text}));matrix.append(top);
      for(const [i,label] of ['Cortex','Medulla'].entries())matrix.append(node('tr',{},[node('th',{text:label}),...m.confusion_matrix[i].map(v=>node('td',{text:fmt(v)}))]));
      details.append(matrix);
      for(const [title,rows] of [['Training',md.training_samples],['Validation',md.validation_samples]])details.append(node('p',{text:title+' samples: '+rows.map(r=>r.name+' ('+r.dataset_name+' / '+r.cohort+'; '+fmt(r.tiles)+' tiles; prediction '+r.prediction_id+'; correction revision '+r.revision+')').join(', ')}));
      container.append(details);
    }
  },
  setup(){
    $('trainingMode').onchange=()=>this.mode();
    for(const id of ['trainingCohort','trainingSplit','trainingSort','trainingSortDirection','trainingColumns'])$(id).onchange=()=>this.filter();
    $('trainingSearch').oninput=()=>this.filter();
    $('refreshTrainingSamples').onclick=()=>this.refresh();
    $('clearTrainingSamples').onclick=()=>{this.train.clear();this.validation.clear();this.render();this.counts();};
    $('trainingSampleViewport').onscroll=()=>{if(!this.frame){this.frame=true;requestAnimationFrame(()=>{this.frame=false;this.render();});}};
    $('submitSelectedTraining').onclick=()=>this.submit();
    $('refreshTrainingResults').onclick=async()=>{try{const boot=await api('/api/bootstrap');S.models=boot.models;fillModels();this.results();}catch(e){toast(e.message,true);}};
    $('trainingResultsButton').onclick=()=>{$('trainingResultsPanel').hidden=false;this.results();$('trainingResultsPanel').scrollIntoView({block:'start',behavior:'smooth'});};
  }
};
