'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {routes,features,fit,apply,solve,equal}=require('./solver');
const [cmd,dir,output,model,cap='12']=process.argv.slice(2);
const load=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const save=(p,data)=>fs.writeFileSync(p,JSON.stringify(data)+'\n');
const files=fs.readdirSync(dir).filter(s=>s.endsWith('.json')).sort();
if(!files.length) throw Error('Empty split');
if(cmd==='train') {
  const policy=[];
  for(const file of files) {
    const task=load(path.join(dir,file)), rewards=[];
    routes.forEach((route,i)=>{
      const card=fit(route,task.train);
      if(card&&task.test.every(e=>equal(apply(card,e.input),e.output))) rewards.push(i);
    });
    policy.push({features:features(task.train),rewards});
  }
  save(output,policy);
} else if(cmd==='strip') {
  fs.mkdirSync(output,{recursive:true});
  for(const file of files) {
    const t=load(path.join(dir,file));
    save(path.join(output,file),{train:t.train,test:t.test.map(({input})=>({input}))});
  }
} else if(cmd==='predict') {
  const policy=model==='fixed'?null:load(model),pred={};
  for(const file of files) {
    const t=load(path.join(dir,file));
    if(t.test.some(e=>'output' in e)) throw Error('Strip test labels before inference');
    pred[file.slice(0,-5)]=solve(t.train,t.test.map(e=>e.input),policy,Number(cap));
  }
  save(output,pred);
} else if(cmd==='score') {
  const pred=load(model),rows=[];
  for(const file of files) {
    const t=load(path.join(dir,file)),id=file.slice(0,-5),p=pred[id];
    if(!p||p.length!==t.test.length) throw Error('Missing predictions '+id);
    const hits=t.test.map((e,i)=>equal(e.output,p[i].attempt_1)?1:equal(e.output,p[i].attempt_2)?2:0);
    rows.push({id,hits});
  }
  const hits=rows.flatMap(r=>r.hits),total=files.length,solved=rows.filter(r=>r.hits.every(Boolean)).length;
  const summary={tasks:total,all_outputs_pass1:rows.filter(r=>r.hits.every(x=>x===1)).length,all_outputs_pass2:solved,
    mean_task_output_pass2:rows.reduce((s,r)=>s+r.hits.filter(Boolean).length/r.hits.length,0)/total,
    test_inputs:hits.length,correct_inputs:hits.filter(Boolean).length,prediction_sha256:crypto.createHash('sha256').update(fs.readFileSync(model)).digest('hex')};
  save(output,{summary,rows});console.log(summary);
} else throw Error('Expected train, strip, predict or score');
