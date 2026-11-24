'use strict';
const fs=require('fs'),path=require('path');
const {learnTask,search}=require('./composition');
const [cmd,dir,out,libraryFile,budget,cpuUs]=process.argv.slice(2);
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const policy=require('../results/policy.json');
const write=(file,data)=>fs.writeFileSync(file,JSON.stringify(data)+'\n');
const files=fs.readdirSync(dir).filter(f=>f.endsWith('.json')).sort();
if(cmd==='mine') {
  const lib=[];
  files.forEach((f,i)=>{
    // Exclude this task's own rewards when ordering its candidate discoveries.
    const recipe=learnTask(read(path.join(dir,f)),policy.filter((_,j)=>i!==j));
    if(recipe) lib.push({...recipe,origin:f.slice(0,-5)});
  });
  write(out,lib);console.log('Discovered cards',lib.length);
} else if(cmd==='predict') {
  const lib=libraryFile==='none'?[]:read(libraryFile),pred={},trace={};
  for(const f of files) {
    const t=read(path.join(dir,f));
    if(t.test.some(e=>'output' in e)) throw Error('Strip query answers first');
    const id=f.slice(0,-5);
    const result=search(t.train,t.test.map(e=>e.input),policy,lib,Number(budget),true,Number(cpuUs??Infinity));
    pred[id]=result.predictions;trace[id]=result.trace;
  }
  write(out,pred);write(out+'.trace.json',trace);
} else throw Error('Expected mine or predict');
