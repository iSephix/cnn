// Retrospective oracle restricted to a single route shared by all queries.
// Never imported by the solver or by the training process.
'use strict';
const {apply,equal,fit,routes}=require('./base');
const {compose,pairs}=require('./composition');
const fs=require('fs'),path=require('path');
const [dir,out]=process.argv.slice(2);
const counts={tasks:0,fit:0,reachable:0};
for(const f of fs.readdirSync(dir).filter(x=>x.endsWith('.json')).sort()){
  const t=JSON.parse(fs.readFileSync(path.join(dir,f))),cache=new WeakMap();
  let fits=false,hits=false;
  for(const route of [...routes,...pairs.map(p=>compose(p,cache))]) {
    const card=fit(route,t.train);
    if(!card) continue;
    fits=true;
    if(t.test.every(e=>equal(apply(card,e.input,route),e.output))) {hits=true;break;}
  }
  counts.tasks++;
  counts.fit+=Number(fits);
  counts.reachable+=Number(hits);
}
fs.writeFileSync(out,JSON.stringify(counts)+'\n');console.log(counts);
