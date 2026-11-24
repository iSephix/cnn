'use strict';
const {routes,fit,apply,equal,features,order}=require('./base');
const byName=new Map(routes.map(r=>[r.name,r]));
const pairs=[];
for(const a of routes.filter(r=>r.mode==='color'&&r.name!=='symmetry0')) {
  for(const b of routes.filter(r=>r.name!=='symmetry0')) {
    if(a.name.startsWith('symmetry')&&b.name.startsWith('symmetry')) continue;
    pairs.push([a.name,b.name]);
  }
}
function compose(path,cache) {
  const [a,b]=path.map(name=>byName.get(name));
  function project(g) {
    let projections=cache.get(g);
    if(!projections) {projections=new Map();cache.set(g,projections);}
    if(!projections.has(a.name)) projections.set(a.name,a.project(g));
    const mid=projections.get(a.name);
    return mid?b.project(mid):null;
  }
  return {name:path.join('/'),mode:b.mode,project};
}
function orderPairs(train,policy) {
  const ranks=new Map(order(train,policy).map((r,i)=>[r.name,i]));
  return pairs.slice().sort((a,b)=>ranks.get(a[0])+ranks.get(a[1])-ranks.get(b[0])-ranks.get(b[1]));
}
function learnTask(t,policy) {
  const exact=(card,route)=>card&&t.test.every(e=>equal(apply(card,e.input,route),e.output));
  if(routes.some(r=>exact(fit(r,t.train),r))) return null;
  const cache=new WeakMap();
  for(const path of orderPairs(t.train,policy)) {
    const route=compose(path,cache),card=fit(route,t.train);
    if(exact(card,route)) return {path,card,features:features(t.train)};
  }
  return null;
}
function search(train,inputs,policy,library,budget,record=false,cpuUs=Infinity) {
  const start=process.cpuUsage();
  const late=()=>{if(cpuUs===Infinity) return false; const t=process.cpuUsage(start);return t.user+t.system>cpuUs;};
  const predictions=inputs.map(()=>[]),cache=new WeakMap(), trace=[];
  function offer(card,route,source) {
    if(!card||late()) return;
    for(let i=0;i<inputs.length;i++) {
      if(predictions[i].length===2) continue;
      const y=apply(card,inputs[i],route);
      if(y&&!late()&&!predictions[i].some(p=>equal(p,y))) {
        predictions[i].push(y);
        if(record) trace.push({input:i,attempt:predictions[i].length,source,route:card.route});
      }
    }
  }
  for(const r of order(train,policy)) {if(late()) break;offer(fit(r,train),r,'single');}
  const x=features(train);
  const distance=f=>x.reduce((s,v,i)=>s+(v-f[i])**2,0);
  const ranked=library.slice().sort((a,b)=>distance(a.features)-distance(b.features));
  let used=0;
  const tested=new Set();
  // One full-card validation or one fitted pair costs one candidate trial.
  for(const saved of ranked) {
    if(used>=budget||late()) break;
    used++;
    const route=compose(saved.path,cache);
    if(train.every(e=>equal(apply(saved.card,e.input,route),e.output))) offer(saved.card,route,'frozen');
  }
  // Compose each frozen learned operation with a new spatial projection and readout.
  const prefixRoutes=order(train,policy);
  for(const tail of prefixRoutes.slice(0,3)) {
    for(const saved of ranked) {
      if(used>=budget||late()) break;
      used++;
      const prefix=compose(saved.path,cache),memo=new WeakMap();
      const route={name:saved.card.route+'>'+tail.name,mode:tail.mode,project(g) {
        if(!memo.has(g)) memo.set(g,apply(saved.card,g,prefix));
        const z=memo.get(g);
        return z?tail.project(z):null;
      }};
      offer(fit(route,train),route,'frozen-prefix');
    }
  }
  for(const path of [...ranked.map(s=>s.path),...(late()?[]:orderPairs(train,policy))]) {
    const key=path.join('/');
    if(tested.has(key)) continue;
    if(used>=budget||late()) break;
    used++;tested.add(key);
    const route=compose(path,cache);
    offer(fit(route,train),route,ranked.some(s=>s.path.join('/')===key)?'retrieved-path':'fresh');
  }
  return {trace, predictions:predictions.map((a,i)=>({attempt_1:a[0]??inputs[i],attempt_2:a[1]??inputs[i]}))};
}
module.exports={learnTask,search,compose,pairs};
