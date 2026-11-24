// Synthetic downstream tasks generated independently of ARC grids. Each target applies
// an ARC-mined card and then a novel color permutation. All cards and tasks are retained.
'use strict';
const fs=require('fs');
const {rng}=require('../acquisition');
const {apply,equal}=require('./base');
const {search,compose}=require('./composition');
const lib=require('./results/library.json'),policy=require('../results/policy.json');
const rows=[];
for(const [i,saved] of lib.entries()) for(let seed=0;seed<8;seed++) {
  const random=rng(1000*i+seed),prefix=compose(saved.path,new WeakMap());
  const shift=1+seed;
  function sample() {
    const h=4+2*Math.floor(random()*4),w=4+2*Math.floor(random()*4);
    const bg=Math.floor(random()*5);
    const g=Array.from({length:h},()=>Array(w).fill(bg));
    for(let r=1;r<h-1;r++) for(let c=1;c<w-1;c++) if(random()<0.58) g[r][c]=Math.floor(random()*6);
    const z=apply(saved.card,g,prefix);
    if(!z) return null;
    return {input:g,output:z.map(row=>row.map(v=>(v+shift)%10))};
  }
  const examples=[];
  for(let j=0;j<100&&examples.length<8;j++) {const e=sample();if(e) examples.push(e);}
  if(examples.length!==8) throw Error('Insufficient valid examples');
  const train=examples.slice(0,5),queries=examples.slice(5),scores={};
  for(const [mode,memory] of [['fresh',[]],['reuse',lib]]) {
    const {predictions,trace}=search(train,queries.map(e=>e.input),policy,memory,64,true);
    const hits=queries.map((e,j)=>Object.values(predictions[j]).some(g=>equal(e.output,g)));
    scores[mode]={solved:hits.every(Boolean),hits,trace};
  }
  rows.push({origin:saved.origin,seed,...scores});
}
fs.writeFileSync(process.argv[2],JSON.stringify(rows)+'\n');
for(const mode of ['fresh','reuse']) console.log(mode,rows.filter(r=>r[mode].solved).length,'/',rows.length);
