'use strict';
const {evalMatrix} = require('./acquisition');
const palette = Array.from({length:10}, (_,i)=>i);
const shape = g=>[g.length,g[0].length];
const grid = (h,w,fn)=>Array.from({length:h},(_,r)=>Array.from({length:w},(_,c)=>fn(r,c)));
const equal = (a,b)=>JSON.stringify(a)===JSON.stringify(b);
function counts(g) {
  const n=Array(10).fill(0);
  for(const row of g) for(const v of row) n[v]++;
  return n;
}
function colors(g) {
  const n=counts(g);
  return palette.filter(i=>n[i]).sort((a,b)=>n[b]-n[a]||a-b);
}
function components(g,bg) {
  const [h,w]=shape(g), seen=new Set(), groups=[];
  for(let r=0;r<h;r++) for(let c=0;c<w;c++) {
    if(g[r][c]===bg||seen.has(r*w+c)) continue;
    const group=[[r,c]];
    seen.add(r*w+c);
    for(let i=0;i<group.length;i++) for(const [dy,dx] of [[0,1],[1,0],[0,-1],[-1,0]]) {
      const y=group[i][0]+dy,x=group[i][1]+dx;
      if(y>=0&&x>=0&&y<h&&x<w&&g[y][x]!==bg&&!seen.has(y*w+x)) {
        seen.add(y*w+x); group.push([y,x]);
      }
    }
    groups.push(group);
  }
  return groups.sort((a,b)=>b.length-a.length);
}
function crop(g,points) {
  if(!points.length) return null;
  const ys=points.map(p=>p[0]),xs=points.map(p=>p[1]);
  return g.slice(Math.min(...ys),Math.max(...ys)+1).map(row=>row.slice(Math.min(...xs),Math.max(...xs)+1));
}
const routes=[];
function add(name,project,mode='color') {routes.push({name,project,mode});}
for(let k=0;k<8;k++) add('symmetry'+k,g=>{
  const [h,w]=shape(g),t=k&4;
  return grid(t?w:h,t?h:w,(y,x)=>{
    const r=t?x:y,c=t?y:x; return g[k&1?h-1-r:r][k&2?w-1-c:c];
  });
});
for(const axis of [0,1]) for(const gap of [0,1]) for(const mode of ['pair','first','second']) {
  add('split-'+[axis,gap,mode].join('-'),g=>{
    const [h,w]=shape(g),n=(axis?w:h)-gap;
    if(n<2||n%2) return null;
    const mid=n/2;
    return grid(axis?h:mid,axis?mid:w,(r,c)=>{
      const a=g[r][c],b=g[r+(axis?0:mid+gap)][c+(axis?mid+gap:0)];
      return mode==='pair'?10*a+b:mode==='first'?a:b;
    });
  },mode==='pair'?'pair':'color');
}
for(const bg of ['zero','mode']) for(const which of ['all','largest','smallest']) add('crop-'+bg+'-'+which,g=>{
  const groups=components(g,bg==='zero'?0:colors(g)[0]);
  return crop(g,which==='all'?groups.flat():which==='largest'?groups[0]||[]:groups.at(-1)||[]);
});
for(const color of palette) add('color-box-'+color,g=>{
  const points=[];
  g.forEach((row,r)=>row.forEach((v,c)=>{if(v===color) points.push([r,c]);}));
  return crop(g,points);
});
for(const k of [2,3,4]) for(const type of ['repeat','tile','subsample']) add(type+k,g=>{
  const [h,w]=shape(g);
  if(type==='subsample') return h%k||w%k?null:grid(h/k,w/k,(r,c)=>g[r*k][c*k]);
  if(h*k>30||w*k>30) return null;
  return grid(h*k,w*k,(r,c)=>type==='tile'?g[r%h][c%w]:g[Math.floor(r/k)][Math.floor(c/k)]);
});
for(const rank of [0,-1]) add('frequency'+rank,g=>[[colors(g).at(rank)]]);
for(const [dy,dx] of [[0,1],[1,0],[0,-1],[-1,0]]) add('neighbor-'+[dy,dx].join(','),g=>
  grid(...shape(g),(r,c)=>10*g[r][c]+(g[r+dy]?.[c+dx]??0)),'pair');

// Each row of the learned threshold layer selects input categories for one color.
function fit(route,examples) {
  const size=route.mode==='pair'?100:10, mapping=new Int8Array(size).fill(-1);
  for(const {input,output} of examples) {
    const z=route.project(input);
    if(!z||z.length!==output.length||z[0].length!==output[0].length) return null;
    for(let r=0;r<z.length;r++) for(let c=0;c<z[0].length;c++) {
      const v=z[r][c],y=output[r][c];
      if(mapping[v]>=0&&mapping[v]!==y) return null;
      mapping[v]=y;
    }
  }
  const idx=[],ptr=[0];
  for(const y of palette) {
    for(let v=0;v<size;v++) {
      const fallback=route.mode==='pair'?Math.floor(v/10):v;
      if((mapping[v]<0?fallback:mapping[v])===y) idx.push(v);
    }
    ptr.push(idx.length);
  }
  return {route:route.name,width:size,block:{repeats:1,layers:[{rows:10,ptr,idx,w:idx.map(()=>1),bias:palette.map(()=>-0.5)}]}};
}
function apply(card,g) {
  const z=routes.find(t=>t.name===card.route).project(g);
  if(!z) return null;
  const cache=new Map();
  return z.map(row=>row.map(v=>{
    if(!cache.has(v)) {
      const x=new Uint8Array(card.width);x[v]=1;
      cache.set(v,evalMatrix(card.block,x).indexOf(1));
    }
    return cache.get(v);
  }));
}
function features(examples) {
  const rows=examples.map(({input,output})=>{
    const [h,w]=shape(input),[a,b]=shape(output);
    return [Math.log2(a/h),Math.log2(b/w),+(h===a&&w===b),+(a===b),+(h===w),colors(input).length/5,colors(output).length/5,+(a*b===1),Math.log2(a*b)/8];
  });
  return rows[0].map((_,j)=>rows.reduce((s,r)=>s+r[j],0)/rows.length);
}
// Kernel regression on training tasks. Each target is an operator's exact success.
function order(examples,policy) {
  if(!policy) return routes;
  const x=features(examples),scores=routes.map(()=>0);
  for(const {features:f,rewards} of policy) {
    const similarity=Math.exp(-3*x.reduce((s,v,i)=>s+(v-f[i])**2,0));
    for(const i of rewards) scores[i]+=similarity;
  }
  return routes.map((r,i)=>({r,i})).sort((a,b)=>scores[b.i]-scores[a.i]||a.i-b.i).map(o=>o.r);
}
function solve(examples,inputs,policy,budget) {
  const out=inputs.map(()=>[]);
  for(const route of order(examples,policy).slice(0,budget)) {
    const card=fit(route,examples);
    if(!card) continue;
    inputs.forEach((g,i)=>{
      if(out[i].length>=2) return;
      const y=apply(card,g);
      if(y&&!out[i].some(a=>equal(a,y))) out[i].push(y);
    });
    if(out.every(x=>x.length===2)) break;
  }
  return out.map((a,i)=>{
    if(a.length<2&&!a.some(y=>equal(y,inputs[i]))) a.push(inputs[i]);
    return {attempt_1:a[0],attempt_2:a[1]??a[0]};
  });
}
module.exports={routes,features,fit,apply,solve,equal};
