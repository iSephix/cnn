/* Concept Engine v0.2. Bounded membership-query learning; no target metadata. */
(function(root) {
  'use strict';
  function rng(seed) {
    let state=seed>>>0;
    return () => {state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  }
  function rowLayer(rows) {
    const out={rows:rows.length,ptr:[0],idx:[],w:[],bias:[]};
    for(const row of rows){for(const [i,w] of row.entries){out.idx.push(i);out.w.push(w);}out.ptr.push(out.idx.length);out.bias.push(row.bias);}
    return out;
  }
  // An ANF polynomial is XOR over AND monomials. Empty term means constant 1.
  function compile(terms,width) {
    if(!Number.isInteger(width)||width<1||width>288)throw Error('Invalid input width');
    const keys=new Set();
    for(const term of terms){
      if(!Array.isArray(term)||term.length>2||term.some((v,i)=>!Number.isInteger(v)||v<0||v>=width||(i>0&&v<=term[i-1])))throw Error('Invalid polynomial term');
      const key=term.join(',');if(keys.has(key))throw Error('Duplicate polynomial term');keys.add(key);
    }
    const layers=[rowLayer(terms.length?terms.map(t=>({entries:t.map(i=>[i,1]),bias:.5-t.length})):[{entries:[],bias:-.5}])];
    let count=Math.max(terms.length,1);
    while(count>1){
      const first=[],second=[];
      for(let i=0;i+1<count;i+=2){
        first.push({entries:[[i,1],[i+1,1]],bias:-.5},{entries:[[i,1],[i+1,1]],bias:-1.5});
        second.push({entries:[[i,1],[i+1,-1]],bias:-.5});
      }
      if(count%2){first.push({entries:[[count-1,1]],bias:-.5});second.push({entries:[[count-1,1]],bias:-.5});}
      layers.push(rowLayer(first),rowLayer(second));count=Math.ceil(count/2);
    }
    return [{repeats:1,layers}];
  }
  function evaluate(terms,x){let y=0;for(const term of terms){let product=1;for(const index of term)product&=x[index];y^=product;}return y;}
  function evalMatrix(block,input){let a=input;for(let rep=0;rep<block.repeats;rep++){for(const layer of block.layers){const b=new Uint8Array(layer.rows);for(let r=0;r<layer.rows;r++){let sum=layer.bias[r];for(let j=layer.ptr[r];j<layer.ptr[r+1];j++)sum+=layer.w[j]*a[layer.idx[j]];b[r]=sum>0?1:0;}a=b;}}return a;}
  function predict(blocks,x){for(const block of blocks)x=evalMatrix(block,x);return x[0];}
  function testInput(width,i,random){
    const x=new Uint8Array(width),mode=i%8;
    if(width<=8){for(let j=0;j<width;j++)x[j]=(i>>j)&1;return x;}
    if(mode===0)x.fill(0);else if(mode===1)x.fill(1);
    else if(mode===2||mode===3||mode===4){const density=mode===2?.5:mode===3?.08:.92;for(let j=0;j<width;j++)x[j]=random()<density?1:0;}
    else if(mode===5){for(let j=0;j<width;j++)x[j]=(j+i)%2;}
    else if(mode===6){for(let j=0;j<width/2;j++){x[j]=random()<.5?1:0;x[width-1-j]=x[j];}}
    else {for(let j=0;j<width;j+=12){const v=random()<.5?1:0;x.fill(v,j,Math.min(j+12,width));}}
    return x;
  }
  async function learn(oracle,{width=288,budget=43000,maxDegree=2,validation=192,seed=1,onProgress=()=>{},yieldEvery=1024}={}){
    if(typeof oracle!=='function'||!Number.isInteger(width)||width<1||width>288||!Number.isInteger(budget)||budget<0||![1,2].includes(maxDegree)||!Number.isInteger(validation)||validation<32)throw Error('Invalid learning contract');
    let queries=0;const trace=[],checkpoints=[],random=rng(seed);
    const progress=async(event)=>{trace.push(event);onProgress({...event,queries});if(yieldEvery)await new Promise(resolve=>setTimeout(resolve,0));};
    const query=x=>{
      if(queries>=budget){const e=Error('Label budget exhausted');e.code='budget';throw e;}
      queries++;const y=oracle(Uint8Array.from(x));
      if(y!==0&&y!==1){const e=Error('Oracle returned a non-binary label');e.code='oracle_contract';throw e;}return y;
    };
    try{
      await progress({stage:'probe',message:'Querying empty world and individual pixels'});
      const x=new Uint8Array(width),constant=query(x),linear=new Uint8Array(width);
      checkpoints.push([Uint8Array.from(x),constant]);
      for(let i=0;i<width;i++){x[i]=1;const y=query(x);linear[i]=y^constant;if(i<8)checkpoints.push([Uint8Array.from(x),y]);x[i]=0;}
      for(const [probe,y] of checkpoints)if(query(probe)!==y)return {status:'abstained',reason:'inconsistent_labels',queries,trace};
      let terms=constant?[[]]:[];for(let i=0;i<width;i++)if(linear[i])terms.push([i]);
      for(let degree=1;degree<=maxDegree;degree++){
        if(degree===2){
          await progress({stage:'expand',message:'Linear hypothesis failed; probing pixel pairs'});
          for(let i=0;i<width;i++){x[i]=1;for(let j=i+1;j<width;j++){
            x[j]=1;const coefficient=query(x)^constant^linear[i]^linear[j];x[j]=0;if(coefficient)terms.push([i,j]);
            if(yieldEvery&&queries%yieldEvery===0)await progress({stage:'pairs',message:'Collecting interaction evidence'});
          }x[i]=0;}
        }
        const blocks=compile(terms,width);let correct=0,firstCounterexample=null;
        await progress({stage:'verify',degree,message:'Testing numerical candidate on fresh mixed-density worlds'});
        const validationCount=width<=8?Math.max(validation,2**width):validation;
        for(let i=0;i<validationCount;i++){
          const probe=testInput(width,i,random),expected=query(probe),got=predict(blocks,probe);
          if(got===expected)correct++;else if(!firstCounterexample)firstCounterexample={input:Array.from(probe),expected,predicted:got};
        }
        const stage={degree,correct,total:validationCount,queries};trace.push({...stage,stage:'validation'});
        if(correct===validationCount)return {status:'verified',degree,terms,blocks,width,queries,trace,validation:stage,identification_queries:1+width+(degree===2?width*(width-1)/2:0),guarantee:'Exact identification conditional on a deterministic degree-at-most-'+degree+' Boolean polynomial oracle; finite validation does not establish that assumption.'};
        if(degree===maxDegree)return {status:'abstained',reason:'hypothesis_falsified',queries,trace,counterexample:firstCounterexample};
      }
    }catch(error){if(error.code)return {status:'abstained',reason:error.code,queries,trace};throw error;}
  }
  const api=Object.freeze({rng,compile,evaluate,evalMatrix,predict,testInput,learn});
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ConceptAcquisition=api;
})(typeof globalThis!=='undefined'?globalThis:this);
