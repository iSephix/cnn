'use strict';
const assert=require('node:assert/strict');
const {apply,equal}=require('./base');
const {compose,pairs,search}=require('./composition');
const library=require('./results/library.json');
const fs=require('fs'),path=require('path');
const dir=process.argv[2];
assert.equal(new Set(pairs.map(p=>p.join('/'))).size,2051);
const before=JSON.stringify(library);
for(const entry of JSON.parse(before)) {
  const task=JSON.parse(fs.readFileSync(path.join(dir,entry.origin+'.json')));
  const route=compose(entry.path,new WeakMap());
  for(const e of [...task.train,...task.test]) assert.ok(equal(apply(entry.card,e.input,route),e.output));
  search(task.train,task.test.map(e=>e.input),require('../results/policy.json'),library,64);
}
assert.equal(JSON.stringify(library),before);
console.log('All serialized cards replayed exactly; reuse preserved their weights.');
