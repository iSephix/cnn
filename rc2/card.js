'use strict';
const fs=require('fs');
const {apply}=require('./base');
const {compose}=require('./composition');
// Replay one of the portable entries produced by "mine" on any ARC input grid.
const [libraryFile,origin,gridFile]=process.argv.slice(2);
const library=JSON.parse(fs.readFileSync(libraryFile));
const entry=library.find(c=>c.origin===origin);
if(!entry) throw Error('Unknown card ID');
const grid=JSON.parse(fs.readFileSync(gridFile));
if(!Array.isArray(grid)||!grid.length||grid.length>30||!Array.isArray(grid[0])||
   !grid[0].length||grid[0].length>30||!grid.every(row=>Array.isArray(row)&&row.length===grid[0].length&&
   row.every(v=>Number.isInteger(v)&&v>=0&&v<=9))) throw Error('Expected a rectangular ARC grid');
const result=apply(entry.card,grid,compose(entry.path,new WeakMap()));
if(!result) throw Error('Projection undefined for this input');
console.log(JSON.stringify(result));
