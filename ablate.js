// Setting every stored feature vector equal gives each training task equal weight.
const fs=require('fs');
const policy=JSON.parse(fs.readFileSync(process.argv[2]));
for(const row of policy) row.features.fill(0);
fs.writeFileSync(process.argv[3],JSON.stringify(policy)+'\n');
