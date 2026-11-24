// Post-evaluation diagnostic. Never used by prediction.
const fs=require('fs'),path=require('path');
const {routes,fit,apply,equal}=require('./solver');
const [dir,out]=process.argv.slice(2);
let fits=0,reaches=0,tasks=0,identity=0;
for(const f of fs.readdirSync(dir).filter(s=>s.endsWith('.json')).sort()) {
  const t=JSON.parse(fs.readFileSync(path.join(dir,f)));
  const cards=routes.map(r=>fit(r,t.train)).filter(Boolean);
  fits+=Number(cards.length>0);
  const copy=t.test.every(e=>equal(e.input,e.output));
  identity+=Number(copy);
  reaches+=Number(copy||cards.some(c=>t.test.every(e=>equal(apply(c,e.input),e.output))));
  tasks++;
}
const report={tasks,any_card_fits_all_demos:fits,single_fitted_card_or_identity_solves_all_queries:reaches,identity};
fs.writeFileSync(out,JSON.stringify(report)+'\n');console.log(report);
