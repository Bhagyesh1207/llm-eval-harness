import { pipeline } from '@huggingface/transformers';
import fs from 'fs';
const items=JSON.parse(fs.readFileSync('../data/dataset.json'));
const models=process.argv.slice(2);
const PROMPTS={
 plain:{system:'You are a helpful assistant. Answer the question in one short sentence.', user:(c,q)=>`Context: ${c}\n\nQuestion: ${q}`},
 grounded:{system:'Answer the question using only the context. If the context does not contain the answer, reply exactly "Not in context." Keep it to one short sentence.', user:(c,q)=>`Context: ${c}\n\nQuestion: ${q}`}
};
const out='../results/answers.jsonl';
const done=new Set(fs.existsSync(out)?fs.readFileSync(out,'utf8').trim().split('\n').filter(Boolean).map(l=>{const r=JSON.parse(l);return r.model+'|'+r.prompt+'|'+r.id}):[]);
for (const m of models){
  const g=await pipeline('text-generation',m,{dtype:process.env.DT||'q4'});
  for (const p of Object.keys(PROMPTS)) for (const it of items){
    const k=m+'|'+p+'|'+it.id; if(done.has(k)) continue;
    const t0=Date.now();
    const o=await g([{role:'system',content:PROMPTS[p].system},{role:'user',content:PROMPTS[p].user(it.context,it.question)}],{max_new_tokens:48,do_sample:false});
    const a=o[0].generated_text.at(-1).content.trim();
    fs.appendFileSync(out,JSON.stringify({model:m,prompt:p,id:it.id,answer:a,latency_s:(Date.now()-t0)/1000})+'\n');
  }
  console.log('done',m);
}
