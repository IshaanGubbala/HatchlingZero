#!/usr/bin/env python3
"""Full real-data curriculum comparison: dense BDH, FactorizedBDH r64/r128, Transformer.

The BDH arms share the canonical training-only 2->4->6->8 schedule; the
Transformer remains at its normal fixed architectural depth.  Arms run one at
a time and report validation, wall clock, allocated VRAM and energy/token.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from reference.hz0a_matched_transformer import MatchedTransformerConfig,MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_energy import TrainingEnergySampler
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_bdh import FactorizedBDH,factorized_variable_depth_forward

def read_batch(handle,batch,seq,device,epochs):
 rows=[]
 while len(rows)<batch:
  line=handle.readline()
  if not line:handle.seek(0);epochs[0]+=1;line=handle.readline()
  row=json.loads(line)
  if len(row)>=seq:rows.append(row[:seq])
 return torch.tensor(np.asarray(rows,dtype=np.int64),device=device)
def lr_at(step,total,warmup,max_lr):
 if step<warmup:return max_lr*(step+1)/max(1,warmup)
 p=min(1,max(0,(step-warmup)/max(1,total-warmup)));return max_lr*.1+max_lr*.9*.5*(1+math.cos(math.pi*p))
def depth_at(tokens,stages):
 for boundary,depth in stages:
  if tokens<boundary:return depth
 return stages[-1][1]
def parse_stages(spec):return sorted((int(x.split(':')[0]),int(x.split(':')[1])) for x in spec.split(','))
def loss_for(model,kind,idx,target,depth):
 if kind=='transformer':
  logits=model(idx);return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
 if kind=='dense':return bdh_variable_depth_forward(model,idx,depth,target)[1]
 return factorized_variable_depth_forward(model,idx,depth,target)[1]
def evaluate(model,kind,path,batch,seq,device,batches,depth):
 model.eval();epochs=[0];vals=[]
 with path.open() as f,torch.no_grad():
  for _ in range(batches):
   z=read_batch(f,batch,seq,device,epochs);vals.append(float(loss_for(model,kind,z[:,:-1].contiguous(),z[:,1:].contiguous(),depth)))
 model.train();return sum(vals)/len(vals)
def train(name,model,kind,a,stages):
 device=torch.device('cuda');steps=math.ceil(a.target_tokens/(a.batch_size*a.sequence_length));opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.1)
 epochs=[0];tokens=0;history=[];best=float('inf');sampler=TrainingEnergySampler();sampler.start();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 with a.data.open() as f:
  for step in range(steps):
   for g in opt.param_groups:g['lr']=lr_at(step,steps,a.warmup_steps,a.learning_rate)
   z=read_batch(f,a.batch_size,a.sequence_length,device,epochs);idx,target=z[:,:-1].contiguous(),z[:,1:].contiguous();depth=depth_at(tokens,stages) if kind!='transformer' else a.n_layer
   opt.zero_grad(set_to_none=True);loss=loss_for(model,kind,idx,target,depth);loss.backward();opt.step();tokens+=a.batch_size*a.sequence_length
   if (step+1)%a.eval_every==0 or step+1==steps:
    val=evaluate(model,kind,a.validation_data,a.validation_batch_size,a.sequence_length,device,a.eval_batches,stages[-1][1] if step+1==steps and kind!='transformer' else depth);best=min(best,val);history.append({'step':step+1,'tokens_seen':tokens,'depth':depth,'validation_loss':val})
 elapsed=time.perf_counter()-started;energy=sampler.stop(tokens=tokens)
 return {'name':name,'kind':kind,'parameter_count':sum(p.numel() for p in model.parameters()),'tokens_seen':tokens,'steps':steps,'epochs':epochs[0],'best_validation_loss':best,'final_validation_loss':history[-1]['validation_loss'],'validation_history':history,'training_seconds':elapsed,'tokens_per_second':tokens/elapsed,'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),**energy}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/packed/hz0h_bytes_25m_train.jsonl'));p.add_argument('--validation-data',type=Path,default=Path('data/packed/hz0h_bytes_25m_val.jsonl'));p.add_argument('--out',type=Path,required=True);p.add_argument('--target-tokens',type=int,default=25_000_000);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--sequence-length',type=int,default=256);p.add_argument('--validation-batch-size',type=int,default=12);p.add_argument('--eval-every',type=int,default=200);p.add_argument('--eval-batches',type=int,default=10);p.add_argument('--warmup-steps',type=int,default=100);p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--seed',type=int,default=7);p.add_argument('--n-embd',type=int,default=512);p.add_argument('--n-layer',type=int,default=8);p.add_argument('--n-head',type=int,default=8);p.add_argument('--mlp-internal-dim-multiplier',type=int,default=32);p.add_argument('--curriculum-stages',default='6250000:2,12500000:4,18750000:6,25000000:8');a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('requires CUDA')
 stages=parse_stages(a.curriculum_stages);device=torch.device('cuda');dtype=torch.bfloat16;bc=BDHConfig(n_layer=a.n_layer,n_embd=a.n_embd,n_head=a.n_head,mlp_internal_dim_multiplier=a.mlp_internal_dim_multiplier,vocab_size=256,dropout=0.0);fc=HZ0IBDHConfig(n_layer=a.n_layer,n_embd=a.n_embd,n_head=a.n_head,mlp_internal_dim_multiplier=a.mlp_internal_dim_multiplier,vocab_size=256,dropout=0.0);tc=MatchedTransformerConfig({'vocab_size':256,'d_model':a.n_embd,'num_layers':6,'num_heads':4,'head_dim':a.n_embd//4,'d_ff':2048,'use_rope':True})
 arms=[]
 for name,kind,rank in [('dense_bdh','dense',None),('factorized_rank64','factorized',64),('factorized_rank128','factorized',128),('matched_transformer','transformer',None)]:
  torch.manual_seed(a.seed)
  model=(BDH(bc) if kind=='dense' else FactorizedBDH(fc,rank) if kind=='factorized' else MatchedTransformerLM(tc)).to(device=device,dtype=dtype)
  if hasattr(model,'attn'):model.attn.freqs=model.attn.freqs.to(torch.float32)
  arms.append(train(name,model,kind,a,stages));del model;torch.cuda.empty_cache()
 report={'device':'cuda','hardware':torch.cuda.get_device_name(device),'dtype':'bfloat16','curriculum_stages':stages,'target_tokens':a.target_tokens,'arms':arms};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
