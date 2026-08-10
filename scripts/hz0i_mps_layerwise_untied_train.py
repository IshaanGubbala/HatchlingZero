import argparse,json,time,random
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_layerwise_untied import FactorizedLayerwiseBDH
from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed',type=int,default=31);p.add_argument('--steps',type=int,default=100);p.add_argument('--seq-len',type=int,default=64);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--batch-policy',choices=['weighted','stratified'],default='weighted');p.add_argument('--sampler-temperature',type=float,default=1.0);p.add_argument('--sampler-min-weight',type=float,default=.05);p.add_argument('--sampler-warmup',type=int,default=0);p.add_argument('--grad-accum',type=int,default=1);p.add_argument('--grad-clip',type=float,default=1.0);p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32');p.add_argument('--ce-mode',choices=['dense','chunked'],default='dense');p.add_argument('--ce-chunk',type=int,default=4096);p.add_argument('--compile',action='store_true');p.add_argument('--lora',action='store_true');p.add_argument('--lora-rank',type=int,default=64);p.add_argument('--stride',type=int,default=2);p.add_argument('--stride-warmup',type=int,default=0);p.add_argument('--capability-warmup',type=int,default=0);p.add_argument('--out',type=Path,required=True);p.add_argument('--moe-capacity',type=float,default=None);p.add_argument('--capacity-warmup',type=int,default=0);p.add_argument('--moe-routing',choices=['top1','top2','adaptive','balanced'],default='top1');p.add_argument('--fallback-threshold',type=float,default=0.0);p.add_argument('--moe-aux-weight',type=float,default=0.0);p.add_argument('--moe-z-weight',type=float,default=0.0);p.add_argument('--learned-triggers',action='store_true');p.add_argument('--trigger-aux-weight',type=float,default=0.0);p.add_argument('--trigger-threshold',type=float,default=0.5);p.add_argument('--trigger-mode',choices=['threshold','topk'],default='threshold');p.add_argument('--trigger-fraction',type=float,default=.0625);p.add_argument('--trigger-fraction-start',type=float,default=None);p.add_argument('--trigger-fraction-warmup',type=int,default=0);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--warmup-steps',type=int,default=0);p.add_argument('--min-lr-ratio',type=float,default=1.0);p.add_argument('--moe-balanced-init',action='store_true');p.add_argument('--moe-router-noise',type=float,default=0.0);p.add_argument('--moe-aux-warmup',type=int,default=0);p.add_argument('--resume',type=Path,default=None);p.add_argument('--checkpoint-every',type=int,default=0);p.add_argument('--diagnostics-every',type=int,default=1);p.add_argument('--trace-every',type=int,default=0);p.add_argument('--trace-out',type=Path,default=None);a=p.parse_args();random.seed(a.seed);torch.manual_seed(a.seed);dev='mps' if torch.backends.mps.is_available() else 'cpu';spec=json.loads(a.manifest.read_text());sampler=AdaptiveKnowledgeSampler(spec['paths'],spec.get('weights'),seed=a.seed,temperature=a.sampler_temperature,min_weight=a.sampler_min_weight);c=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True,moe_capacity_factor=a.moe_capacity,moe_routing=a.moe_routing,moe_fallback_threshold=a.fallback_threshold,moe_aux_weight=a.moe_aux_weight,moe_z_weight=a.moe_z_weight,moe_balanced_init=a.moe_balanced_init,moe_router_noise=a.moe_router_noise,learned_triggers=a.learned_triggers,trigger_aux_weight=a.trigger_aux_weight,trigger_threshold=a.trigger_threshold,trigger_mode=a.trigger_mode,trigger_fraction=a.trigger_fraction);dtype=getattr(torch,a.dtype);m=FactorizedLayerwiseBDH(c,704,a.stride).to(device=dev,dtype=dtype);m.attn.freqs=m.attn.freqs.float()
 if a.lora:
  for pp in m.parameters(): pp.requires_grad=False
  ra=a.lora_rank;H=m.config.n_head;D=m.config.n_embd;N=D*m.config.mlp_internal_dim_multiplier//H
  def lp(shape): return torch.nn.Parameter(torch.zeros(shape,device=dev,dtype=dtype))
  m.lora_enc_a=lp((H,D,ra));m.lora_enc_b=lp((H,ra,N));m.lora_val_a=lp((H,D,ra));m.lora_val_b=lp((H,ra,N));m.lora_dec_a=lp((H,N,ra));m.lora_dec_b=lp((H,ra,D))
  with torch.no_grad():
   for aa in ('lora_enc_a','lora_val_a','lora_dec_a'): getattr(m,aa).normal_(std=.01)
  def enc_lo(x,l,r):
   z=torch.einsum('bhtd,hdr->bhtr',x,l);z=torch.einsum('bhtr,hrn->bhtn',z,r)
   return z+torch.einsum('bhtd,hda,han->bhtn',x,m.lora_enc_a,m.lora_enc_b)
  def val_lo(x,l,r):
   z=torch.einsum('bhtd,hdr->bhtr',x,l);z=torch.einsum('bhtr,hrn->bhtn',z,r)
   return z+torch.einsum('bhtd,hda,han->bhtn',x,m.lora_val_a,m.lora_val_b)
  def dec_lo(x):
   z=torch.einsum('bhtn,hnr->bhtr',x,m.dec_l);z=torch.einsum('bhtr,hrd->bhtd',z,m.dec_r).sum(1,keepdim=True)
   return z+torch.einsum('bhtn,hna,had->bhtd',x,m.lora_dec_a,m.lora_dec_b).sum(1,keepdim=True)
  import types as _t
  def fh(self,idx,layer_hook=None):
   C=self.config;B,TT=idx.shape;x=self.ln(self.embed(idx).unsqueeze(1))
   for level in range(C.n_layer):
    xs=torch.relu(enc_lo(x,self.enc_l,self.enc_r));ykv=self.ln(self.attn(Q=xs,K=xs,V=x));ys=torch.relu(val_lo(ykv,self.val_l,self.val_r))
    x=self.ln(x+self.ln(dec_lo(self.drop(xs*ys))))
    if layer_hook is not None and level%self.layer_stride==0: x=layer_hook(x,level)
   return x.view(B,TT,C.n_embd)
  m.forward_hidden=_t.MethodType(fh,m)
  for name,pp in m.named_parameters():
   if name.startswith('lora_') or name=='lm_head' or name.endswith('_gate'): pp.requires_grad=True
 if a.compile and dev!='cpu':
  try:m=torch.compile(m,backend='aot_eager')
  except Exception as ex:a.compile=False;print('compile disabled:',ex)
 o=torch.optim.AdamW([pp for pp in m.parameters() if pp.requires_grad],lr=a.lr);start_step=0
 if a.resume is not None:
  q=torch.load(a.resume,map_location='cpu',weights_only=False);m.load_state_dict(q['model'],strict=not a.lora);start_step=int(q.get('step',0));
  if q.get('optimizer') is not None and not a.lora:o.load_state_dict(q['optimizer'])
  if q.get('sampler') is not None:sampler.load_state_dict(q['sampler'])
 ls=[];trace_records=[];counts={k:0 for k in sampler.names};expert_counts=[0,0,0,0];expert_by_domain={k:[0,0,0,0] for k in sampler.names};t=time.perf_counter()
 for local_i in range(a.steps):
  i=start_step+local_i
  if a.warmup_steps>0:scale=min(1.,(i+1)/a.warmup_steps)
  else:scale=1.
  if i>=a.warmup_steps and a.steps>a.warmup_steps:
   import math as _math;progress=(i-a.warmup_steps)/max(1,a.steps-a.warmup_steps);scale=a.min_lr_ratio+(1-a.min_lr_ratio)*.5*(1+_math.cos(_math.pi*progress))
  for group in o.param_groups:group['lr']=a.lr*scale
  if a.trigger_fraction_start is not None and a.trigger_fraction_warmup>0 and m.trigger_gate is not None:
   u=min(1.,(i+1)/a.trigger_fraction_warmup);m.trigger_gate.fraction=a.trigger_fraction_start+(a.trigger_fraction-a.trigger_fraction_start)*u
  if a.stride_warmup>0 and i>=a.stride_warmup: m.layer_stride=1
  if a.capability_warmup>0: m.capabilities_enabled=i>=a.capability_warmup
  if a.moe_aux_warmup>0: m.config.moe_aux_weight=a.moe_aux_weight*min(1.,(i+1)/a.moe_aux_warmup)
  if a.moe_capacity is not None and i==a.capacity_warmup: m.moe.capacity_factor=a.moe_capacity
  rows=(sampler.sample_stratified(a.batch_size,a.seq_len+1) if a.batch_policy=='stratified' else sampler.sample(a.batch_size,a.seq_len+1));x=torch.tensor([vals for _,vals in rows],device=dev);tr=None if a.learned_triggers else torch.zeros(a.batch_size,a.seq_len,dtype=torch.bool,device=dev);
  if tr is not None:tr[:,::8]=1
  step_trace=[] if a.trace_every>0 and (i+1)%a.trace_every==0 else None
  if a.ce_mode=='chunked':
   h,_=m(x[:,:-1],triggers=tr,targets=None,trace=step_trace,return_hidden=True)
   from reference.hz0i_memory_efficient_ce import chunked_cross_entropy
   tok_loss=torch.stack([chunked_cross_entropy(h[bi:bi+1],m.lm_head,x[bi:bi+1,1:],chunk=a.ce_chunk) for bi in range(a.batch_size)])
   l=tok_loss.mean();row_losses=tok_loss
  else:
   logits,_=m(x[:,:-1],triggers=tr,targets=None,trace=step_trace);tok_loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),x[:,1:].contiguous().reshape(-1),reduction='none').view(a.batch_size,-1);l=tok_loss.mean();row_losses=tok_loss.mean(1)
  if not torch.isfinite(l): raise FloatingPointError(f'non-finite loss at step {i} with dtype={a.dtype}; use bfloat16 or float32')
  if local_i%a.grad_accum==0:o.zero_grad(set_to_none=True)
  (l/a.grad_accum).backward()
  if ((local_i+1)%a.grad_accum==0) or local_i==a.steps-1:
   torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip);o.step()
  ls.append(float(l.detach()));
  if step_trace is not None:
   with torch.no_grad():
    pnorm=(sum(float(p.detach().float().pow(2).sum()) for p in m.parameters())**.5)
    gnorm=(sum(float(p.grad.detach().float().pow(2).sum()) for p in m.parameters() if p.grad is not None)**.5)
   trace_records.append({'step':i+1,'loss':float(l.detach()),'parameter_rms':pnorm/(sum(p.numel() for p in m.parameters())**.5),'gradient_rms':gnorm/(sum(p.numel() for p in m.parameters())**.5),'throughput_tok_s':(local_i+1)*a.seq_len*a.batch_size/max(1e-9,time.perf_counter()-t),'lr':o.param_groups[0]['lr'],'trigger_rate':None if m.trigger_gate is None else m.trigger_gate.last_rate,'expert_counts':None if m.moe is None else m.moe.last_counts.detach().cpu().tolist(),'layers':step_trace})
   if a.trace_out is not None: a.trace_out.write_text(json.dumps(trace_records,indent=2))
  for k,_ in rows: counts[k]+=1
  collect_diag=a.diagnostics_every>0 and ((i+1)%a.diagnostics_every==0)
  if collect_diag and m.moe is not None: expert_counts=[a+b for a,b in zip(expert_counts,m.moe.last_counts.detach().cpu().tolist())]
  if collect_diag and m.moe is not None and m.moe.last_choice.numel()==len(rows)*a.seq_len:
   choices=m.moe.last_choice.detach().cpu().reshape(len(rows),a.seq_len)
   for row_i,(domain,_) in enumerate(rows):
    cc=torch.bincount(choices[row_i],minlength=4).tolist();expert_by_domain[domain]=[u+v for u,v in zip(expert_by_domain[domain],cc)]
  elif collect_diag and m.moe is not None:
   expert_by_domain[rows[0][0]]=[u+v for u,v in zip(expert_by_domain[rows[0][0]],m.moe.last_counts.detach().cpu().tolist())]
  domain_loss_values={}
  for j,(domain,_) in enumerate(rows): domain_loss_values.setdefault(domain,[]).append(float(row_losses[j].detach()))
  if i>=a.sampler_warmup: sampler.update_losses({domain:sum(vals)/len(vals) for domain,vals in domain_loss_values.items()},decay=.99)
  if a.checkpoint_every and (local_i+1)%a.checkpoint_every==0:
   torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'sampler':sampler.state_dict(),'step':i+1,'config':dict(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576)},a.out.with_name(a.out.stem+f'.step{i+1}.pt'))
 if dev=='mps':torch.mps.synchronize()
 torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'sampler':sampler.state_dict(),'step':start_step+a.steps,'config':dict(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,moe_capacity_factor=a.moe_capacity,moe_routing=a.moe_routing,moe_fallback_threshold=a.fallback_threshold,moe_aux_weight=a.moe_aux_weight,moe_z_weight=a.moe_z_weight,moe_balanced_init=a.moe_balanced_init,moe_router_noise=a.moe_router_noise,learned_triggers=a.learned_triggers,trigger_aux_weight=a.trigger_aux_weight,trigger_threshold=a.trigger_threshold,trigger_mode=a.trigger_mode,trigger_fraction=a.trigger_fraction),'stride':m.layer_stride,'stride_initial':a.stride,'stride_warmup':a.stride_warmup},a.out.with_suffix('.pt'));out={'device':dev,'stride':m.layer_stride,'stride_initial':a.stride,'stride_warmup':a.stride_warmup,'capability_warmup':a.capability_warmup,'checkpoint_every':a.checkpoint_every,'diagnostics_every':a.diagnostics_every,'trace_every':a.trace_every,'trace_out':str(a.trace_out) if a.trace_out else None,'moe_capacity':a.moe_capacity,'moe_routing':a.moe_routing,'fallback_threshold':a.fallback_threshold,'moe_aux_weight':a.moe_aux_weight,'moe_z_weight':a.moe_z_weight,'moe_balanced_init':a.moe_balanced_init,'moe_router_noise':a.moe_router_noise,'moe_aux_warmup':a.moe_aux_warmup,'lr':a.lr,'warmup_steps':a.warmup_steps,'min_lr_ratio':a.min_lr_ratio,'capacity_warmup':a.capacity_warmup,'params':sum(p.numel() for p in m.parameters()),'gates':{k:float(torch.tanh(getattr(m,k)).detach()) for k in ('conditional_gate','fast_gate','moe_gate')},'trigger_rate':None if m.trigger_gate is None else m.trigger_gate.last_rate,'trigger_mode':a.trigger_mode,'trigger_fraction':a.trigger_fraction,'trigger_fraction_start':a.trigger_fraction_start,'trigger_fraction_warmup':a.trigger_fraction_warmup,'steps':a.steps,'batch_size':a.batch_size,'batch_policy':a.batch_policy,'sampler_temperature':a.sampler_temperature,'sampler_min_weight':a.sampler_min_weight,'sampler_warmup':a.sampler_warmup,'seed':a.seed,'grad_accum':a.grad_accum,'grad_clip':a.grad_clip,'dtype':a.dtype,'ce_mode':a.ce_mode,'ce_chunk':a.ce_chunk,'compile':a.compile,'lora':a.lora,'lora_rank':a.lora_rank,'global_step':start_step+a.steps,'loss_first':ls[0],'loss_last':ls[-1],'domain_counts':counts,'sampler_weights':dict(sampler.weights),'loss_ema':dict(sampler.loss_ema),'expert_counts':expert_counts,'expert_counts_by_domain':expert_by_domain,'tok_s':a.steps*a.seq_len*a.batch_size/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())};a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
