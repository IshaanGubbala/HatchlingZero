#!/usr/bin/env python3
"""CUDA raw-vs-fused preflight for a trained learned-gate Direct Split-V checkpoint.

This clones the supplied checkpoint for parity/timing; it never mutates it and
never treats synthetic-step timing as trained quality evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_block_gated_torch import (BDHBlockGated, BDHBlockGatedConfig,
    bdh_block_gated_annealed_direct_split_v_chunk_gla_forward, bdh_block_gated_annealed_direct_split_v_forward)


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def make(config,state,dtype):
    m=BDHBlockGated(config).to('cuda',dtype=dtype);m.attn.freqs=m.attn.freqs.float();m.load_state_dict(state);return m


def call(model,x,y,fraction,fused):
    fn=bdh_block_gated_annealed_direct_split_v_chunk_gla_forward if fused else bdh_block_gated_annealed_direct_split_v_forward
    return fn(model,x,fraction,targets=y)


def measure(config,state,x,y,fraction,dtype,fused,warmup,steps):
    m=make(config,state,dtype).train();o=torch.optim.AdamW(m.parameters(),lr=1e-3,fused=True)
    for _ in range(warmup):
        o.zero_grad(set_to_none=True);_,loss=call(m,x,y,fraction,fused);loss.backward();o.step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for _ in range(steps):
        o.zero_grad(set_to_none=True);_,loss=call(m,x,y,fraction,fused);loss.backward();o.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    out={'seconds_per_step':elapsed/steps,'tokens_per_second':x.numel()*steps/elapsed,'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'last_loss':float(loss.detach()),'finite_loss':bool(torch.isfinite(loss)),'finite_gradients':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())}
    del m,o;torch.cuda.empty_cache();return out


def measure_transformer(x,y,dtype,warmup,steps,seed):
    torch.manual_seed(seed);c=MatchedTransformerConfig({'vocab_size':256,'d_model':512,'num_layers':6,'num_heads':4,'head_dim':128,'d_ff':2048,'use_rope':True});m=MatchedTransformerLM(c).to('cuda',dtype=dtype).train();o=torch.optim.AdamW(m.parameters(),lr=1e-3,fused=True)
    def step():
        o.zero_grad(set_to_none=True);logits=m(x);loss=torch.nn.functional.cross_entropy(logits.view(-1,256),y.view(-1));loss.backward();o.step();return loss
    for _ in range(warmup):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for _ in range(steps):loss=step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    out={'seconds_per_step':elapsed/steps,'tokens_per_second':x.numel()*steps/elapsed,'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'last_loss':float(loss.detach()),'finite_loss':bool(torch.isfinite(loss)),'finite_gradients':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())}
    count=sum(p.numel() for p in m.parameters());del m,o;torch.cuda.empty_cache();return out,count


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--sequence-length',type=int,default=256);p.add_argument('--active-fraction',type=float,default=.5);p.add_argument('--warmup',type=int,default=5);p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=7);p.add_argument('--n-layer',type=int,default=4);p.add_argument('--block-size',type=int,default=16);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('requires CUDA and flash-linear-attention/Triton')
    if not 0<a.active_fraction<1:raise ValueError('active fraction must be strictly sparse')
    c=BDHBlockGatedConfig(n_layer=a.n_layer,n_embd=512,n_head=8,mlp_internal_dim_multiplier=32,vocab_size=256,dropout=0.,block_size=a.block_size)
    blob=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state=blob['model'] if 'model' in blob else blob
    dtype=torch.bfloat16;x=torch.randint(0,256,(a.batch_size,a.sequence_length),device='cuda');y=torch.randint(0,256,(a.batch_size,a.sequence_length),device='cuda')
    raw_m,fused_m=make(c,state,dtype).eval(),make(c,state,dtype).eval()
    with torch.inference_mode():raw_logits,raw_loss=call(raw_m,x,y,a.active_fraction,False);fused_logits,fused_loss=call(fused_m,x,y,a.active_fraction,True)
    maxdiff=float((raw_logits-fused_logits).abs().max());lossdiff=float((raw_loss-fused_loss).abs())
    raw_m.train();fused_m.train();_,rl=call(raw_m,x,y,a.active_fraction,False);_,fl=call(fused_m,x,y,a.active_fraction,True);rl.backward();fl.backward();rg,fg=raw_m.encoder.grad,fused_m.encoder.grad;gmax=float((rg-fg).abs().max());grel=float((rg-fg).norm()/rg.norm().clamp_min(1e-12));finite=bool(torch.isfinite(rg).all() and torch.isfinite(fg).all());del raw_m,fused_m,raw_logits,fused_logits,raw_loss,fused_loss,rl,fl;torch.cuda.empty_cache()
    raw=measure(c,state,x,y,a.active_fraction,dtype,False,a.warmup,a.steps);fused=measure(c,state,x,y,a.active_fraction,dtype,True,a.warmup,a.steps);transformer,tp=measure_transformer(x,y,dtype,a.warmup,a.steps,a.seed);cp=sum(p.numel() for p in BDHBlockGated(c).parameters())
    report={'architecture':'block_gated_bdh_direct_split_v_chunk_gla_derivative','exact_bdh':False,'trained_weights':True,'claim_eligible':False,'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'device':'cuda','hardware_id':torch.cuda.get_device_name(),'dtype':'bfloat16','batch_size':a.batch_size,'sequence_length':a.sequence_length,'effective_batch_tokens':a.batch_size*a.sequence_length,'active_fraction':a.active_fraction,'parameter_count':cp,'transformer_parameter_count':tp,'parameter_ratio_to_transformer':cp/tp,'optimizer':'AdamW fused','compile_step':False,'numerical_preflight':{'max_logit_difference':maxdiff,'loss_difference':lossdiff,'encoder_gradient_max_difference':gmax,'encoder_gradient_relative_l2_difference':grel,'encoder_gradients_finite':finite},'raw':raw,'chunk_gla':fused,'matched_rope_transformer':transformer,'chunk_gla_over_raw_speed_ratio':fused['tokens_per_second']/raw['tokens_per_second'],'chunk_gla_over_raw_peak_memory_ratio':fused['peak_memory_bytes']/raw['peak_memory_bytes'],'chunk_gla_over_transformer_speed_ratio':fused['tokens_per_second']/transformer['tokens_per_second'],'chunk_gla_over_transformer_peak_memory_ratio':fused['peak_memory_bytes']/transformer['peak_memory_bytes'],'comparison_scope':'trained checkpoint parity plus synthetic-step kernel screen; not a full trained target gate'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
