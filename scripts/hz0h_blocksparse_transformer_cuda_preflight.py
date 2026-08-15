#!/usr/bin/env python3
"""CUDA synthetic-step screen: exact-parameter BlockBDH versus RoPE Transformer.

This is an optimizer-inclusive systems probe only. It intentionally creates one
arm at a time so CUDA peak allocation is not inflated by inactive controls.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward, compute_active_blocks
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize()
    elif device.type == 'mps': torch.mps.synchronize()

def clear(device):
    if device.type == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def peak(device):
    return int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else (int(torch.mps.current_allocated_memory()) if device.type == 'mps' else None)

def measure(model, step, tokens, *, warmup, steps, device, fused):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,fused=fused)
    model.train()
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True);loss=step();loss.backward();opt.step()
    sync(device);clear(device);sync(device);started=time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True);loss=step();loss.backward();opt.step()
    sync(device);elapsed=time.perf_counter()-started
    return {'steps':steps,'tokens_per_second':tokens.numel()*steps/elapsed,'seconds_per_step':elapsed/steps,'peak_memory_bytes':peak(device),'last_loss':float(loss.detach()),'finite_loss':bool(torch.isfinite(loss)),'finite_gradients':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu');p.add_argument('--batch-size',type=int,default=12);p.add_argument('--sequence-length',type=int,default=256);p.add_argument('--steps',type=int,default=20);p.add_argument('--warmup',type=int,default=5);p.add_argument('--active-fraction',type=float,default=.5);p.add_argument('--block-size',type=int,default=16);p.add_argument('--dtype',choices=('float32','float16','bfloat16'),default='bfloat16');p.add_argument('--fused-optimizer',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dev=torch.device(a.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if dev.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if a.fused_optimizer and dev.type!='cuda':raise ValueError('fused optimizer requires CUDA')
    dtype={'float32':torch.float32,'float16':torch.float16,'bfloat16':torch.bfloat16}[a.dtype]
    torch.manual_seed(7);tokens=torch.randint(0,256,(a.batch_size,a.sequence_length),device=dev);targets=torch.randint(0,256,tokens.shape,device=dev)
    bc=BDHConfig(n_layer=8,n_embd=512,n_head=8,mlp_internal_dim_multiplier=32,vocab_size=256,dropout=0.0)
    tc=MatchedTransformerConfig({'vocab_size':256,'d_model':512,'num_layers':6,'num_heads':4,'head_dim':128,'d_ff':2048,'use_rope':True})
    torch.manual_seed(7);dense=BDH(bc).to(dev,dtype);dense.attn.freqs=dense.attn.freqs.float();bp=sum(q.numel() for q in dense.parameters());dense_result=measure(dense,lambda:dense(tokens,targets=targets)[1],tokens,warmup=a.warmup,steps=a.steps,device=dev,fused=a.fused_optimizer);del dense;clear(dev)
    torch.manual_seed(7);sparse=BDH(bc).to(dev,dtype);sparse.attn.freqs=sparse.attn.freqs.float()
    def sparse_step():
        active=compute_active_blocks(sparse,tokens,block_size=a.block_size,active_fraction=a.active_fraction)
        return bdh_blocksparse_forward(sparse,tokens,active,block_size=a.block_size,targets=targets)[1]
    sparse_result=measure(sparse,sparse_step,tokens,warmup=a.warmup,steps=a.steps,device=dev,fused=a.fused_optimizer);del sparse;clear(dev)
    torch.manual_seed(7);transformer=MatchedTransformerLM(tc).to(dev,dtype);tp=sum(q.numel() for q in transformer.parameters())
    transformer_result=measure(transformer,lambda:F.cross_entropy(transformer(tokens).reshape(-1,256),targets.reshape(-1)),tokens,warmup=a.warmup,steps=a.steps,device=dev,fused=a.fused_optimizer);del transformer;clear(dev)
    ratio=bp/tp
    out={'architecture':'block_bdh_derivative','exact_bdh':False,'claim_eligible':False,'trained_weights':False,'device':str(dev),'dtype':a.dtype,'batch_size':a.batch_size,'sequence_length':a.sequence_length,'effective_batch_tokens':tokens.numel(),'optimizer':'AdamW','fused_optimizer':a.fused_optimizer,'compile_step':False,'block_size':a.block_size,'active_fraction':a.active_fraction,'parameter_count':bp,'transformer_parameter_count':tp,'parameter_ratio_to_transformer':ratio,'parameter_match':.9901<=ratio<=1.01,'dense_bdh':dense_result,'blocksparse':sparse_result,'matched_rope_transformer':transformer_result,'blocksparse_over_dense_speed_ratio':sparse_result['tokens_per_second']/dense_result['tokens_per_second'],'blocksparse_over_dense_peak_memory_ratio':sparse_result['peak_memory_bytes']/dense_result['peak_memory_bytes'] if sparse_result['peak_memory_bytes'] and dense_result['peak_memory_bytes'] else None,'blocksparse_over_transformer_speed_ratio':sparse_result['tokens_per_second']/transformer_result['tokens_per_second'],'blocksparse_over_transformer_peak_memory_ratio':sparse_result['peak_memory_bytes']/transformer_result['peak_memory_bytes'] if sparse_result['peak_memory_bytes'] and transformer_result['peak_memory_bytes'] else None,'comparison_scope':'untrained synthetic optimizer-step screen only; no quality eligibility'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
