#!/usr/bin/env python3
"""Real-corpus soft-to-hard learned-gate BlockBDH training runner.

This is an explicitly labelled derivative. Dense learned gates receive gradient
before hard selected blocks are enabled by a token-threshold curriculum; this
is distinct from hard heuristic routing from step zero.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0h_bdh_block_gated_torch import (BDHBlockGated, BDHBlockGatedConfig,
    bdh_block_gated_annealed_direct_split_v_forward, bdh_block_gated_annealed_forward,
    compute_active_blocks_by_gate, compute_block_gate)
from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_energy import TrainingEnergySampler


def read_batch(handle, batch_size, seq, device, epochs=None):
    rows=[]
    while len(rows)<batch_size:
        line=handle.readline()
        if not line:
            handle.seek(0)
            if epochs is not None: epochs[0]+=1
            line=handle.readline()
        row=json.loads(line)
        if len(row)>=seq: rows.append(row[:seq])
    return torch.tensor(np.asarray(rows,dtype=np.int64),device=device)


def resolve(name):
    if name=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def peak(device):
    if device.type=='cuda': return int(torch.cuda.max_memory_allocated())
    if device.type=='mps': return int(torch.mps.current_allocated_memory())
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)


def reset_peak(device):
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()


def lr_at(step,total,warmup,max_lr):
    if warmup and step<warmup:return max_lr*(step+1)/warmup
    p=min(max((step-warmup)/max(1,total-warmup),0),1)
    return max_lr*(0.1+0.9*0.5*(1+math.cos(math.pi*p)))


def parse_stages(value):
    stages=[]
    for item in value.split(','):
        token, fraction=item.split(':')
        stages.append((int(token),float(fraction)))
    if not stages or stages[0][0]!=0: raise ValueError('--curriculum-stages must start at 0:fraction')
    if any(not 0 < f <= 1 for _,f in stages) or stages != sorted(stages): raise ValueError('stages must be sorted and fractions in (0,1]')
    return stages


def fraction_at(stages,tokens):
    return [f for threshold,f in stages if threshold<=tokens][-1]


def fingerprint(model):
    h=hashlib.sha256()
    for p in model.parameters(): h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--validation-data',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--target-tokens',type=int,default=25_000_000);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--sequence-length',type=int,default=256);p.add_argument('--validation-batch-size',type=int,default=12)
    p.add_argument('--n-embd',type=int,default=512);p.add_argument('--n-layer',type=int,default=8);p.add_argument('--n-head',type=int,default=8);p.add_argument('--mlp-internal-dim-multiplier',type=int,default=32);p.add_argument('--vocab-size',type=int,default=256);p.add_argument('--block-size',type=int,default=16)
    p.add_argument('--curriculum-stages',default='0:1.0,6250000:0.75,12500000:0.6,18750000:0.5')
    p.add_argument('--value-path',choices=('vanilla','direct_split_v'),default='vanilla',help='direct_split_v is an equal-parameter experimental derivative, not exact BDH')
    p.add_argument('--max-lr',type=float,default=1e-3);p.add_argument('--warmup-steps',type=int,default=100);p.add_argument('--weight-decay',type=float,default=.1);p.add_argument('--dtype',choices=('float32','bfloat16','float16'),default='bfloat16');p.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--checkpoint-interval',type=int,default=200);p.add_argument('--validation-interval',type=int,default=200);p.add_argument('--fused-optimizer',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();stages=parse_stages(a.curriculum_stages)
    if a.block_size<=0 or a.block_size%2:raise ValueError('block size must be positive and even')
    N=a.n_embd*a.mlp_internal_dim_multiplier//a.n_head
    if N%a.block_size:raise ValueError('block size must divide latent width')
    dev=resolve(a.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if dev.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if a.fused_optimizer and dev.type!='cuda':raise ValueError('fused optimizer requires CUDA')
    hardware=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'Apple MPS' if dev.type=='mps' else 'CPU'
    dtype={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}[a.dtype]
    a.run_dir.mkdir(parents=True,exist_ok=True);checkpoint=a.run_dir/'block_gated_checkpoint';log=a.run_dir/'block_gated_memory.jsonl'
    torch.manual_seed(a.seed);config=BDHBlockGatedConfig(n_layer=a.n_layer,n_embd=a.n_embd,n_head=a.n_head,mlp_internal_dim_multiplier=a.mlp_internal_dim_multiplier,vocab_size=a.vocab_size,dropout=0.,block_size=a.block_size)
    model=BDHBlockGated(config).to(device=dev,dtype=dtype);model.attn.freqs=model.attn.freqs.float()
    total=math.ceil(a.target_tokens/(a.batch_size*a.sequence_length));opt=torch.optim.AdamW(model.parameters(),lr=a.max_lr,weight_decay=a.weight_decay,fused=a.fused_optimizer)
    step=tokens=batch_index=0;metrics=[];best=None;epochs=[0]
    if a.resume and Path(str(checkpoint)+'.pt').exists():
        blob=torch.load(str(checkpoint)+'.pt',map_location=dev,weights_only=False);model.load_state_dict(blob['model']);opt.load_state_dict(blob['optimizer']);meta=json.loads(Path(str(checkpoint)+'.json').read_text());step,tokens,batch_index,metrics,best=meta['step'],meta['tokens_seen'],meta['batch_index'],meta['metrics'],meta.get('best_validation_loss')
    gated_forward = bdh_block_gated_annealed_direct_split_v_forward if a.value_path == 'direct_split_v' else bdh_block_gated_annealed_forward
    def forward(x,y=None,fraction=None): return gated_forward(model,x,fraction_at(stages,tokens) if fraction is None else fraction,targets=y)
    def save():
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},str(checkpoint)+'.pt');Path(str(checkpoint)+'.json').write_text(json.dumps({'step':step,'tokens_seen':tokens,'batch_index':batch_index,'metrics':metrics,'best_validation_loss':best}))
    sampler=TrainingEnergySampler();sampler.start();started=time.perf_counter()
    with a.data.open() as tr,a.validation_data.open() as va:
        for _ in range(batch_index):read_batch(tr,a.batch_size,a.sequence_length,dev)
        val=read_batch(va,a.validation_batch_size,a.sequence_length,dev);reset_peak(dev)
        while tokens<a.target_tokens:
            batch=read_batch(tr,a.batch_size,a.sequence_length,dev,epochs);xx,yy=shifted_target_batch(batch);frac=fraction_at(stages,tokens);opt.zero_grad(set_to_none=True);_,loss=forward(xx,yy,frac);loss.backward()
            lr=lr_at(step,total,a.warmup_steps,a.max_lr)
            for group in opt.param_groups:group['lr']=lr
            opt.step();step+=1;batch_index+=1;tokens+=a.batch_size*a.sequence_length
            item={'step':step,'tokens_seen':tokens,'loss':float(loss.detach()),'active_fraction':frac,'lr':lr,'wall_time':time.perf_counter()-started,'peak_memory_bytes':peak(dev),'epoch_or_data_pass':epochs[0]}
            if step%a.validation_interval==0 or tokens>=a.target_tokens:
                model.eval()
                with torch.no_grad():
                    vx,vy=shifted_target_batch(val);_,vl=forward(vx,vy,frac);g=compute_block_gate(model,model.ln(model.embed(vx).unsqueeze(1))).mean(dim=(0,1));active=compute_active_blocks_by_gate(model,vx,frac)
                model.train();item.update(validation_loss=float(vl),gate_mean=float(g.mean()),gate_std=float(g.std()),active_blocks=[int(v) for v in active.cpu().tolist()])
                if best is None or item['validation_loss']<best:best=item['validation_loss']
            metrics.append(item)
            with log.open('a') as h:h.write(json.dumps(item)+'\n')
            if step%a.checkpoint_interval==0 or tokens>=a.target_tokens:save()
    report={'backend':'torch','device':str(dev),'hardware_id':hardware,'effective_batch_tokens':a.batch_size*a.sequence_length,'compile_step':False,'compile_mode':None,'fused_optimizer':a.fused_optimizer,'architecture':'block_gated_bdh_direct_split_v_derivative' if a.value_path == 'direct_split_v' else 'block_gated_bdh_derivative','exact_bdh':False,'claim_eligible':False,'value_path':a.value_path,'dtype':a.dtype,'parameter_count':sum(p.numel() for p in model.parameters()),'block_size':a.block_size,'curriculum_stages':stages,'steps':step,'tokens_seen':tokens,'target_tokens':a.target_tokens,'budget_complete':tokens>=a.target_tokens,'best_validation_loss':best,'metrics':metrics,'checkpoint':str(checkpoint),'training_seconds':time.perf_counter()-started,'tokens_per_second':tokens/max(time.perf_counter()-started,1e-9),'peak_memory_bytes':peak(dev),'initialization_seed':a.seed,'final_parameter_sha256':fingerprint(model)}
    report.update(sampler.stop(tokens=tokens));(a.run_dir/'block_gated_training.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
