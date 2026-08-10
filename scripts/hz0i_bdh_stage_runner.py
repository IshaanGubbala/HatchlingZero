"""Experimental HZ-0I BDH runner using the audited JSONL token pipeline.

This is intentionally separate from the canonical MLX HZ-0A runner while I2
proves data order, checkpoint/resume, and dashboard-compatible metrics.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig

def read_rows(path, vocab, seq_len, batch_size, cursor=0):
 rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; out=[]
 for i in range(batch_size):
  row=rows[(cursor+i)%len(rows)]; vals=[int(x)%vocab for x in row[:seq_len]]
  if len(vals)<seq_len: vals += [0]*(seq_len-len(vals))
  out.append(vals)
 return torch.tensor(out), (cursor+batch_size)%len(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--resume',action='store_true');p.add_argument('--seq-len',type=int,default=32);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--vocab-size',type=int,default=256);p.add_argument('--seed',type=int,default=17);a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=a.vocab_size,dropout=0.0);m=HZ0IBDH(c);o=torch.optim.AdamW(m.parameters(),lr=1e-3); step=cursor=0; metrics=[]; ck=a.run_dir/'checkpoint.pt';
 if a.resume and ck.exists():
  q=torch.load(ck,weights_only=False);m.load_state_dict(q['model']);o.load_state_dict(q['optimizer']);step=q['step'];cursor=q['cursor'];metrics=q['metrics'];
 rows=a.data; start=time.perf_counter(); log=a.run_dir/'native_metal_memory.jsonl'
 for _ in range(step,a.steps):
  b,cursor=read_rows(rows,a.vocab_size,a.seq_len,a.batch_size,cursor);_,loss=m(b[:,:-1].contiguous(),targets=b[:,1:].contiguous());o.zero_grad();loss.backward();o.step();step+=1; item={'step':step,'tokens_seen':step*a.batch_size*(a.seq_len-1),'loss':float(loss.detach()),'wall_time':time.perf_counter()-start};metrics.append(item);log.open('a').write(json.dumps(item)+'\n')
  if step%10==0 or step==a.steps: torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'step':step,'cursor':cursor,'metrics':metrics},ck)
 (a.run_dir/'native_metal.json').write_text(json.dumps({'architecture':'hz0i_bdh','steps':step,'tokens_seen':step*a.batch_size*(a.seq_len-1),'parameter_count':sum(x.numel() for x in m.parameters()),'checkpoint':str(ck),'metrics':metrics},indent=2))
if __name__=='__main__':main()
