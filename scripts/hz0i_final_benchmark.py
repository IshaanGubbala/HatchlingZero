"""I-plan final scoped audit benchmark: quality, speed, state, FLOP estimates, graph."""
import argparse,json,time,resource
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0h_bdh_graph import extract_effective_graph,graph_stats
from reference.hz0a_torch_model import HZ0AConfig,HZ0AModel
from scripts.hz0i_realdata_architecture_comparison import batches
def bdh_flops(B,T,D,H,N,L):
 return int(L*(2*B*T*H*D*N*3 + 2*B*T*T*N + 2*B*T*T*N*D + 2*B*T*N*H*D*N))
def gdn_flops(B,T,D,H,K,L): return int(L*(2*B*T*D*(4*H*K+2*H*K)+2*B*T*H*K*K))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--validation-data',type=Path,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--seed',type=int,default=12);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(a.seed);bs=batches(a.data,24576,n=a.steps);val=batches(a.validation_data,24576,n=10)
 models={'bdh':BDH(BDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=24576,dropout=0.0)),'gdn2_fix':HZ0AModel(HZ0AConfig(24576,96,16,4,24,24,2048,tuple(range(0,16,2)),'gdn2_fix',False))};out={'steps':a.steps,'seed':a.seed,'results':{}}
 for name,m in models.items():
  opt=torch.optim.AdamW(m.parameters(),lr=1e-3); losses=[];t=time.perf_counter()
  for b in bs:
   x,y=b[:,:-1].contiguous(),b[:,1:].contiguous(); logits,_=m(x,targets=y) if name=='bdh' else (m(x)[0],None); l=torch.nn.functional.cross_entropy(logits.reshape(-1,24576),y.reshape(-1));opt.zero_grad();l.backward();opt.step();losses.append(float(l.detach()))
  m.eval();vl=[]
  with torch.no_grad():
   for b in val:
    x,y=b[:,:-1].contiguous(),b[:,1:].contiguous(); logits,_=m(x,targets=y) if name=='bdh' else (m(x)[0],None);vl.append(float(torch.nn.functional.cross_entropy(logits.reshape(-1,24576),y.reshape(-1))))
  elapsed=time.perf_counter()-t; res={'params':sum(x.numel() for x in m.parameters()),'loss_first':losses[0],'loss_last':losses[-1],'validation_loss':sum(vl)/len(vl),'tok_s':len(bs)*4*31/elapsed,'finite':all(torch.isfinite(x).all().item() for x in m.parameters()),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
  if name=='bdh':
   N=384*96//4;res.update({'state_bytes':2*4*N*96*4,'estimated_forward_flops_per_batch':bdh_flops(4,31,96,4,N,2),'graph_head0':graph_stats(extract_effective_graph(m,0)[:256,:256]).__dict__})
  else:res.update({'state_bytes_per_recurrent_layer':4*4*24*24*4,'estimated_forward_flops_per_batch':gdn_flops(4,31,96,4,24,16)})
  out['results'][name]=res
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
