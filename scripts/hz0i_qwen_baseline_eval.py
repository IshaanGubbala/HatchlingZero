"""Evaluate Qwen3-0.6B on the same raw corpus slice used for HZ0I planning."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
def main():
 p=argparse.ArgumentParser();p.add_argument('--text',type=Path,required=True);p.add_argument('--model',default='Qwen/Qwen3-0.6B');p.add_argument('--sequences',type=int,default=32);p.add_argument('--seq-len',type=int,default=256);p.add_argument('--out',type=Path,required=True);a=p.parse_args();tok=AutoTokenizer.from_pretrained(a.model);m=AutoModelForCausalLM.from_pretrained(a.model);m.eval();text=a.text.read_text(errors='ignore');ids=tok(text,return_tensors='pt',add_special_tokens=False).input_ids[0];n=min(a.sequences,(len(ids)-1)//a.seq_len);losses=[];t=time.perf_counter();
 with torch.no_grad():
  for i in range(n):
   x=ids[i*a.seq_len:(i+1)*a.seq_len].unsqueeze(0); y=m(x,labels=x).loss;losses.append(float(y))
 out={'model':a.model,'parameters':sum(p.numel() for p in m.parameters()),'sequences':n,'seq_len':a.seq_len,'mean_loss':sum(losses)/len(losses),'ppl':float(torch.exp(torch.tensor(losses).mean())),'seconds':time.perf_counter()-t,'source':str(a.text),'note':'pretrained baseline; not yet a matched-training comparison'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
