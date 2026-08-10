"""Explicit serializable BDH persistent-state checkpoints."""
from __future__ import annotations
from pathlib import Path
import torch
def save_state_checkpoint(path,states,position:int,*,model_fingerprint=None):
 packed=[]
 for s in states:
  if hasattr(s,'values') and hasattr(s,'scale'): packed.append({'quantized':True,'values':s.values.cpu(),'scale':s.scale.cpu(),'shape':s.shape,'dtype':str(s.dtype)})
  else: packed.append({'quantized':False,'tensor':s.detach().cpu()})
 torch.save({'states':packed,'position':int(position),'model_fingerprint':model_fingerprint},Path(path))
def load_state_checkpoint(path,*,device=None,dtype=None):
 q=torch.load(Path(path),map_location='cpu',weights_only=False);from reference.hz0i_state_storage import QuantizedState
 states=[]
 for item in q['states']:
  if item.get('quantized'): states.append(QuantizedState(item['values'].to(device),item['scale'].to(device),tuple(item['shape']),dtype or torch.float32))
  else: states.append(item['tensor'].to(device=device,dtype=dtype or item['tensor'].dtype))
 return states,int(q['position']),q.get('model_fingerprint')
