"""Optional sampled-vocabulary objective for faster BDH pretraining."""
from __future__ import annotations
import torch
import torch.nn.functional as F
def sampled_softmax_loss(hidden,lm_head,targets,num_negatives=1024,generator=None):
 B,T,D=hidden.shape;V=lm_head.shape[1];flat_h=hidden.reshape(-1,D);flat_t=targets.reshape(-1);n=flat_t.numel();
 neg=torch.randint(0,V,(n,num_negatives),device=hidden.device,generator=generator);neg=neg.masked_fill(neg.eq(flat_t[:,None]),0);classes=torch.cat([flat_t[:,None],neg],dim=1);w=lm_head[:,classes.reshape(-1)].transpose(0,1).reshape(n,num_negatives+1,D);scores=torch.bmm(w,flat_h[:,:,None]).squeeze(-1);return F.cross_entropy(scores,torch.zeros(n,dtype=torch.long,device=hidden.device))
def bdhi_sampled_loss(model,idx,targets,num_negatives=1024):
 hidden=model.forward_hidden(idx);return sampled_softmax_loss(hidden,model.lm_head,targets,num_negatives)
