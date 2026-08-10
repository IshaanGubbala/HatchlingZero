"""Combined low-rank and tied-vocabulary BDH path."""
import torch
import torch.nn.functional as F
from reference.hz0i_factorized_bdh import FactorizedBDH
class FactorizedTiedBDH(FactorizedBDH):
 @classmethod
 def from_dense(cls,config,rank,source=None):
  dense=source if source is not None else __import__("reference.hz0i_bdh_model",fromlist=["HZ0IBDH"]).HZ0IBDH(config)
  approx=FactorizedBDH.from_dense(config,rank,dense);m=cls(config,rank)
  with torch.no_grad():
   m.embed.weight.copy_(approx.embed.weight)
   for n in ("enc_l","enc_r","val_l","val_r","dec_l","dec_r"):
    getattr(m,n).copy_(getattr(approx,n))
   m.logit_scale.fill_(torch.tensor(10.).log())
  return m
 def __init__(self,config,rank=256,logit_scale=10.):
  super().__init__(config,rank);del self.lm_head;self.logit_scale=torch.nn.Parameter(torch.tensor(float(logit_scale)).log())
 def forward(self,idx,targets=None):
  h=F.normalize(self.forward_hidden(idx),dim=-1);w=F.normalize(self.embed.weight,dim=1);logits=torch.exp(self.logit_scale)*F.linear(h,w);loss=None
  if targets is not None:loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
  return logits,loss
