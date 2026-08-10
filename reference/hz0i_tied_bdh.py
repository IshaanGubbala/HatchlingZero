"""Weight-tied BDH experimental variant: one vocabulary matrix for input/output."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
class TiedBDH(HZ0IBDH):
 def __init__(self,config,logit_scale=10.0):super().__init__(config);del self.lm_head;self.logit_scale=torch.nn.Parameter(torch.tensor(float(logit_scale)).log())
 def forward(self,idx,targets=None):
  h=self.forward_hidden(idx);h=F.normalize(h,dim=-1);w=F.normalize(self.embed.weight,dim=1);logits=torch.exp(self.logit_scale)*F.linear(h,w);loss=None
  if targets is not None:loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
  return logits,loss
