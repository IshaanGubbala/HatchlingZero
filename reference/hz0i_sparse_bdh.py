"""Experimental top-k sparse latent BDH variant."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from reference.hz0h_bdh_torch import BDH
def topk_relu(x,ratio,n_head=None):
 relu=F.relu(x)
 if n_head is not None:
  n=relu.shape[-1]//n_head; z=relu.reshape(*relu.shape[:-1],n_head,n); k=max(1,int(n*ratio)); vals,idx=torch.topk(z,k,dim=-1); out=torch.zeros_like(z).scatter(-1,idx,vals); return out.reshape_as(relu)
 k=max(1,int(relu.shape[-1]*ratio)); vals,idx=torch.topk(relu,k,dim=-1); out=torch.zeros_like(relu);return out.scatter(-1,idx,vals)
class SparseBDH(BDH):
 def __init__(self,config,latent_topk_ratio=.25):super().__init__(config);self.latent_topk_ratio=latent_topk_ratio
 def forward(self,idx,targets=None):
  C=self.config;B,T=idx.shape;D=C.n_embd;nh=C.n_head;N=D*C.mlp_internal_dim_multiplier//nh;x=self.ln(self.embed(idx).unsqueeze(1))
  for _ in range(C.n_layer):
   xs=topk_relu(x@self._w(self.encoder),self.latent_topk_ratio,nh);ykv=self.ln(self.attn(Q=xs,K=xs,V=x));ys=topk_relu(ykv@self._w(self.encoder_v),self.latent_topk_ratio,nh);xy=self.drop(xs*ys);ymlp=xy.transpose(1,2).reshape(B,1,T,N*nh)@self._w(self.decoder);x=self.ln(x+self.ln(ymlp))
  logits=x.view(B,T,D)@self.lm_head;loss=None
  if targets is not None:loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
  return logits,loss
