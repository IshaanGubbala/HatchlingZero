"""Layerwise HZ capability composition for BDH experiments."""
import torch
from torch import nn
from dataclasses import replace
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
from reference.hz0i_optional_integrations import ConditionalAnchorAttention,SessionFastWeights,RoutedSwiGLU
class LayerwiseIntegratedBDH(nn.Module):
 def __init__(self,config:HZ0IBDHConfig):
  super().__init__();self.base=HZ0IBDH(replace(config,use_session_memory=False,use_conditional_attention=False,use_fast_weights=False,use_moe=False));D=config.n_embd;self.conditional=ConditionalAnchorAttention(D,config.n_head) if config.use_conditional_attention else None;self.fast=SessionFastWeights(D) if config.use_fast_weights else None;self.moe=RoutedSwiGLU(D,D*2,capacity_factor=config.moe_capacity_factor,routing=config.moe_routing,fallback_threshold=config.moe_fallback_threshold,balanced_init=config.moe_balanced_init,router_noise=config.moe_router_noise) if config.use_moe else None
 def forward(self,idx,*,triggers=None):
  if self.conditional is not None and triggers is None: raise ValueError('triggers required')
  def hook(x,level):
   h=x[:,0]
   if self.conditional is not None:h=h+self.conditional(h,triggers)
   if self.fast is not None:h=h+self.fast.apply(h)
   if self.moe is not None:h=h+self.moe(h)[0]
   return h.unsqueeze(1)
  h=self.base.forward_hidden(idx,layer_hook=hook);return h@self.base.lm_head
