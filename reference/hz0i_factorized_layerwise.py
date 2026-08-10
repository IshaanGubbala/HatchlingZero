"""Efficient factorized+tied BDH with layerwise capability hooks."""
import torch
from dataclasses import replace
import torch.nn.functional as F
from torch import nn
from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_optional_integrations import ConditionalAnchorAttention,SessionFastWeights,RoutedSwiGLU,LearnedTriggerGate
class FactorizedLayerwiseTiedBDH(FactorizedTiedBDH):
 def __init__(self,config:HZ0IBDHConfig,rank=256,layer_stride=1):
  super().__init__(replace(config,use_session_memory=False,use_conditional_attention=False,use_fast_weights=False,use_moe=False),rank);self.config=config;self.capabilities_enabled=True;self.layer_stride=max(1,layer_stride);D=config.n_embd;self.conditional=ConditionalAnchorAttention(D,config.n_head) if config.use_conditional_attention else None;self.trigger_gate=LearnedTriggerGate(D,config.trigger_threshold,config.trigger_mode,config.trigger_fraction) if config.learned_triggers else None;self.fast=SessionFastWeights(D) if config.use_fast_weights else None;self.moe=RoutedSwiGLU(D,D*2,capacity_factor=config.moe_capacity_factor,routing=config.moe_routing,fallback_threshold=config.moe_fallback_threshold,balanced_init=config.moe_balanced_init,router_noise=config.moe_router_noise) if config.use_moe else None;self.conditional_gate=torch.nn.Parameter(torch.tensor(0.)) if self.conditional is not None else None;self.fast_gate=torch.nn.Parameter(torch.tensor(0.)) if self.fast is not None else None;self.moe_gate=torch.nn.Parameter(torch.tensor(0.)) if self.moe is not None else None
 def forward_hidden(self,idx,layer_hook=None):
  C=self.config;B,T=idx.shape;x=self.ln(self.embed(idx).unsqueeze(1))
  for level in range(C.n_layer):
   xs=F.relu(self._enc(x,self.enc_l,self.enc_r));ykv=self.ln(self.attn(Q=xs,K=xs,V=x));ys=F.relu(self._enc(ykv,self.val_l,self.val_r));x=self.ln(x+self.ln(self._dec(self.drop(xs*ys))))
   if layer_hook is not None and level % self.layer_stride == 0:x=layer_hook(x,level)
  return x.view(B,T,C.n_embd)
 def forward(self,idx,*,triggers=None,targets=None,trace=None):
  if self.conditional is not None and triggers is None and self.trigger_gate is None:raise ValueError('triggers required')
  def hook(x,level):
   if not self.capabilities_enabled:return x
   h=x[:,0]
   if self.conditional is not None:
    local_triggers=self.trigger_gate(h)[1] if triggers is None else triggers
    h=h+torch.tanh(self.conditional_gate)*self.conditional(h,local_triggers)
   if self.fast is not None:h=h+torch.tanh(self.fast_gate)*self.fast.apply_masked(h,triggers)
   if self.moe is not None:h=h+torch.tanh(self.moe_gate)*self.moe(h)[0]
   if trace is not None:
    rec={'layer':level,'hidden_rms':float(h.detach().float().pow(2).mean().sqrt()),'conditional_gate':float(torch.tanh(self.conditional_gate).detach()) if self.conditional_gate is not None else 0.0,'fast_gate':float(torch.tanh(self.fast_gate).detach()) if self.fast_gate is not None else 0.0,'moe_gate':float(torch.tanh(self.moe_gate).detach()) if self.moe_gate is not None else 0.0}
    if self.trigger_gate is not None: rec['trigger_rate']=float(self.trigger_gate.last_rate)
    if self.moe is not None: rec['expert_counts']=self.moe.last_counts.detach().cpu().tolist()
    trace.append(rec)
   return h.unsqueeze(1)
  h=self.forward_hidden(idx,hook if any((self.conditional,self.fast,self.moe)) and self.capabilities_enabled else None);w=F.normalize(self.embed.weight,dim=1);logits=torch.exp(self.logit_scale)*F.linear(F.normalize(h,dim=-1),w);loss=None
  if targets is not None:
   loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
   if self.moe is not None and self.config.moe_aux_weight:loss=loss+self.config.moe_aux_weight*self.moe.last_balance_loss
   if self.moe is not None and self.config.moe_z_weight:loss=loss+self.config.moe_z_weight*self.moe.last_z_loss
   if self.trigger_gate is not None and self.config.trigger_aux_weight:loss=loss+self.config.trigger_aux_weight*self.trigger_gate.last_sparsity_loss
  return logits,loss


 def stream(self,idx,chunk_sizes,triggers=None,state_storage="full",adapt_fast=False,include_moe=True,state_quantize_every=1):
  from reference.hz0i_factorized_bdh import init_factorized_states,factorized_stream_chunk
  B,T=idx.shape
  if triggers is None and self.conditional is not None: raise ValueError('triggers required')
  from reference.hz0i_state_storage import QuantizedState,quantize_int8,quantize_int8_per_head,dequantize_int8
  states=init_factorized_states(self,B,device=idx.device,dtype=self.embed.weight.dtype);kv_cache=[None]*self.config.n_layer;parts=[];pos=0
  for chunk_i,n in enumerate(chunk_sizes):
   trig=None if triggers is None else triggers[:,pos:pos+n]
   def hook(x,level,trig=trig,include_moe=include_moe):
    if level % self.layer_stride != 0:return x
    h=x[:,0]
    if self.conditional is not None:
     local_trig=self.trigger_gate(h)[1] if triggers is None and self.trigger_gate is not None else trig
     att,kv_cache[level]=self.conditional.forward_cached(h,local_trig,kv_cache[level],max_cache_len=T if kv_cache[level] is None else None);h=h+torch.tanh(self.conditional_gate)*att
    if self.fast is not None:
     h=h+torch.tanh(self.fast_gate)*self.fast.apply_masked(h,trig)
     if adapt_fast:self.fast.adapt(h,mask=trig)
    if include_moe and self.moe is not None:h=h+torch.tanh(self.moe_gate)*self.moe(h)[0]
    return h.unsqueeze(1)
   if state_storage in ('int8','int8_head') and states and isinstance(states[0],QuantizedState): states=[dequantize_int8(q).to(idx.device) for q in states]
   states,logits=factorized_stream_chunk(self,states,idx[:,pos:pos+n],pos,hook if any((self.conditional,self.fast,self.moe)) and self.capabilities_enabled else None);parts.append(logits);pos+=n
   if state_storage=='int8' and chunk_i%state_quantize_every==0: states=[quantize_int8(st) for st in states]
   elif state_storage=='int8_head' and chunk_i%state_quantize_every==0: states=[quantize_int8_per_head(st) for st in states]
   elif state_storage not in ('full','int8','int8_head'): raise ValueError('state_storage must be full, int8, or int8_head')
  return states,torch.cat(parts,dim=1)
