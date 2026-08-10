"""Full-capacity untied factorized BDH with layerwise capabilities."""
import torch
import torch.nn.functional as F
from reference.hz0i_factorized_layerwise import FactorizedLayerwiseTiedBDH
class FactorizedLayerwiseBDH(FactorizedLayerwiseTiedBDH):
 def __init__(self,config,rank=704,layer_stride=2):
  super().__init__(config,rank,layer_stride);self.lm_head=torch.nn.Parameter(torch.zeros(config.n_embd,config.vocab_size).normal_(std=.02))
 def forward(self,idx,*,triggers=None,targets=None,trace=None,return_hidden=False):
  if self.conditional is not None and triggers is None and self.trigger_gate is None:raise ValueError('triggers required')
  def hook(x,level):
   if not self.capabilities_enabled:return x
   h=x[:,0]
   if self.conditional is not None:
    local_triggers=self.trigger_gate(h)[1] if triggers is None else triggers
    h=h+torch.tanh(self.conditional_gate)*self.conditional(h,local_triggers)
   if self.fast is not None:
    h=h+torch.tanh(self.fast_gate)*self.fast.apply_masked(h,triggers)
    self.fast.adapt(h,mask=triggers)
   if self.moe is not None:h=h+torch.tanh(self.moe_gate)*self.moe(h)[0]
   if trace is not None:
    rec={'layer':level,'hidden_rms':float(h.detach().float().pow(2).mean().sqrt()),'conditional_gate':float(torch.tanh(self.conditional_gate).detach()) if self.conditional_gate is not None else 0.0,'fast_gate':float(torch.tanh(self.fast_gate).detach()) if self.fast_gate is not None else 0.0,'moe_gate':float(torch.tanh(self.moe_gate).detach()) if self.moe_gate is not None else 0.0}
    if self.trigger_gate is not None: rec['trigger_rate']=float(self.trigger_gate.last_rate)
    if self.moe is not None: rec['expert_counts']=self.moe.last_counts.detach().cpu().tolist()
    trace.append(rec)
   return h.unsqueeze(1)
  h=self.forward_hidden(idx,hook if any((self.conditional,self.fast,self.moe)) and self.capabilities_enabled else None)
  if return_hidden: return h,None
  logits=h@self.lm_head;loss=None
  if targets is not None:
   loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
   if self.moe is not None and self.config.moe_aux_weight:loss=loss+self.config.moe_aux_weight*self.moe.last_balance_loss
   if self.moe is not None and self.config.moe_z_weight:loss=loss+self.config.moe_z_weight*self.moe.last_z_loss
   if self.trigger_gate is not None and self.config.trigger_aux_weight:loss=loss+self.config.trigger_aux_weight*self.trigger_gate.last_sparsity_loss
  return logits,loss
