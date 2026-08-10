"""Grouped-head factorization for compact BDH capacity sharing."""
import torch
from reference.hz0i_factorized_bdh import FactorizedBDH
class GroupedFactorizedBDH(FactorizedBDH):
 def __init__(self,config,rank=256,head_groups=4):
  super().__init__(config,rank);H=config.n_head
  if H%head_groups: raise ValueError('n_head must divide head_groups')
  self.head_groups=head_groups;self.register_buffer('head_group_index',torch.arange(H)%head_groups)
  old_e=self.enc_l.detach();old_v=self.val_l.detach();del self.enc_l,self.val_l
  self.enc_l=torch.nn.Parameter(old_e[:head_groups].clone());self.val_l=torch.nn.Parameter(old_v[:head_groups].clone())
 def _enc(self,x,left,right):
  H=self.config.n_head
  if x.shape[1]==1:x=x.expand(-1,H,-1,-1)
  left=left[self.head_group_index];z=torch.einsum('bhtd,hdr->bhtr',x,left);return torch.einsum('bhtr,hrn->bhtn',z,right)
