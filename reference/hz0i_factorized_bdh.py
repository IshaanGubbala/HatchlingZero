"""Experimental low-rank/factorized BDH projection path."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
from reference.hz0i_sparse_bdh import topk_relu
class FactorizedBDH(HZ0IBDH):
 def __init__(self,config,rank=256,latent_topk_ratio=None):
  super().__init__(config);self.rank=rank;self.latent_topk_ratio=latent_topk_ratio;H,D=config.n_head,config.n_embd;N=D*config.mlp_internal_dim_multiplier//H
  del self.encoder,self.encoder_v,self.decoder
  self.enc_l=torch.nn.Parameter(torch.randn(H,D,rank)*.02);self.enc_r=torch.nn.Parameter(torch.randn(H,rank,N)*(.02/(rank**.5)))
  self.val_l=torch.nn.Parameter(torch.randn(H,D,rank)*.02);self.val_r=torch.nn.Parameter(torch.randn(H,rank,N)*(.02/(rank**.5)))
  self.dec_l=torch.nn.Parameter(torch.randn(H,N,rank)*(.02/(N**.5)));self.dec_r=torch.nn.Parameter(torch.randn(H,rank,D)*.02)
 def _sparse(self,x): return topk_relu(x,self.latent_topk_ratio) if self.latent_topk_ratio is not None else torch.relu(x)
 def _enc(self,x,left,right):
  H=self.config.n_head
  if x.shape[1]==1:x=x.expand(-1,H,-1,-1)
  # Broadcasted matmul keeps both factor stages on regular batched-GEMM
  # paths instead of relying on einsum's backend-specific contraction choice.
  z=torch.matmul(x,left.unsqueeze(0));return torch.matmul(z,right.unsqueeze(0))
 def _dec(self,x):
  z=torch.matmul(x,self.dec_l.unsqueeze(0));z=torch.matmul(z,self.dec_r.unsqueeze(0));return z.sum(dim=1,keepdim=True)
 @classmethod
 def from_dense(cls,config,rank,source=None):
  dense=source if source is not None else HZ0IBDH(config);m=cls(config,rank)
  with torch.no_grad():
   m.embed.weight.copy_(dense.embed.weight);m.lm_head.copy_(dense.lm_head)
   def fac(w,r):
    u,sv,vh=torch.linalg.svd(w.float(),full_matrices=False);u=u[...,:r];sv=sv[...,:r].clamp_min(1e-8).sqrt();return (u*sv.unsqueeze(-2)).to(w.dtype),(sv.unsqueeze(-1)*vh[...,:r,:]).to(w.dtype)
   for target,weight in [(('enc_l','enc_r'),dense.encoder),(('val_l','val_r'),dense.encoder_v)]:
    left,right=fac(weight,rank);getattr(m,target[0]).copy_(left);getattr(m,target[1]).copy_(right)
   left,right=fac(dense.decoder.reshape(config.n_head,-1,config.n_embd),rank);m.dec_l.copy_(left);m.dec_r.copy_(right)
  return m
 def forward_hidden(self,idx):
  C=self.config;B,T=idx.shape;x=self.ln(self.embed(idx).unsqueeze(1))
  for _ in range(C.n_layer):
   xs=self._sparse(self._enc(x,self.enc_l,self.enc_r));ykv=self.ln(self.attn(Q=xs,K=xs,V=x));ys=self._sparse(self._enc(ykv,self.val_l,self.val_r));x=self.ln(x+self.ln(self._dec(self.drop(xs*ys))))
  return x.view(B,T,C.n_embd)
 def forward(self,idx,targets=None):
  h=self.forward_hidden(idx);logits=h@self.lm_head;loss=None
  if targets is not None:loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
  return logits,loss


def factorized_variable_depth_forward(model, idx, n_iterations: int, targets=None):
 """Exact FactorizedBDH forward with a training-controlled recurrent depth.

 ``n_iterations == model.config.n_layer`` follows ``FactorizedBDH.forward``
 exactly; smaller positive values only shorten the shared recurrent loop, the
 same training-only mechanism used by canonical dense BDH curriculum runs.
 """
 if n_iterations < 1:
  raise ValueError("n_iterations must be positive")
 C=model.config;B,T=idx.shape;x=model.ln(model.embed(idx).unsqueeze(1))
 for _ in range(n_iterations):
  xs=model._sparse(model._enc(x,model.enc_l,model.enc_r));ykv=model.ln(model.attn(Q=xs,K=xs,V=x));ys=model._sparse(model._enc(ykv,model.val_l,model.val_r));x=model.ln(x+model.ln(model._dec(model.drop(xs*ys))))
 h=x.view(B,T,C.n_embd);logits=h@model.lm_head;loss=None
 if targets is not None:loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
 return logits,loss

def init_factorized_states(model,batch_size,device=None,dtype=None):
 H=model.config.n_head;D=model.config.n_embd;N=D*model.config.mlp_internal_dim_multiplier//H;device=device or model.embed.weight.device;dtype=dtype or model.embed.weight.dtype
 return [torch.zeros(batch_size,H,N,D,device=device,dtype=dtype) for _ in range(model.config.n_layer)]
def factorized_stream_chunk(model,states,idx_chunk,start_position,layer_hook=None):
 c=model.config;B,L=idx_chunk.shape;H=c.n_head;D=c.n_embd;N=D*c.mlp_internal_dim_multiplier//H;dev=idx_chunk.device;x=model.ln(model.embed(idx_chunk).unsqueeze(1));ph=(torch.arange(start_position,start_position+L,device=dev,dtype=model.attn.freqs.dtype).view(1,1,L,1)*model.attn.freqs);new=[]
 for level in range(c.n_layer):
  V=x.expand(-1,H,-1,-1);xs=model._sparse(model._enc(x,model.enc_l,model.enc_r));qr=model.attn.rope(ph,xs);intra=(qr@qr.mT).tril(diagonal=-1)@V;cross=qr@states[level];ykv=model.ln(intra+cross);ys=model._sparse(model._enc(ykv,model.val_l,model.val_r));x=model.ln(x+model.ln(model._dec(model.drop(xs*ys))))
  if layer_hook is not None:x=layer_hook(x,level)
  new.append(states[level]+qr.mT@V)
 h=x.view(B,L,D)
 if hasattr(model,'lm_head'):logits=h@model.lm_head
 else:logits=torch.exp(model.logit_scale)*F.linear(F.normalize(h,dim=-1),F.normalize(model.embed.weight,dim=1))
 return new,logits
def factorized_stream_sequence(model,idx,chunk_sizes,states=None,state_storage="full",retention=1.0):
 from reference.hz0i_state_storage import quantize_int8,quantize_int8_per_head,dequantize_int8,QuantizedState
 B,T=idx.shape
 if states is None:states=init_factorized_states(model,B,device=idx.device,dtype=model.embed.weight.dtype)
 parts=[];pos=0
 for n in chunk_sizes:
  if state_storage in ('int8','int8_head') and states and isinstance(states[0],QuantizedState):states=[dequantize_int8(q).to(idx.device) for q in states]
  if retention<1.0:states=[st*retention for st in states]
  states,logits=factorized_stream_chunk(model,states,idx[:,pos:pos+n],pos);parts.append(logits);pos+=n
  if state_storage=='int8':states=[quantize_int8(st) for st in states]
  elif state_storage=='int8_head':states=[quantize_int8_per_head(st) for st in states]
  elif state_storage!='full':raise ValueError('state_storage must be full, int8, or int8_head')
 return states,torch.cat(parts,dim=1)
