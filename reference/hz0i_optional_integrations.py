"""Torch-side I4/I5 integration primitives for the experimental BDH shell.

These are portable correctness gates; promotion still requires matched quality
runs on the actual BDH backbone.
"""
from __future__ import annotations
import torch
from torch import nn

class ConditionalAnchorAttention(nn.Module):
    def __init__(self, dim:int, heads:int=4):
        super().__init__(); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
    def forward(self,x:torch.Tensor,triggers:torch.Tensor):
        # Vectorized triggered-query attention: Q is computed only for trigger
        # positions, while K/V are shared across the sequence.
        import torch.nn.functional as F
        B,T,D=x.shape; H=self.attn.num_heads; dh=D//H; mask=triggers.to(torch.bool)
        bi,ti=torch.nonzero(mask,as_tuple=True); out=torch.zeros_like(x)
        if bi.numel()==0: return out
        W=self.attn.in_proj_weight;bias=self.attn.in_proj_bias
        q=F.linear(x[bi,ti],W[:D],None if bias is None else bias[:D]).view(-1,H,dh)
        k_all=F.linear(x,W[D:2*D],None if bias is None else bias[D:2*D]);v_all=F.linear(x,W[2*D:],None if bias is None else bias[2*D:])
        k_all=k_all.view(B,T,H,dh).transpose(1,2); v_all=v_all.view(B,T,H,dh).transpose(1,2)
        # Gather K/V per triggered query and apply a prefix mask.
        kg=k_all[bi]; vg=v_all[bi]; scores=(q.unsqueeze(2)*kg).sum(-1)/dh**0.5
        positions=torch.arange(T,device=x.device).view(1,-1); allowed=positions<=ti.view(-1,1)
        scores=scores.masked_fill(~allowed.unsqueeze(1),torch.finfo(scores.dtype).min); weights=scores.softmax(-1)
        attended=(weights.unsqueeze(-1)*vg).sum(2).reshape(-1,D); attended=F.linear(attended,self.attn.out_proj.weight,self.attn.out_proj.bias)
        out[bi,ti]=attended-x[bi,ti]
        return out

    def forward_cached(self,x,triggers,cache=None,max_cache_len=None):
        """Causal triggered attention with an optional persistent K/V cache."""
        import torch.nn.functional as F
        B,T,D=x.shape;H=self.attn.num_heads;dh=D//H;mask=triggers.to(torch.bool);bi,ti=torch.nonzero(mask,as_tuple=True);out=torch.zeros_like(x)
        W=self.attn.in_proj_weight;bias=self.attn.in_proj_bias
        k=F.linear(x,W[D:2*D],None if bias is None else bias[D:2*D]).view(B,T,H,dh).transpose(1,2)
        v=F.linear(x,W[2*D:],None if bias is None else bias[2*D:]).view(B,T,H,dh).transpose(1,2)
        past=0
        if cache is not None:
            pk,pv=cache;past=pk.shape[2];k=torch.cat((pk,k),2);v=torch.cat((pv,v),2)
        new_cache=(k,v)
        if bi.numel()==0:return out,new_cache
        q=F.linear(x[bi,ti],W[:D],None if bias is None else bias[:D]).view(-1,H,dh);kg=k[bi];vg=v[bi];scores=(q.unsqueeze(2)*kg).sum(-1)/dh**0.5
        pos=torch.arange(k.shape[2],device=x.device).view(1,-1);allowed=pos<=past+ti.view(-1,1);scores=scores.masked_fill(~allowed.unsqueeze(1),torch.finfo(scores.dtype).min);weights=scores.softmax(-1);att=(weights.unsqueeze(-1)*vg).sum(2).reshape(-1,D);out[bi,ti]=F.linear(att,self.attn.out_proj.weight,self.attn.out_proj.bias)-x[bi,ti]
        return out,new_cache

class SessionFastWeights(nn.Module):
    def __init__(self,dim:int,rank:int=8,max_norm:float=1.0):
        super().__init__(); self.dim=dim; self.rank=rank; self.max_norm=max_norm
        self.a=nn.Parameter(torch.randn(dim,rank)*.02); self.b=nn.Parameter(torch.zeros(rank,dim))
    def delta(self):
        # b.detach().clone(): b is Hebbian-updated in-place by adapt() (never
        # gradient-trained), and this module's single shared instance is
        # called once per layer within one forward pass. Reading self.b
        # directly and later calling adapt() in the same pass mutates the
        # exact tensor PyTorch saved for this read's backward, raising
        # "modified by an inplace operation". clone() snapshots the value so
        # later in-place writes to the real b can't invalidate it; detach()
        # keeps b (rightly) out of the autograd graph since only a and the
        # gate are meant to receive gradients.
        b=self.b.detach().clone()
        d=self.a@b; n=d.float().norm().clamp_min(1e-8); return d*min(1.0,self.max_norm/max(float(n.detach()),1e-6))
    def apply(self,x): return x @ self.delta().T
    @torch.no_grad()
    def adapt(self,x,lr=1e-3,mask=None):
        chosen=x if mask is None else x[mask]
        if chosen.numel()==0:return
        chosen=chosen.reshape(-1,self.dim);z=chosen@self.a;update=(z.transpose(-1,-2)@chosen)/max(1,chosen.shape[0]);self.b.add_(lr*update);n=self.b.norm().clamp_min(1e-8);self.b.mul_(min(1.0,float(self.max_norm/n)))
    def apply_masked(self,x,mask):
        if mask is None:return self.apply(x)
        out=torch.zeros_like(x);chosen=x[mask]
        if chosen.numel():out[mask]=self.apply(chosen)
        return out
    def reset(self):
        with torch.no_grad(): self.a.normal_(0,.02); self.b.zero_()

class RoutedSwiGLU(nn.Module):
    def __init__(self,dim:int,hidden:int,experts:int=4,capacity_factor:float|None=None,routing:str="top1",fallback_threshold:float=0.0,balanced_init:bool=False,router_noise:float=0.0):
        if routing not in ("top1","top2","adaptive","balanced"): raise ValueError("routing must be top1, top2, or adaptive")
        super().__init__(); self.router=nn.Linear(dim,experts);
        if balanced_init: nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Linear(hidden,dim)) for _ in range(experts)]);self.capacity_factor=capacity_factor;self.router_noise=router_noise;self.routing=routing;self.fallback_threshold=fallback_threshold;self.last_dropped=0;self.last_fallback=0;self.last_balance_loss=torch.tensor(0.);self.last_z_loss=torch.tensor(0.);self.last_counts=torch.zeros(experts,dtype=torch.long);self.last_choice=torch.empty(0,dtype=torch.long)
    def forward(self,x):
        shape=x.shape;flat=x.reshape(-1,shape[-1]);router_logits=self.router(flat)
        if self.training and self.router_noise>0: router_logits=router_logits+torch.randn_like(router_logits)*self.router_noise
        self.last_z_loss=torch.logsumexp(router_logits,dim=-1).square().mean();probs=torch.softmax(router_logits,-1);importance=probs.mean(0);load=torch.nn.functional.one_hot(probs.argmax(-1),num_classes=len(self.experts)).float().mean(0);self.last_balance_loss=len(self.experts)*(importance*load).sum();top=probs.topk(min(2,len(self.experts)),-1);tokens=flat.shape[0];choice=top.indices[:,0]
        if self.routing=="balanced":
            # Vectorized quota repair: retain each expert's strongest initial
            # tokens, then fill remaining expert slots from unassigned tokens.
            cap=(tokens+len(self.experts)-1)//len(self.experts);choice=torch.full((tokens,),-1,dtype=torch.long,device=x.device);assigned=torch.zeros(tokens,dtype=torch.bool,device=x.device);used=torch.zeros(len(self.experts),dtype=torch.long,device=x.device)
            for e in range(len(self.experts)):
                ids=(top.indices[:,0]==e).nonzero(as_tuple=False).flatten();keep=min(cap,ids.numel())
                if keep:
                    ids=ids[torch.topk(router_logits[ids,e],keep).indices];choice[ids]=e;assigned[ids]=True;used[e]=keep
            for e in range(len(self.experts)):
                ids=(~assigned).nonzero(as_tuple=False).flatten();slots=int(cap-used[e])
                if ids.numel()==0 or slots<=0: continue
                take=min(slots,ids.numel());ids=ids[torch.topk(router_logits[ids,e],take).indices];choice[ids]=e;assigned[ids]=True;used[e]+=take
            choice[choice<0]=probs[choice<0].argmax(-1)
        self.last_choice=choice.detach();self.last_counts=torch.bincount(choice,minlength=len(self.experts));out=torch.zeros_like(flat);self.last_dropped=0;self.last_fallback=0;capacity=None if self.capacity_factor is None else max(1,int(torch.ceil(torch.tensor(self.capacity_factor*tokens/len(self.experts))).item()));assigned=torch.zeros(tokens,dtype=torch.bool,device=x.device)
        for i,e in enumerate(self.experts):
            mask=choice==i
            if capacity is not None and mask.sum()>capacity:
                ids=mask.nonzero(as_tuple=False).squeeze(-1);keep=probs[ids,i].topk(capacity).indices;limited=torch.zeros_like(mask);limited[ids[keep]]=True;mask=limited
            if mask.any(): out[mask]=e(flat[mask])*probs[mask,i].unsqueeze(-1);assigned|=mask
        if self.routing in ("top2","adaptive") and capacity is not None:
            dropped=~assigned;second=top.indices[:,1]
            if self.routing=="adaptive": dropped=dropped & (probs.gather(1,second[:,None]).squeeze(1)>=self.fallback_threshold);
            for i,e in enumerate(self.experts):
                mask=dropped & (second==i);available=capacity-int((choice==i).logical_and(assigned).sum());
                if available>0 and mask.sum()>available:
                    ids=mask.nonzero(as_tuple=False).squeeze(-1);keep=probs[ids,i].topk(available).indices;limited=torch.zeros_like(mask);limited[ids[keep]]=True;mask=limited
                if mask.any(): out[mask]=e(flat[mask])*probs[mask,i].unsqueeze(-1);assigned|=mask;self.last_fallback+=int(mask.sum())
        self.last_dropped+=int((~assigned).sum());return out.reshape(shape),choice.reshape(shape[:-1])



class LearnedTriggerGate(nn.Module):
    """Predicts sparse conditional-attention trigger positions from hidden states."""
    def __init__(self,dim:int,threshold:float=.5,mode:str="threshold",fraction:float=.0625):
        super().__init__();self.proj=nn.Linear(dim,1);self.threshold=threshold;self.mode=mode;self.fraction=fraction;self.last_rate=0.0;self.last_sparsity_loss=torch.tensor(0.)
        if mode not in ("threshold","topk"): raise ValueError("mode must be threshold or topk")
    def forward(self,hidden):
        scores=torch.sigmoid(self.proj(hidden).squeeze(-1))
        if self.mode=="topk":
            k=max(1,int(scores.shape[-1]*self.fraction));idx=scores.topk(k,dim=-1).indices;mask=torch.zeros_like(scores,dtype=torch.bool).scatter(-1,idx,True)
        else: mask=scores>=self.threshold
        self.last_rate=float(mask.float().mean().detach());self.last_sparsity_loss=scores.mean();return scores,mask
