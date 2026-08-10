"""Experimental HZ-0I BDH-centered model shell.

This intentionally reuses the independently-tested BDH oracle rather than
copying its math. HZ-0I integration layers will be added behind explicit
flags after I1 parity gates; canonical HZ-0A remains unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import torch
from reference.hz0h_bdh_torch import BDH, BDHConfig

@dataclass
class HZ0IBDHConfig(BDHConfig):
    use_session_memory: bool = False
    use_conditional_attention: bool = False
    use_fast_weights: bool = False
    use_moe: bool = False
    moe_capacity_factor: float | None = None
    moe_routing: str = "top1"
    moe_fallback_threshold: float = 0.0
    moe_balanced_init: bool = False
    moe_router_noise: float = 0.0
    moe_aux_weight: float = 0.0
    moe_z_weight: float = 0.0
    learned_triggers: bool = False
    trigger_threshold: float = 0.5
    trigger_aux_weight: float = 0.0
    trigger_mode: str = "threshold"
    trigger_fraction: float = 0.0625

class HZ0IBDH(BDH):
    """BDH backbone with explicit, currently-disabled HZ integration flags.

    Flags are rejected until their I3-I5 matched revalidation exists; this
    prevents silently claiming that HZ memory/MoE semantics transfer to BDH.
    """
    def __init__(self, config: HZ0IBDHConfig):
        if any((config.use_session_memory, config.use_conditional_attention, config.use_fast_weights, config.use_moe)):
            raise NotImplementedError('HZ-0I integration flags require their matched revalidation gates')
        super().__init__(config)

    def forward_hidden(self, idx, layer_hook=None):
        import torch.nn.functional as F
        C=self.config; B,T=idx.size(); D=C.n_embd; nh=C.n_head; N=D*C.mlp_internal_dim_multiplier//nh
        x=self.ln(self.embed(idx).unsqueeze(1))
        for _level in range(C.n_layer):
            xs=F.relu(x @ self._w(self.encoder)); ykv=self.attn(Q=xs,K=xs,V=x); ykv=self.ln(ykv)
            ys=F.relu(ykv @ self._w(self.encoder_v)); xy=self.drop(xs*ys)
            ymlp=xy.transpose(1,2).reshape(B,1,T,N*nh) @ self._w(self.decoder)
            x=self.ln(x+self.ln(ymlp))
            if layer_hook is not None: x=layer_hook(x,_level)
        return x.view(B,T,D)

    def forward_stream(self, idx, chunk_sizes):
        """Run the exact persistent BDH outer-product state path."""
        from reference.hz0h_bdh_torch import bdh_stream_sequence
        states, logits = bdh_stream_sequence(self, idx, chunk_sizes)
        return logits, states


def parameter_count(config: HZ0IBDHConfig) -> int:
    return sum(p.numel() for p in HZ0IBDH(config).parameters())


def write_bdh_memory(memory_state, key, value, strength, *, step: int):
    """Explicit I3 write bridge; returns a new immutable MemoryState."""
    from reference.hz0b_memory_simulator_torch import write
    return write(memory_state, key, value, strength, step=step)


class HZ0IBDHMemory(torch.nn.Module):
    """Read-only HZ-0B memory adapter over the BDH residual representation.

    This is I3-only: memory is supplied as an immutable torch MemoryState and
    no write path is exposed. The adapter is deliberately not enabled on the
    base model until its matched quality gate is run.
    """
    def __init__(self, model: HZ0IBDH, memory_state, key_dim: int):
        import torch
        super().__init__(); self.model=model; self.memory_state=memory_state
        self.query= torch.nn.Linear(model.config.n_embd,key_dim)
        self.value_to_hidden=torch.nn.Linear(memory_state.values.shape[-1],model.config.n_embd)
        self.gate=torch.nn.Linear(model.config.n_embd,model.config.n_embd)
    def forward(self, idx):
        import torch
        from reference.hz0b_memory_simulator_torch import read
        hidden=self.model.forward_hidden(idx)
        B,T,D=hidden.shape
        query=self.query(hidden).reshape(B*T,-1)
        state=self.memory_state
        state=type(state)(*(getattr(state,k).repeat_interleave(T,dim=0) for k in state.__dataclass_fields__))
        readout,_=read(state,query)
        contribution=self.value_to_hidden(readout).reshape(B,T,D)
        mixed=hidden+torch.sigmoid(self.gate(hidden))*contribution
        return mixed @ self.model.lm_head


class HZ0IBDHIntegrated(torch.nn.Module):
    """Explicit opt-in I4/I5 composition shell for ablation experiments."""
    def __init__(self, config: HZ0IBDHConfig):
        super().__init__(); from dataclasses import replace; self.base=HZ0IBDH(replace(config,use_session_memory=False,use_conditional_attention=False,use_fast_weights=False,use_moe=False)); self.use_conditional_attention=config.use_conditional_attention; self.use_fast_weights=config.use_fast_weights; self.use_moe=config.use_moe
        from reference.hz0i_optional_integrations import ConditionalAnchorAttention,SessionFastWeights,RoutedSwiGLU
        self.conditional=ConditionalAnchorAttention(config.n_embd,config.n_head) if self.use_conditional_attention else None
        self.fast=SessionFastWeights(config.n_embd) if self.use_fast_weights else None
        self.moe=RoutedSwiGLU(config.n_embd,config.n_embd*2,balanced_init=config.moe_balanced_init,router_noise=config.moe_router_noise) if self.use_moe else None
    def forward(self, idx, *, triggers=None):
        hidden=self.base.forward_hidden(idx)
        if self.conditional is not None:
            if triggers is None: raise ValueError('triggers required when conditional attention is enabled')
            hidden=hidden+self.conditional(hidden,triggers)
        if self.fast is not None: hidden=hidden+self.fast.apply(hidden)
        if self.moe is not None: hidden=hidden+self.moe(hidden)[0]
        return hidden @ self.base.lm_head


class HZ0IEnhancedBDH(torch.nn.Module):
    """Experimental capability-first composition matching the HZ premise.

    Persistent BDH state remains in the backbone; optional explicit memory,
    triggered attention, low-rank fast weights, and routed sparse capacity are
    applied to the residual representation with diagnostics returned.
    """
    def __init__(self, config: HZ0IBDHConfig, *, memory_state=None):
        super().__init__(); self.base=HZ0IBDH(replace(config,use_session_memory=False,use_conditional_attention=False,use_fast_weights=False,use_moe=False)); self.memory_state=memory_state
        from reference.hz0i_optional_integrations import ConditionalAnchorAttention,SessionFastWeights,RoutedSwiGLU
        self.memory = None
        if memory_state is not None:
            self.memory_query=torch.nn.Linear(config.n_embd,memory_state.keys.shape[-1]); self.memory_value=torch.nn.Linear(memory_state.values.shape[-1],config.n_embd); self.memory_gate=torch.nn.Linear(config.n_embd,config.n_embd)
        self.conditional=ConditionalAnchorAttention(config.n_embd,config.n_head) if config.use_conditional_attention else None
        self.fast=SessionFastWeights(config.n_embd) if config.use_fast_weights else None
        self.moe=RoutedSwiGLU(config.n_embd,config.n_embd*2,balanced_init=config.moe_balanced_init,router_noise=config.moe_router_noise) if config.use_moe else None
    def forward(self,idx,*,triggers=None):
        from reference.hz0b_memory_simulator_torch import read
        h=self.base.forward_hidden(idx); B,T,D=h.shape; diagnostics={}
        if self.memory_state is not None:
            q=self.memory_query(h).reshape(B*T,-1); st=self.memory_state; st=type(st)(*(getattr(st,k).repeat_interleave(T,dim=0) for k in st.__dataclass_fields__)); ro,w=read(st,q); c=self.memory_value(ro).reshape(B,T,D); h=h+torch.sigmoid(self.memory_gate(h))*c; diagnostics['memory_read_mean']=float(w.mean().detach())
        if self.conditional is not None:
            if triggers is None: raise ValueError('triggers required')
            h=h+self.conditional(h,triggers); diagnostics['trigger_rate']=float(triggers.float().mean())
        if self.fast is not None: h=h+self.fast.apply(h); diagnostics['fast_delta_norm']=float(self.fast.delta().norm().detach())
        if self.moe is not None:
            z,route=self.moe(h); h=h+z; diagnostics['expert_counts']=torch.bincount(route.reshape(-1),minlength=len(self.moe.experts)).detach().tolist()
        return h @ self.base.lm_head, diagnostics
