#!/usr/bin/env python3
"""LoRA-continuation benchmark for the 0.3B BDH.
Loads a checkpoint, freezes the base, adds low-rank adapters on the factorized
enc/val/dec projections + trains lm_head/gates, then measures MPS memory and
tok/s at several batch sizes (user-requested LoRA/QLoRA strategy test).
"""
import argparse, time, torch, sys, os
sys.path.insert(0, os.getcwd())
from pathlib import Path
from reference.hz0i_factorized_layerwise_untied import FactorizedLayerwiseBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig


def build(config_d, stride, dev, ckpt):
    c = HZ0IBDHConfig(**{k: v for k, v in config_d.items() if k in
        ("n_layer","n_embd","n_head","mlp_internal_dim_multiplier","vocab_size",
         "moe_capacity_factor","moe_routing","moe_fallback_threshold","moe_aux_weight",
         "moe_z_weight","moe_balanced_init","moe_router_noise","learned_triggers",
         "trigger_aux_weight","trigger_threshold","trigger_mode","trigger_fraction")},
        dropout=0., use_conditional_attention=True, use_fast_weights=True, use_moe=True)
    m = FactorizedLayerwiseBDH(c, 704, stride).to(device=dev, dtype=torch.bfloat16)
    m.attn.freqs = m.attn.freqs.float()
    m.load_state_dict(ckpt["model"])
    return m


def add_lora(m, ra):
    H, D, N = m.config.n_head, m.config.n_embd, m.config.mlp_internal_dim_multiplier*m.config.n_embd//m.config.n_head
    for p in m.parameters(): p.requires_grad = False
    dev = next(m.parameters()).device; dt = next(m.parameters()).dtype
    def lora_param(shape):
        return torch.nn.Parameter(torch.zeros(shape, device=dev, dtype=dt))
    m.lora_enc_a = lora_param((H, D, ra)); m.lora_enc_b = lora_param((H, ra, N))
    m.lora_val_a = lora_param((H, D, ra)); m.lora_val_b = lora_param((H, ra, N))
    m.lora_dec_a = lora_param((H, N, ra)); m.lora_dec_b = lora_param((H, ra, D))
    with torch.no_grad():
        for a in ("lora_enc_a","lora_val_a","lora_dec_a"):
            getattr(m, a).normal_(std=0.01)
    def enc_lo(x, l, r):
        z = torch.einsum("bhtd,hdr->bhtr", x, l); z = torch.einsum("bhtr,hrn->bhtn", z, r)
        return z + torch.einsum("bhtd,hda,han->bhtn", x, m.lora_enc_a, m.lora_enc_b)
    def val_lo(x, l, r):
        z = torch.einsum("bhtd,hdr->bhtr", x, l); z = torch.einsum("bhtr,hrn->bhtn", z, r)
        return z + torch.einsum("bhtd,hda,han->bhtn", x, m.lora_val_a, m.lora_val_b)
    def dec_lo(x):
        z = torch.einsum("bhtn,hnr->bhtr", x, m.dec_l); z = torch.einsum("bhtr,hrd->bhtd", z, m.dec_r).sum(1, keepdim=True)
        add = torch.einsum("bhtn,hna,had->bhtd", x, m.lora_dec_a, m.lora_dec_b).sum(1, keepdim=True)
        return z + add
    m._enc = enc_lo; m._enc = val_lo if False else m._enc; m._val_enc = val_lo
    # forward_hidden calls self._enc for BOTH enc and val -> need to route val separately:
    import types, reference.hz0i_factorized_layerwise as fl
    orig_fh = m.forward_hidden
    def fh(self, idx, layer_hook=None):
        C=self.config; B,T=idx.shape; x=self.ln(self.embed(idx).unsqueeze(1))
        for level in range(C.n_layer):
            xs=torch.relu(enc_lo(x,self.enc_l,self.enc_r))
            ykv=self.ln(self.attn(Q=xs,K=xs,V=x))
            ys=torch.relu(val_lo(ykv,self.val_l,self.val_r))
            x=self.ln(x+self.ln(dec_lo(self.drop(xs*ys))))
            if layer_hook is not None and level%self.layer_stride==0: x=layer_hook(x,level)
        return x.view(B,T,C.n_embd)
    m.forward_hidden = types.MethodType(fh, m)
    # trainable set: adapters + lm_head + gates
    for name, p in m.named_parameters():
        if name.startswith("lora_") or name == "lm_head" or name.endswith("_gate"):
            p.requires_grad = True
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[LoRA] adapters added, trainable params: {ntr/1e6:.1f}M", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/hz0i_codereason_5000.pt")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    m = add_lora(build(ck["config"], ck.get("stride", 2), dev, ck), a.lora_rank)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    def step(b):
        x = torch.randint(0, 24576, (b, a.seq), device=dev)
        logits, loss = m(x[:, :-1], triggers=None, targets=x[:, 1:])
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    step(a.batch); torch.mps.synchronize()
    mem = torch.mps.current_allocated_memory()/1e9
    for _ in range(2): step(a.batch)
    torch.mps.synchronize(); t0 = time.perf_counter()
    for _ in range(a.steps): step(a.batch)
    torch.mps.synchronize(); dt = (time.perf_counter()-t0)/a.steps
    print(f"LoRA b{a.batch} s{a.seq}: {a.batch*a.seq/dt:.0f} tok/s  mps_mem {mem:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
