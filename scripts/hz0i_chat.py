#!/usr/bin/env python3
"""Talk to the trained 0.3B HZ-0I BDH. Interactive REPL or --prompt one-shot.
Usage:
  python scripts/hz0i_chat.py                          # interactive
  python scripts/hz0i_chat.py --prompt "def fib(n):" --max-new 96
"""
import argparse, torch
from pathlib import Path
from reference.hz0i_factorized_layerwise_untied import FactorizedLayerwiseBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
from tokenizer.hz0a_tokenizer import HZ0ATokenizer

VOCAB_USED = 6358  # ids actually present in corpus; head rows beyond are untrained


def load(ckpt: Path, dev: str = "mps"):
    q = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_d = {k: v for k, v in q["config"].items() if k in
             ("n_layer","n_embd","n_head","mlp_internal_dim_multiplier","vocab_size",
              "moe_capacity_factor","moe_routing","moe_fallback_threshold","moe_aux_weight",
              "moe_z_weight","moe_balanced_init","moe_router_noise","learned_triggers",
              "trigger_aux_weight","trigger_threshold","trigger_mode","trigger_fraction")}
    c = HZ0IBDHConfig(**cfg_d, dropout=0., use_conditional_attention=True,
                      use_fast_weights=True, use_moe=True)
    stride = q.get("stride", 2)
    m = FactorizedLayerwiseBDH(c, 704, stride).to(device=dev, dtype=torch.bfloat16)
    m.attn.freqs = m.attn.freqs.float()
    m.load_state_dict(q["model"]); m.eval()
    return m, q["step"]


@torch.no_grad()
def generate(m, toks: list[int], max_new: int = 80, temperature: float = 0.8,
             top_k: int = 50, bos_id: int = 0, pad_id: int = 2):
    dev = next(m.parameters()).device
    if toks[0] != bos_id:
        toks = [bos_id] + toks
    x = torch.tensor([toks], device=dev)
    for _ in range(max_new):
        tr = None  # learned triggers compute their own mask
        logits, _ = m(x[:, :-1], triggers=tr, targets=None) if x.shape[1] > 1 else m(x, triggers=tr, targets=None, return_hidden=False)
        # when x.shape[1]==1 the slice x[:,:-1] would be empty; handle below
        break
    # cleaner loop: always keep at least 1 context token
    x = torch.tensor([toks], device=dev)
    for _ in range(max_new):
        ctx = x if x.shape[1] >= 1 else x
        if ctx.shape[1] == 1:
            logits,_ = m(ctx, triggers=None, targets=None)
        else:
            logits,_ = m(ctx[:, :-1], triggers=None, targets=None)
        z = logits[0, -1].float()
        z[:VOCAB_USED] = z[:VOCAB_USED] - 1e9 if False else z[:VOCAB_USED]
        z[pad_id] = -1e9
        z[VOCAB_USED:] = -1e9
        z = z / max(temperature, 1e-4)
        if top_k and top_k > 0:
            k = min(top_k, VOCAB_USED)
            tv, ti = torch.topk(z, k)
            z = torch.full_like(z, -float("inf")); z[ti] = tv
        p = torch.softmax(z, dim=-1)
        nxt = torch.multinomial(p, 1).item()
        x = torch.cat([x, torch.tensor([[nxt]], device=dev)], dim=1)
        if nxt == 1:  # eos
            break
    return x[0].tolist()[len(toks):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/hz0i_codereason_5000.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/hz0a_24576.json")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = HZ0ATokenizer.from_file(a.tokenizer)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m, step = load(Path(a.checkpoint), dev)
    print(f"[loaded ckpt step {step} on {dev}]")
    if a.prompt is not None:
        out = generate(m, tok.encode(a.prompt), a.max_new, a.temperature, a.top_k)
        print("\n--- model ---\n" + a.prompt + tok.decode(out))
        return
    print("Type a prompt (Ctrl-D to quit).")
    while True:
        try:
            p = input("> ")
        except EOFError:
            break
        if not p.strip(): continue
        out = generate(m, tok.encode(p), a.max_new, a.temperature, a.top_k)
        print(p + tok.decode(out) + "\n")


if __name__ == "__main__":
    main()
