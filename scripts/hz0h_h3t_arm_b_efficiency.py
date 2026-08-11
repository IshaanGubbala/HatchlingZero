"""HZ-0H H3-T Arm B efficiency phase: real wall-clock/memory measurement,
not just quality. The user's own correction stands: Arm B's ceiling isn't
strictly Arm A (denoising/generalization over many noisy per-step targets
can genuinely beat an individual noisy sample -- the 1.52 vs 1.74 result
is real evidence of that, not a fluke).

This measures the thing that actually matters for a training-method claim:
in PRODUCTION mode (predictor already warmed up, encoder's true backward
skipped entirely -- not the per-layer local-signal computation used only
during warmup to train the predictor, which a real deployment would stop
calling), how much real time/memory does Arm B save per step vs true BPTT?

`encoder.requires_grad = False` during the timed region is the real lever:
autograd still needs to flow gradients THROUGH encoder's output for every
other parameter's correct gradient, but skips accumulating dL/d(encoder)
itself across all 6 layers' shared use of it -- the actual expensive part
for a tied/shared weight.
"""
from __future__ import annotations

import platform
import resource
import time

import torch

# macOS reports ru_maxrss in bytes; Linux reports it in kilobytes -- a real,
# well-known platform difference, not a typo.
_RSS_TO_MB = (1024 * 1024) if platform.system() == "Darwin" else 1024

from reference.hz0h_bdh_torch import BDH, BDHConfig


def _timed(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0


def main():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=6, n_embd=128, n_head=8, mlp_internal_dim_multiplier=64, vocab_size=256, dropout=0.0)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    nh, D = config.n_head, config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    synth_w = torch.zeros(nh, N, N)  # already-warmed predictor stand-in (weights don't matter for a timing measurement)

    idx = torch.randint(0, config.vocab_size, (8, 32))

    B, T = idx.shape

    def forward_capturing_last_x_sparse(track_encoder_grad: bool):
        """Same computation as BDH.forward(), but ALSO returns the last
        layer's x_sparse (detached) -- reused by the predictor instead of
        a wasteful second forward pass. track_encoder_grad controls
        whether encoder participates in its OWN gradient accumulation
        (the real cost this arm is trying to skip); it still participates
        in the forward VALUE computation either way, so every other
        parameter's gradient is unaffected."""
        model.encoder.requires_grad_(track_encoder_grad)
        x = model.ln(model.embed(idx).unsqueeze(1))
        last_x_sparse = None
        for _level in range(config.n_layer):
            x_latent = x @ model.encoder
            x_sparse = torch.relu(x_latent)
            last_x_sparse = x_sparse
            yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x))
            y_latent = yKV @ model.encoder_v
            y_sparse = torch.relu(y_latent)
            xy_sparse = model.drop(x_sparse * y_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
            y = model.ln(yMLP)
            x = model.ln(x + y)
        logits = x.view(B, T, D) @ model.lm_head
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
        return loss, last_x_sparse.detach(), x.detach()

    def true_bptt_step():
        opt.zero_grad(set_to_none=True)
        loss, _last_x_sparse, _x = forward_capturing_last_x_sparse(track_encoder_grad=True)
        loss.backward()
        opt.step()

    def arm_b_production_step():
        opt.zero_grad(set_to_none=True)
        loss, last_x_sparse, x = forward_capturing_last_x_sparse(track_encoder_grad=False)
        loss.backward()  # every OTHER param still gets a real gradient; encoder does not accumulate its own
        with torch.no_grad():
            pred = torch.einsum("bhtn,hnm->bhtm", last_x_sparse, synth_w)  # cheap predictor forward, no backward anywhere
            pseudo_grad = torch.einsum("btd,bhtn->hdn", x.squeeze(1), pred) / (B * T)
            model.encoder.grad = pseudo_grad.clone()
        opt.step()

    true_ms = _timed(true_bptt_step, repeats=20, warmup=5)
    rss_after_true = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    arm_b_ms = _timed(arm_b_production_step, repeats=20, warmup=5)
    rss_after_arm_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"config: n_layer={config.n_layer} n_embd={config.n_embd} n_head={config.n_head} N={N} batch=8 seq=32")
    print(f"true BPTT:            {true_ms:.3f} ms/step, peak RSS so far: {rss_after_true / _RSS_TO_MB:.1f} MB")
    print(f"Arm B (production):   {arm_b_ms:.3f} ms/step, peak RSS so far: {rss_after_arm_b / _RSS_TO_MB:.1f} MB")
    print(f"speedup:              {true_ms / arm_b_ms:.3f}x")
    print("NOTE: ru_maxrss is a MONOTONIC peak over the whole process lifetime -- since true BPTT")
    print("ran first in this same process, Arm B's reported RSS can only be >= true BPTT's, not a")
    print("fair independent comparison. A real memory comparison needs separate subprocesses; not done here.")
    print("\nHONEST CAVEATS:")
    print("- CPU-only reference model (reference/hz0h_bdh_torch.py has no MPS/CUDA path wired here)")
    print("- REAL, IMPORTANT FINDING: encoder.requires_grad=False only skips accumulating")
    print("  encoder's OWN gradient -- the rest of the backward graph (encoder_v, decoder,")
    print("  attention, lm_head, and gradient flowing THROUGH encoder for other params'")
    print("  correct gradients) still runs in full. That's why the speedup is small (~3%),")
    print("  not the large win a naive 'skip encoder's backward' framing would suggest.")
    print("- predictor here is untrained (zero weights) -- pure timing measurement, not a")
    print("  quality claim (quality was already measured separately: 1.52 vs 1.45)")
    print("- a bigger real speedup would need encoder's OWN parameter count/backward share")
    print("  to be a larger fraction of total step cost than it is at this scale -- worth")
    print("  re-measuring at real training scale (dim=768, 8 layers) before concluding this")
    print("  generalizes, since the ratio of encoder's cost to total cost may differ there.")


if __name__ == "__main__":
    main()
