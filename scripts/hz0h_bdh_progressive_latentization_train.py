#!/usr/bin/env python3
"""Arm D of the progressive-latentization falsification experiment
(plans/newnewplan.md, "Progressive Latentization Training", 2026-08-31)
-- the real, novel piece. Coconut-style curriculum: start fully
explicit (identical to Arm C), then progressively replace the FIRST k
of a trace's intermediate-result digits with real recurrent computation
instead of visible text, growing k over training. LOTUS-style step
alignment: each latent position is supervised via the model's own real
lm_head against what the hidden digit WOULD have said, at a weight
that decays across the curriculum instead of staying fixed forever
(so explicit reasoning teaches the dynamics, but the final regime is
free to find something more efficient than literally reproducing
English/ASCII digits internally).

Real architectural design, since this needs a genuinely heterogeneous
forward pass (visible text mixed with latent-only computation) that
none of this project's existing forward functions support:

  1. Embed the question prefix ("r=3. add 2 sub 1. Results:") and run
     it through the normal parallel-over-T adaptive-gate forward
     (n_layer=8 rounds, always-refresh, matching how the base
     checkpoint was trained) -- nothing latentizes here, the question
     (including every operation) is never secret.
  2. For each of the first `n_latent` reasoning steps: APPEND one new
     sequence position whose initial value is a COPY of the current
     last position's state (not a token embedding -- there is no token
     here), then run exactly ONE real recurrent round (one exact
     address + one adaptive-gate-controlled write) over the WHOLE
     sequence so far. That new position's post-round state is this
     step's real latent computation. LOTUS supervision reads it out
     through the model's own lm_head against the ASCII digit that step
     truly produces.
  3. The remaining reasoning steps (n_latent+1 .. n_steps) plus the
     final ". Answer: D" span stay real, visible text -- embedded and
     appended normally, then the WHOLE sequence (question + latent
     positions + this visible continuation) gets n_layer=8 full rounds,
     exactly like ordinary LM training, with ordinary next-byte loss
     over only the real visible continuation positions.

Real, disclosed choice: one latent step = one recurrent ROUND (not a
full n_layer-deep sub-forward) -- matches this project's own "round =
one unit of computation, same tied weights every time" convention, and
directly gives each hidden reasoning step the same computational unit
the plan's own notation implies (Q -> h_1 -> h_2 -> ... -> h_R -> A,
"the latent token becomes a BDH recurrent iteration").

Recomputes rounds over the whole growing sequence at every step rather
than doing anything incremental/cached -- correct (full causal
self-attention over real history is preserved exactly), and cheap
enough at these trace lengths (tens of bytes) that efficiency wasn't
worth the real complexity of a cached/streaming variant here.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate, _refresh_iteration
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_register_machine_task import generate_register_machine_cot_example
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize


def load_adaptive_gate_checkpoint(config: BDHVBSubspaceDecoderConfig, checkpoint_path: Path, gate_hidden: int, device) -> BDHVBSubspaceDecoder:
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=gate_hidden, g_init=0.58, state_independent=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("answer_head")]
    assert not real_missing, f"real missing keys: {real_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"[load] loaded {checkpoint_path}", flush=True)
    return model


def _embed_bytes(model: BDHVBSubspaceDecoder, byte_ids: list[int], device) -> torch.Tensor:
    idx = torch.tensor([byte_ids], dtype=torch.long, device=device)
    return model.ln(model.embed(idx).unsqueeze(1))  # (1,1,T,D)


def _full_rounds(x: torch.Tensor, model: BDHVBSubspaceDecoder, n_rounds: int, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
    h_prev = x
    for _ in range(n_rounds):
        x_new, _e, _g = torch.utils.checkpoint.checkpoint(_refresh_iteration, x, h_prev, model, B, T, D, nh, N, use_reentrant=False)
        h_prev = x
        x = x_new
    return x


def forward_progressive_latent(model: BDHVBSubspaceDecoder, question_prefix: str, results_suffix_digits: list[int],
                                answer_digit: int, n_latent: int, n_layer: int, lambda_step: float, device):
    C = model.config
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    n_steps = len(results_suffix_digits)
    assert 0 <= n_latent <= n_steps

    q_bytes = list(question_prefix.encode("utf-8"))
    x = _embed_bytes(model, q_bytes, device)
    B, _, T0, _ = x.shape
    x = _full_rounds(x, model, n_layer, B, T0, D, nh, N)

    step_loss = x.new_zeros(())
    T = T0
    for i in range(n_latent):
        new_pos = x[:, :, -1:, :]
        x = torch.cat([x, new_pos], dim=2)
        T += 1
        x = _full_rounds(x, model, 1, B, T, D, nh, N)  # one real recurrent round = one reasoning step
        h_i = x[:, :, -1, :].reshape(B, D)
        logits_i = h_i @ model.lm_head  # LOTUS-style: read the latent state through the real lm_head
        target_byte = 48 + results_suffix_digits[i]  # ASCII '0'-'9'
        step_loss = step_loss + F.cross_entropy(logits_i, torch.tensor([target_byte], device=device))

    visible_digits = results_suffix_digits[n_latent:]
    visible_text = "".join(f" {d}" for d in visible_digits) + f". Answer: {answer_digit}"
    v_bytes = list(visible_text.encode("utf-8"))
    if len(v_bytes) < 2:
        v_bytes = v_bytes + [ord(".")]  # keep at least one real predicted byte even in the fully-latent-except-answer edge case
    v_embed = _embed_bytes(model, v_bytes, device)
    x = torch.cat([x, v_embed], dim=2)
    T = x.shape[2]
    x = _full_rounds(x, model, n_layer, B, T, D, nh, N)

    # ordinary next-byte LM loss over the visible continuation only:
    # position (T - len(v_bytes) - 1 + j) predicts v_bytes[j] for j=1..len-1
    # (the first v_byte has no real "previous real token" prediction target
    # inside this continuation -- its predecessor is the last latent/prefix
    # position, which legitimately predicts it, so include it too).
    start = T - len(v_bytes)
    logits = x[:, :, start - 1:T - 1, :].reshape(-1, D) @ model.lm_head
    targets = torch.tensor(v_bytes, device=device)
    lm_loss = F.cross_entropy(logits, targets)

    total_loss = lm_loss + (lambda_step * step_loss / n_latent if n_latent > 0 else lm_loss.new_zeros(()))
    return total_loss, lm_loss.detach(), (step_loss / max(n_latent, 1)).detach()


def lambda_schedule(step: int, total_steps: int) -> float:
    """1 -> 0.5 -> 0.1 -> 0 over training, per the plan's own schedule."""
    frac = step / max(total_steps, 1)
    if frac < 0.25:
        return 1.0
    if frac < 0.5:
        return 0.5
    if frac < 0.75:
        return 0.1
    return 0.0


def curriculum_n_latent(step: int, total_steps: int, n_steps: int) -> int:
    """Stage k grows 0 -> n_steps across training, matching the plan's
    table (stage 0 fully explicit == Arm C; stage n_steps fully latent
    except the final Answer)."""
    frac = step / max(total_steps, 1)
    k = int(frac * (n_steps + 1))
    return max(0, min(n_steps, k))


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = load_adaptive_gate_checkpoint(config, args.init_checkpoint, args.gate_hidden, device)
    optimizer = make_optimizer(model.parameters(), args, device)
    rng = random.Random(args.seed)
    step_pool = [1, 2, 3, 4, 6, 8]
    started = time.perf_counter()
    for step in range(args.n_examples):
        n_steps = rng.choice(step_pool)
        n_latent = curriculum_n_latent(step, args.n_examples, n_steps)
        lam = lambda_schedule(step, args.n_examples)
        q_prefix, _suffix, step_targets, answer = generate_register_machine_cot_example(rng, n_steps)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args, device):
            loss, lm_loss, step_loss = forward_progressive_latent(
                model, q_prefix, step_targets, answer, n_latent, args.n_layer, lam, device)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if args.log_every and (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            rate = (step + 1) / (now - started)
            eta = (args.n_examples - step - 1) / max(rate, 1e-6)
            print(f"[prog_latent] example {step+1}/{args.n_examples} n_steps={n_steps} n_latent={n_latent} "
                  f"lambda={lam:.2f} loss={float(loss):.4f} lm_loss={float(lm_loss):.4f} step_loss={float(step_loss):.4f} "
                  f"{rate:.1f} ex/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[prog_latent] DONE {args.n_examples} examples in {elapsed:.0f}s", flush=True)
    model.eval()
    return model, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_adaptive_gate_retrain_checkpoint.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-examples", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[prog_latent] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arm": "D", "elapsed_s": elapsed, "n_examples": args.n_examples}, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
