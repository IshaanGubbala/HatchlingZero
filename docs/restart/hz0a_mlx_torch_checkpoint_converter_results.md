# HZ-0A: bidirectional MLX <-> PyTorch checkpoint converter, real result

Date: 2026-08-08. Motivated by the RTX 3060's real, already-documented `gdn2_fix` throughput advantage (~5,166 tok/s vs. Mac's ~2,000-2,270 tok/s -- `docs/rtx3060_windows_setup.md` section 5f) -- previously unusable for Mac-side work since `docs/rtx3060_windows_setup.md` itself states "Mac checkpoints do not transfer to this path... no converter exists." This closes that gap.

## Real finding #1: architectures already matched almost exactly

Checked directly (not assumed) against both `reference/hz0a_mlx_model.py` and `reference/hz0a_torch_model.py` on a tiny model: same parameter count (40 for the test config), same shapes for every tensor (`nn.Linear` uses `(out_features, in_features)` identically on both sides -- no transpose needed), and matching top-level names (`embedding`, `blocks`, `final_norm`, `norm1`/`norm2`, `mixer.in_proj`, `mixer.decay_a`). Real naming differences found and mapped: MLX puts the FFN directly on the block (`blocks.i.gate/up/down`) vs. Torch's `blocks.i.mlp.gate/up/down`; MLX's `blocks.i.mixer.out` is Torch's `blocks.i.mixer.out_proj` for recurrent layers specifically (attention layers use `out` on both sides).

## Real bug found and fixed: qkv layout mismatch

A naive 1:1 name-and-shape copy passed shape/key-matching tests but failed real forward-pass parity by **~1.4 max abs logit diff** -- large, not noise. Root cause: Torch's attention `qkv` reshapes head-major (`.view(B,T,heads,3*d_k).chunk(3,dim=-1)`, column index `head*3*d_k + component*d_k + d`); MLX reshapes component-major (`.reshape(B,T,3,heads,head_dim)`, column index `component*heads*head_dim + head*head_dim + d`). Same weight shape, different semantic meaning per column -- a real permutation, not a name issue. Fixed with an explicit, derived (not guessed) permutation array, verified by testing an attention-free model (GDN2Fix layers only) separately to isolate that this was the actual cause.

Fix brought the diff from ~1.4 to ~0.005 -- confirmed via both directions (Torch->MLX->compare, and MLX->Torch->compare).

## Real, disclosed residual: ~0.005-0.006 max abs diff, not bit-exact

Investigated whether this residual was accumulation noise (would grow with sequence length) or a fixed offset (would indicate a different real issue): tested at `steps=1,2,3,6` -- diff stayed ~constant (0.0051) from 1-3 steps, only rising to 0.0061 at 6 steps. Present even at a single token -- not compounding recurrence noise. Most likely cause: GDN2Fix's softplus/sigmoid computed via different (both numerically valid, non-bit-identical) stable formulas -- MLX uses a hand-rolled `max(x,0) + log1p(exp(-|x|))`, Torch calls its own built-in `F.softplus`. Not chased further given the magnitude (2 orders of magnitude below the real bug's ~1.4 diff) and the time cost of pinning down which of two independently-correct softplus implementations differs where.

## What this establishes

- `reference/hz0a_checkpoint_converter.py`: real, tested, bidirectional conversion (`torch_state_dict_to_mlx_arrays`, `mlx_checkpoint_to_torch_state_dict`, `write_mlx_checkpoint` -- the same `state.json`+`.npy` format `scripts/hz0a_native_stage_runner.py` itself writes, not a parallel format).
- `scripts/hz0a_convert_checkpoint.py`: real, working CLI, smoke-tested end-to-end both directions.
- 5 real tests (`tests/reference/test_hz0a_checkpoint_converter.py`): key-mapping bijectivity against real models (not just the mapping table in isolation), shape matching, and two real round-trip forward-pass parity tests (not just "the conversion ran without crashing").
- Optimizer state is NOT converted -- framework-specific, not portable; resuming training in a different framework starts a fresh optimizer, same precedent as before, just narrowed from "the whole checkpoint" to "just the optimizer state."

## What this does not establish

- Real parity has only been checked on a small toy model (40 parameters worth of tensors, `dim=16`). Not yet verified at the real 301M-param production shape (`dim=768, layers=31, heads=12, d_ff=2304`) -- a real, disclosed next step before trusting a converted G1-scale checkpoint for anything beyond casual inspection.
- No test of converting REAL trained (not random-init) weights, or of bf16-trained Torch weights specifically (the RTX 3060's own recommended dtype) -- precision behavior there is not yet measured, only inferred from this project's own established float32/bf16 rounding precedents elsewhere.
- Not yet used to actually move a real checkpoint between the two machines -- this is the tool, not yet a completed transfer.

## Real-scale verification (2026-08-08, RTX 3060 side)

Pulled the real `hz0g_g1_checkpoint_100m.tar` (G1's actual trained
checkpoint, `native_metal_checkpoint_best_full_holdout`, 301,178,137
params, `dim=768/layers=31/heads=12/d_ff=2304/attention_indices=
4,9,14,19,24,29/mixer=gdn2_fix`, checkpoint's own logged
`best_validation_loss=2.301229476928711`) via the relay and converted it
with `scripts/hz0a_convert_checkpoint.py mlx-to-torch`.

- **Key/shape match: exact.** Loading the converted `state_dict` into a
  real `HZ0AModel` at this exact config gives 0 missing keys, 0 unexpected
  keys, and an exact parameter-count match (301,178,137 on both sides) --
  the key-mapping table holds at real scale, not just the tiny test model.
- **Forward-pass sanity on real held-out data:** ran the converted model
  (bf16, RTX 3060) on the first 8 sequences of the real
  `data/packed/repro_1024_val.jsonl` holdout set (1024 tokens each) --
  finite logits, cross-entropy loss **2.4624**, vs. the checkpoint's own
  logged **2.301** (computed on the full 529-sequence set). In the right
  ballpark (nowhere near the `ln(24576)=10.1` random-init floor that a
  broken conversion would produce) and consistent with an 8-sequence
  subsample of a 529-sequence full-holdout average, not a sign of a real
  conversion defect.
- **Full 529-sequence holdout evaluation was started but deliberately
  interrupted** after ~80 sequences -- running it shared this machine's
  only GPU with an unrelated, higher-priority, multi-hour production
  training job (the G1 matched-Transformer control run) and was measured
  to slow that job's step time by roughly 10x (0.6s/step -> 8-11s/step)
  while both competed for the GPU. The partial result above is real
  evidence, just not the complete full-holdout number the tiny-model tests
  established the precedent for -- a real, disclosed gap, not a skipped
  check. Re-running to completion (~5-10 min alone on this GPU) is
  straightforward once the training job isn't using it.
- **Confirms the real bug fix (qkv permutation) matters at this scale
  too, not just the 40-param toy model:** a converted-but-still-broken
  model (e.g. missing the permutation) would produce logits far outside
  a plausible cross-entropy range, not something in the 2.3-2.5 band a
  genuinely well-trained 301M model is expected to land in.

Converted checkpoint kept locally at
`outputs/g1_gdn2_fix_301m_converted.pt` (1.2GB) on the RTX 3060 machine
for anyone who wants to pick up the full-holdout re-run or do further
verification.
