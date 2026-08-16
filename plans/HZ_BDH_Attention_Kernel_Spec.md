# HZ BDH attention kernel: build spec

**Status**: implementation complete locally; real-CUDA correctness and
benchmark verification pending. The exact-math bounded reference path and
the optional compiled Triton path are implemented in separate files, with
local parity coverage and a full-suite pass. This plan remains open until
the RTX 3060 artifacts required below are downloaded and audited.

## 1. Why this exists (context, briefly)

HatchlingZero is a research project comparing a BDH (Dragon Hatchling)
language-model architecture against a parameter-matched Transformer, on
real hardware, with real measured results (see `README.md` for the full
picture). The headline finding so far: BDH wins on quality (validation
loss) by a real, decisive margin at both ~25M and ~100M parameters, but
loses on training speed and memory by a large margin (currently ~5-7x
slower, ~2-6x more memory depending on variant and whether activation
checkpointing is applied — see `docs/restart/hz0h_phase_f_training_target_gate_results.md`
and `docs/restart/hz0h_phase_g_checkpointed_retry_results.md` for the
real numbers).

Real, profiler-confirmed root cause of part of that gap
(`docs/restart/hz0h_bdh_fused_attention_results.md`): BDH's own
attention has **no fused GPU kernel at all**. PyTorch's
`scaled_dot_product_attention` (what the Transformer control uses) is a
hand-tuned, vendor-maintained fused kernel — but it always applies
softmax internally, and BDH's attention has no softmax, so it cannot be
used. The one fused-kernel alternative already tried this session
(`flash_linear_attention`'s `chunk_gla`, the same Triton kernel family
RetNet/GLA/Mamba use) made things **worse**, not better — 1.5x to 49x
slower depending on config, profiler-confirmed real reason: `chunk_gla`
is built for the `T ≫ N` regime (long sequences, small per-head state —
the regime RetNet/GLA/Mamba actually operate in), and BDH's real shape
at this project's configs is the **opposite**: `N ≫ T` (large per-head
state, short-to-moderate sequences). See
`docs/restart/hz0h_bdh_fused_attention_results.md`'s "Root cause of the
remaining ~2.6x" section for the full profiler breakdown — do not
re-attempt `chunk_gla` or any kernel from that same family without a
real reason to expect the shape mismatch is fixed.

**This spec is for a genuinely new kernel, hand-built for BDH's own
actual shape (`N ≫ T`), not adapted from an existing kernel library
built for a different regime.** No such kernel exists anywhere in the
current ecosystem — this is real, from-scratch infrastructure work, not
a config change.

## 2. The exact math to implement

Source of truth: `reference/hz0h_bdh_torch.py`'s `Attention` class
(read it directly, this section is a faithful transcription, not a
summary). **This file is a verbatim, byte-faithful port of the real
upstream `pathwaycom/bdh` — do not modify it. All new work goes in a
separate file (see section 5).**

```python
# RoPE frequency buffer, computed once at model init, ALWAYS float32
# regardless of the model's own training dtype (this is a real,
# enforced invariant elsewhere in this codebase — see section 4):
def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return 1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi)

freqs = get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)  # shape (1,1,1,N)

# Per-forward-call RoPE application:
def phases_cos_sin(phases):
    phases = (phases % 1) * (2 * math.pi)
    return torch.cos(phases), torch.sin(phases)

def rope(phases, v):
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    phases_cos, phases_sin = phases_cos_sin(phases)
    return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

# The actual attention computation (Attention.forward):
def attention_forward(Q, V, freqs):
    # Q: (B, nh, T, N).  K IS Q — the real upstream code asserts `K is Q`,
    # there is no separate key tensor at all.
    # V: (B, 1, T, D) — ONE value tensor, broadcast (not split) across
    # every head. This is a real, deliberate asymmetry vs standard
    # multi-head attention — do not "fix" it by splitting V, that would
    # change the math (a different, already-explored idea, see
    # reference/hz0h_bdh_split_v_torch.py, NOT what this kernel is for).
    B, nh, T, N = Q.shape
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs
    QR = rope(r_phases, Q)          # (B, nh, T, N)
    KR = QR                         # literally the same tensor, not a copy
    scores = (QR @ KR.mT).tril(diagonal=-1)   # (B, nh, T, T) — STRICTLY lower
                                                # triangular: diagonal=-1 means
                                                # position t attends to s < t
                                                # ONLY, never to itself. This is
                                                # a real, deliberate design
                                                # choice in the upstream source,
                                                # not a bug — get this exactly
                                                # right, off-by-one here is a
                                                # real, silent correctness bug.
    return scores @ V                # (B, nh, T, D), V broadcasts (B,1,T,D) -> (B,nh,T,D)
```

**No softmax anywhere. No normalization anywhere.** This is a raw,
unnormalized, strictly-causal weighted sum. That is the entire reason a
standard fused-attention kernel cannot be reused — this is the actual
math the new kernel must implement exactly.

## 3. The real target shape regime (the whole point of this kernel)

Two real, currently-used configs, both with `N ≫ T`:

| config | `n_embd` (D) | `n_head` (nh) | `mult` | `N = D*mult/nh` | `T` (seq_len) | `B` (batch) |
|---|---:|---:|---:|---:|---:|---:|
| Phase F (25.4M params) | 512 | 8 | 32 | **2048** | 256 | 12 |
| Phase G (101M params) | 1024 | 8 | 32 | **4096** | 256 | 12 |

Design and tune the kernel for `N` in the low-thousands and `T` in the
low-hundreds — the opposite of what `chunk_gla`/FlashAttention-style
kernels assume (`T` in the thousands-to-millions, `N`/head-dim in the
tens-to-low-hundreds). Do not assume techniques from that literature
transfer without re-deriving them for this regime. `dtype` is
`bfloat16` for all real runs (the RoPE `freqs` buffer itself must stay
`float32` regardless — a real, enforced assertion in the existing code,
`assert self.freqs.dtype == torch.float32`, replicate this constraint).

## 4. Real, non-negotiable correctness bar

Every existing "alternative attention path" in this repo is held to
this exact standard — do not ship anything looser. See
`reference/hz0h_bdh_fused_attention_torch.py` and
`reference/hz0h_bdh_compiled_attention_torch.py` for the precedent this
follows exactly:

1. **Forward exactness**: given the same `Q`, `V`, `freqs`, and RNG
   seed, the new kernel's output must match `Attention.forward`'s own
   output within `torch.allclose(..., atol=1e-3, rtol=1e-3)` — tight
   tolerance, small differences from accumulation-order are expected
   and fine, anything looser is not.
2. **Multiple shapes/seeds**: test at several `(B, nh, T, N)`
   combinations, not just the two target configs above — a bug that
   only shows up at odd `T` or `nh>1` with the broadcast `V` path is
   real and has happened before in this codebase.
3. **Gradient correctness**: `Q.grad` after `.backward()` through the
   new kernel must match the oracle's own `Q.grad` within
   `torch.allclose(..., atol=1e-2, rtol=1e-2)` (looser than forward —
   different kernels/accumulation orders produce close-but-not-identical
   gradients, this is the established tolerance for that, not a
   loophole to be looser than necessary). `K is Q` in the real math —
   gradients must correctly accumulate into the SAME tensor from both
   its uses, not silently drop one path.
4. Real correctness tests belong in a new
   `tests/reference/test_hz0h_bdh_<kernel-name>_torch.py`, following
   the exact structure of
   `tests/reference/test_hz0h_bdh_fused_attention_torch.py` (read that
   file directly as the template).

**No benchmark number from this kernel is meaningful until correctness
passes.** Do not report speed/memory results before the correctness
tests above are green.

## 5. Real repo integration conventions (non-negotiable)

- **Never modify `reference/hz0h_bdh_torch.py`.** It is a verbatim,
  byte-faithful port of the real upstream source, diffed line-by-line
  against a fresh fetch — treated as read-only ground truth throughout
  this repo. All new work goes in a new file,
  `reference/hz0h_bdh_<kernel-name>_torch.py` (pick a real, descriptive
  name, e.g. `hz0h_bdh_native_kernel_attention_torch.py`), following the
  exact pattern of `reference/hz0h_bdh_fused_attention_torch.py`: a long
  module docstring explaining real motivation/status/tolerances (model
  this doc's own section 1-4 above), a `bdh_kernel_forward(model, idx,
  targets=None)` function that is byte-identical to `BDH.forward`
  except for the one attention call, which routes through the new
  kernel instead of `model.attn(...)`.
- Every new script (benchmarks, standalone test runners) must start
  with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
  before importing anything from `reference.*` — this exact bug (a
  missing version of this line) has broken multiple new scripts this
  session when run via `subprocess`/dispatched to a different machine;
  don't repeat it.
- Use `.venv/bin/python3` explicitly for anything run locally in this
  repo (not bare `python`/`python3`).
- Before considering any change done: run the FULL test suite
  (`.venv/bin/python3 -m pytest -q tests/`) and confirm it's green (as
  of this spec being written: 758 passed, 103 skipped is the expected
  baseline — a new PASS count higher than 758 after adding tests is
  correct, a LOWER count or new failures means something broke).
- **Never fabricate or round up a result.** If the kernel is correct
  but slower, or only helps in one config and not the other, report
  that plainly — this project's entire evidence trail
  (`docs/restart/`) is built on real, disclosed negative results
  alongside positive ones, and every doc in that directory follows the
  same discipline: real numbers, real caveats, no overclaiming.

## 6. Real benchmark and reporting requirements

Once correctness passes:

1. **Raw kernel vs. existing raw matmul** (`Attention.forward`'s own
   current implementation), at both real configs from section 3, real
   CUDA hardware (this project develops on a Mac + dispatches real GPU
   work to a Windows/RTX3060 machine — CPU/MPS numbers are not the real
   target and should be marked as such if that's all that's available).
   Real training-step benchmark: forward + backward + optimizer step,
   5 warmup + 15-20 timed steps, `batch=12, seq=256`, `bf16`, report
   tokens/sec and peak memory
   (`torch.cuda.max_memory_allocated()`/`reset_peak_memory_stats()`,
   the correct API — a prior script in this repo used a non-existent
   `torch.synchronize()` instead of `torch.cuda.synchronize()` and it
   silently broke on first real CUDA use; verify API calls against real
   CUDA, not assumed from CPU/MPS testing).
2. **New kernel vs. the matched Transformer directly**, not just vs.
   raw BDH matmul — this project was burned once already by reporting
   a real win against a weak internal control (BlockBDH vs. dense BDH)
   that did not translate into closing the gap against the actual
   Transformer baseline (`docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md`
   is the real example of this mistake and its correction — read it as
   a cautionary precedent). Use
   `scripts/hz0h_training_target_gate.py`'s own real thresholds as the
   bar: throughput ratio ≥1.30, peak RAM ratio ≤0.70, both vs. a
   parameter-matched Transformer under matched hardware/dtype/token
   budget/optimizer conditions — report the real ratio whether or not
   it clears the gate.
3. Write up the real result in a new
   `docs/restart/hz0h_bdh_native_kernel_results.md`
   (match the tone/structure of
   `docs/restart/hz0h_bdh_fused_attention_results.md`: real numbers,
   real root-cause analysis if it underperforms, no overclaiming if it
   doesn't clear the gate — a real, well-understood partial win is a
   valid, useful, reportable outcome, same as every other result this
   project has produced).

## 7. How to get real CUDA hardware for this (the dispatch relay)

This repo is worked on from a Mac (no CUDA) that dispatches real GPU
work to a separate Windows machine with an RTX 3060, through a
Raspberry Pi acting as a relay hub. **Full protocol reference:
`CLAUDE.md` at the repo root — read that file directly, it is the
canonical, currently-accurate source, not this summary.** Key points:

- All three machines share a Tailscale tailnet already — the Pi's
  Tailscale IP is `100.87.180.87`, port `8899`, HTTPS (self-signed
  cert, use `curl -sk`).
- Auth is an `X-Auth: <token>` header. **The real token is deliberately
  NOT in this file or anywhere in git** (this repo is public) — read it
  directly from `~/hz0a_transfer/serve.py` on whichever machine you're
  targeting (see `CLAUDE.md`'s own "Finding the real credentials"
  section). If you don't have access to that file, ask the user for it
  rather than guessing.
- Real HTTP surface once you have the token: `GET /` (list the Pi's
  outbox), `GET /outbox/<name>` / `GET /inbox/<name>` (download),
  `PUT /inbox/<name>` (upload, body = raw bytes), `POST /chat/<name>`
  (plain-text status ping, body = raw UTF-8), `GET /chat?since=N`
  (read the chat log incrementally). There is no `PUT /outbox/<name>`
  — writing INTO the Pi's outbox (the channel Windows polls for new
  work) goes via SSH/scp to `gubbi@100.87.180.87:~/hz0a_transfer/outbox/`
  instead (or the dashboard's upload form, see `CLAUDE.md`).
- **Real dispatch pattern**: write a plain-text request file describing
  the exact command(s) to run and what to report back (see any existing
  file in the Pi's outbox for the real style — list them with
  `GET /`), drop it in the outbox, then poll `GET /inbox/<expected-result-filename>`
  (404 until it lands) or watch `/chat` for a status ping. This is
  real, human-readable async coordination with whoever/whatever is
  running on the Windows side — not a job-queue API — so write the
  request like a message to a colleague: exact commands, exact config,
  exact filenames to report back, and what question the result should
  answer.
- For this kernel spec specifically: dispatch the correctness tests
  first (cheap, fast, must pass before anything else matters), then the
  benchmark runs from section 6 above, at both real configs, requesting
  the exact JSON/text output files be uploaded back so the real numbers
  can be pulled and written up honestly.

## 8. Explicitly out of scope for this kernel

- Do not change BDH's math (no softmax added, no `V` splitting per
  head, no normalization) — those are different, separate, already-
  explored ideas (see `reference/hz0h_bdh_split_v_torch.py` for the
  `V`-splitting one) with their own real results. This spec is for an
  exact-math-preserving kernel only.
- Do not attempt to also fuse the surrounding block computation
  (`encoder`/`encoder_v`/`decoder`/gating) in this same deliverable —
  scope this to the attention step specifically
  (`scores = tril(QR@KR.mT); out = scores@V`, plus the RoPE application
  that feeds it). A wider-scope fused kernel for the whole per-layer
  block is a real, separate, larger follow-up, not this spec.
- GDN-2 Fix's own kernel need (a different mixer, blocked on a missing
  exact delta-rule CUDA kernel — see
  `docs/restart/hz0h_gdn2_fix_efficiency_kernel_prerequisite.md`) is a
  **separate, unrelated** kernel-writing task for a different
  architecture. Do not conflate the two.
