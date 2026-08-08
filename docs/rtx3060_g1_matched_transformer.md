# RTX 3060: matched-Transformer control for G1's corpus

Written 2026-08-07. Purpose: the one comparison that's currently missing.
`HZ-0G`'s `G1` run (Mac, `gdn2_fix`, live) trains on a real, diverse 112M-token
corpus (general text, code, docs, math, JSON, terminal transcripts) at
`sequence_length=1024`. The only existing matched-Transformer control
(`outputs/hz0a_stage2_100m_transformer_seed7`) was trained on a *different*,
pure-general-text corpus at `sequence_length=256` -- comparing G1's numbers
against it directly is invalid (three real confounds: corpus composition,
context length, eval-set size). This doc is how to remove that confound: a
same-corpus, same-seq-len, same-token-budget Transformer control, trained on
this machine so it doesn't compete with G1 for the Mac's GPU.

Read `docs/rtx3060_windows_setup.md` first if this is a cold start on this
machine -- environment setup, `flash-linear-attention`, throughput-tuning
history, and known gaps vs. the Mac runner are all there and not repeated
here. This doc only covers what's different for this specific run.

## Correction (2026-08-07, RTX 3060 run): don't use `--chunk-length`/`--truncate-backward`/`--compile-step` here

Three findings from actually running this doc's original commands, before
either the smoke test or real run below was retried with these dropped:

1. **`--chunk-length`/`--truncate-backward` silently break `--architecture
   transformer`.** `scripts/hz0a_torch_stage2_runner.py`'s truncate-backward
   loop (`main()`, around line 404-410) calls `train_step_fn(chunk)` with
   `next_state = None` unconditionally for `--architecture transformer` --
   there's no recurrent state to carry across chunks the way there is for
   `gdn2`/`gdn3`/`gdn2_fix`. So `--chunk-length 8 --truncate-backward` doesn't
   do truncated-BPTT for a plain transformer; it silently trains on
   independent 8-token windows with **no cross-chunk context at all**,
   defeating the entire point of this run (matching G1's `sequence_length
   =1024`). It's also badly inefficient this way: 1024/8 = 128 microbatches
   of full 31-layer forward+backward launch overhead per optimizer step for
   trivial 8-token computations -- observed as a genuine stall (95-99% GPU
   utilization, memory pinned near 11-12GB, **zero** steps completing in 3+
   minutes) at every batch size tried (128, 256), easy to mistake for the
   WDDM slowdown pattern in `docs/rtx3060_windows_setup.md` section 5b but
   actually a different failure mode. **Do not pass either flag with
   `--architecture transformer`** -- let the runner's non-truncated branch
   run full self-attention over the whole `--sequence-length` each step
   (this is what it does automatically when both flags are omitted).
2. **`--compile-step` hangs (not just slow) on `--architecture transformer`
   at this scale.** 9+ minutes with zero steps logged and no crash --
   consistent with `docs/rtx3060_windows_setup.md` section 8's known-gaps
   note that `--compile-step` is validated only for the recurrent mixers,
   but this is worse than "unvalidated": it didn't complete at all in a
   reasonable window. Dropped for this run entirely; since plain transformer
   has no recurrence loop to fuse, the benefit was expected to be small
   anyway (per this doc's original text).
3. **Initial loss is ~550, not ~10 -- this is normal for this specific
   architecture, not a bug.** Verified via an `lr=0` diagnostic (weights
   never update) that this is a pure initialization artifact, present at
   `sequence_length=256` too (so not length-dependent): `MatchedTransformerLM`
   (`reference/hz0a_matched_transformer.py`) has no explicit position
   embeddings and no depth-scaled residual initialization across its 31
   layers, so the residual stream is large from the first forward pass.
   `assert_finite` never trips (loss stays large-but-finite, never NaN/Inf),
   and real training recovers fine: a 200-step diagnostic at `--max-lr 1e-4
   --warmup-steps 50` on this exact corpus took loss from ~550 to ~48. Don't
   mistake a huge loss in the first ~50-100 steps for a diverged run --
   check whether it's still finite and trending down, not its absolute
   value, until warmup completes.

With all three dropped, real full-`sequence_length=1024` self-attention
(quadratic memory in sequence length, unlike the chunked recurrent-mixer
runs) needed fresh VRAM tuning at this scale (768 dim / 31 layers / 12
heads): `--batch-size 8` OOM'd (10.96GB allocated, tried to allocate 768MB
more); `--batch-size 4` fit comfortably at **8.85GB peak**, safely under the
~11.5-12GB WDDM danger zone. **`--batch-size 4`, no `--chunk-length`/
`--truncate-backward`/`--compile-step`, is what the real-run command below
now reflects.**

## What's different from every prior Windows run on this machine

Every run in `docs/rtx3060_windows_setup.md` (sections 5-5f) used
`data/packed/stage2_100m_train_seq256.jsonl` at `--sequence-length 256`. This
run uses a different file at 4x the sequence length:

| | Prior runs | This run |
| --- | --- | --- |
| Data | `data/packed/stage2_100m_train_seq256.jsonl` | `data/packed/hz0g_g1_100m_train.jsonl` |
| Validation | `data/packed/repro_256_val.jsonl` | `data/packed/repro_1024_val.jsonl` |
| `--sequence-length` | 256 | 1024 |
| Architecture | `hybrid` (various mixers) or `transformer` | `transformer` only -- this run's whole point is the control, not another mixer variant |
| `--d-ff` | 2304 (hybrid) / 2944 (transformer) | 2944 (transformer, same param-matching as before -- `dim`/`layers`/`heads` unchanged) |

**Every batch-size/chunk-length number in `docs/rtx3060_windows_setup.md`
was tuned at `sequence_length=256`.** At 1024 (4x), per-sequence activation
memory during backward scales up correspondingly -- **do not reuse those
batch sizes**. Start conservative and re-sweep; section 5b's own VRAM-ceiling
behavior (WDDM silently paging into shared memory past ~11.5-12GB, causing a
3-10x throughput collapse rather than a clean OOM) applies exactly the same
way here, just at different absolute batch-size numbers.

## Files to copy (not in git -- `data/` is gitignored)

| Path | Size | Needed for |
| --- | --- | --- |
| `data/packed/hz0g_g1_100m_train.jsonl` | 508 MB | training data (required) |
| `data/packed/repro_1024_val.jsonl` | 2.2 MB | fixed validation set (required) |

Everything else (`reference/hz0a_torch_model.py`,
`scripts/hz0a_torch_stage2_runner.py`) is already in git if this clone is
up to date with `main`.

## Smoke test first

Same discipline as the main setup doc -- verify the environment and data
path before spending real GPU time:

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/hz0g_g1_100m_train.jsonl \
  --validation-data data/packed/repro_1024_val.jsonl \
  --run-dir outputs/smoke_test_g1_transformer --target-tokens 8192 --batch-size 2 \
  --sequence-length 1024 --chunk-length 128 --truncate-backward \
  --gradient-accumulation-chunks 2 --checkpoint-interval 3 --validation-interval 3 \
  --vocab-size 24576 --dim 64 --layers 6 --heads 4 --d-ff 128 \
  --architecture transformer --device cuda --dtype float32 --seed 7
```

Confirm `"budget_complete": true` before moving on.

## Real run

**Superseded by the correction section above** -- do not use
`--chunk-length`/`--truncate-backward`/`--compile-step` here, and do not
reuse the hybrid-mixer batch-size precedents (they're chunked-recurrence
numbers, not full-self-attention ones). What actually ran, after re-tuning
`--batch-size` against real full-`sequence_length=1024` attention memory
(`--batch-size 8` OOM'd at 10.96GB; `--batch-size 4` fit at 8.85GB peak):

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/hz0g_g1_100m_train.jsonl \
  --validation-data data/packed/repro_1024_val.jsonl \
  --run-dir outputs/rtx3060_g1_matched_transformer --target-tokens 100000000 \
  --batch-size 4 --sequence-length 1024 \
  --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 32 --seed 7 \
  --milestone-tokens 10000000,25000000,50000000,75000000,100000000 \
  --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2944 \
  --architecture transformer --device cuda --dtype bfloat16
```

`--gradient-accumulation-chunks` is omitted deliberately: it only has an
effect inside the `--truncate-backward` loop (see
`scripts/hz0a_torch_stage2_runner.py` around line 419), which this run
doesn't use -- `effective_batch_tokens` here is just `--batch-size *
--sequence-length` = 4 * 1024 = 4096 tokens/step -- larger than G1's own
`batch_size=12 * chunk_length=128 = 1536` per-microbatch shape, but exact
matching wasn't pursued given the VRAM ceiling already dictated
`--batch-size 4` here.

If `nvidia-smi` shows meaningful headroom (8.85GB peak leaves ~3GB before
the ~11.5-12GB WDDM danger zone from section 5b), `--batch-size` could be
pushed to 5 or 6 -- not attempted here since 4 already gave comfortable
margin and every batch-size probe costs several minutes of compile/OOM-retry
time on this card.

## Sending results back

Once this finishes (or at any milestone you want to check in on), run the
full-holdout sweep the same way the Mac side's comparisons were built, then
report `best_validation_loss`, the milestone-by-milestone validation losses,
and `tokens_per_second` back via the file-drop relay (`~/hz0a_transfer/`) --
see the `windows-transfer-relay` reference note for the exact protocol
(outbox/inbox, `X-Auth` header, `https://192.168.68.85:8899`). **As of
2026-08-07, that relay server was not reachable from the Mac side
(connection refused) -- confirm `serve.py` is running on the Mac before
relying on it; if it's down, drop results in `~/hz0a_transfer/outbox/` on
this machine's own filesystem, or wait for it to come back up.**

## What this run answers, and what it doesn't

A real, genuine "does the corrected `gdn2_fix` backbone beat a matched
Transformer on G1's actual corpus" comparison **requires this run to finish
and be compared against G1's own final held-out loss on the same eval set**
(full-holdout, not G1's periodic in-loop 529-sequence check -- run the same
full-holdout sweep script against both checkpoints once both exist). Until
then, this run alone establishes what a plain Transformer achieves on this
specific corpus/seq-len/budget -- useful on its own, but the actual
architecture comparison isn't real until both sides are measured the same
way.
