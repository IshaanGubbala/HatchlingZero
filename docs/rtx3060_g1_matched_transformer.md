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

Starting point only -- **re-sweep `--batch-size` against this card's actual
VRAM** the same way section 5b/5f did, watching `peak_memory_bytes` in
`torch_stage2_memory.jsonl`, not just whether the run completes. A
conservative starting guess: take the closest known-good config at
`sequence_length=256` (transformer hasn't been throughput-tuned on this
machine yet, only `hybrid`/`gdn2`/`gdn3`/`gdn2_fix` have -- see section 5b's
`--batch-size 256 --chunk-length 8` and 5f's `--batch-size 288
--chunk-length 8` for the hybrid precedent) and divide batch size by
roughly 4x for the 4x longer sequence, then adjust from there:

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/hz0g_g1_100m_train.jsonl \
  --validation-data data/packed/repro_1024_val.jsonl \
  --run-dir outputs/rtx3060_g1_matched_transformer --target-tokens 100000000 \
  --batch-size 64 --sequence-length 1024 --chunk-length 8 --truncate-backward \
  --gradient-accumulation-chunks 4 --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 32 --seed 7 --compile-step \
  --milestone-tokens 10000000,25000000,50000000,75000000,100000000 \
  --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2944 \
  --architecture transformer --device cuda --dtype bfloat16
```

If `nvidia-smi` shows meaningful headroom during the smoke test, increase
`--batch-size` (or decrease `--gradient-accumulation-chunks` to keep
`effective_batch_tokens` roughly matched to G1's own `batch_size=12 *
chunk_length=128 = 1536` per-microbatch shape -- see
`outputs/hz0g_g1_gdn2_fix_301m/config_snapshot.json` for G1's exact numbers
if you want to match `effective_batch_tokens` precisely rather than just
being in the same ballpark). If you hit OOM or the WDDM slowdown described
above, back off the same way section 5b describes.

`--compile-step` is unvalidated on this hardware for the plain
`transformer` architecture specifically (validated so far only for the
recurrent mixers, per section 8's known-gaps list) -- if it causes problems,
drop it; `--architecture transformer` has no recurrence loop to compile
anyway, so the benefit is smaller than for `hybrid` runs regardless.

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
