# HZ-0C C6 Conditional Attention Results

Date: 2026-08-03. This is the first evaluation that reruns the frozen
HZ-0A graph with conditional attention rather than scoring trigger
positions in isolation.

The six existing attention layers use the same causal masked-attention
implementation. Each policy is evaluated on 32 real corpus sequences
at sequence length 128.

The reported comparison aggregates seeds `1, 2, 3, 4, 5`; each seed
uses a shuffled disjoint 128-sequence training split and 32-sequence
held-out evaluation split.

| Policy | Anchor rate | Loss | Perplexity |
| --- | ---: | ---: | ---: |
| No anchor | 0.0% | 2.5514 +/- 0.1050 | 12.9250 |
| Fixed periodic, exact-rate | 15.0% | 2.5367 +/- 0.0931 | 12.7380 |
| Random matched | 14.9% | 2.5374 +/- 0.1042 | 12.7460 |
| State novelty | 14.9% | 2.5391 +/- 0.1049 | 12.7680 |
| Offline token-loss teacher | 14.9% | 2.5289 +/- 0.1019 | 12.6380 |
| Learned inference-safe controller (projection-aware, 4x positive weighting) | 15.0% | **2.5203 +/- 0.0876** | **12.4328** |
| Full attention | 100.0% | 2.3953 +/- 0.0864 | 11.0790 |

## Fresh Reproducibility Check

Command: `PYTHONPATH=. .venv/bin/python scripts/hz0c_c6_conditional_attention_eval.py --seed 555`

The post-cleanup single-seed run preserves the ordering:

| Policy | Anchor rate | Loss | Perplexity |
| --- | ---: | ---: | ---: |
| No anchor | 0.0000 | 2.5883 | 13.3072 |
| Fixed periodic | 0.1504 | 2.5734 | 13.1100 |
| Random matched | 0.1494 | 2.5745 | 13.1242 |
| State novelty | 0.1494 | 2.5781 | 13.1718 |
| Offline token-loss teacher | 0.1494 | 2.5648 | 12.9977 |
| Learned controller | 0.1504 | **2.5533** | **12.8490** |
| Full attention | 1.0000 | 2.4319 | 11.3809 |

This is a sanity check, not a replacement for the five-seed aggregate. The
learned controller improves fixed periodic by `0.0185` loss at matched cost.

## Finding

The projection-aware learned inference-safe controller is the strongest sparse
policy across five held-out splits, improving loss over exact-rate fixed
periodic by `0.0164` while staying under the hard
15% cap. The previous C7 event-recall score was not a
sufficient optimization target: state novelty reached `0.311` event
recall but produced no LM-loss improvement. The controller is trained
from the offline teacher on a disjoint split and evaluated without
future-token access.

The evaluator now collects the six-layer Q/K/V demand features during the
same frozen-backbone pass that produces the hidden states, rather than
replaying all 30 blocks a second time. This preserves the demand values and
made the five-seed replay complete reliably instead of leaving long-lived
MLX processes.

This closes the graph-level and learned-controller execution gaps.
Multi-seed LM-loss stability is now verified; the memory-preservation
audit is covered by the upstream HZ-0B invariant suite: 50 memory,
read/write, real-integration, and serialization tests pass. The C6
chunked audit (`scripts/hz0c_c6_chunked_memory_audit.py`) also passes on
1,024 tokens with 16-token chunks: output error `0.0`, every memory
state-field error `0.0`, and finite outputs.

## Historical Three-Seed Ablation (2026-08-03)

The controller was then given four additional causal features from the actual
frozen attention projections at the six anchor layers: Q, K, and V energy plus
QKV variance. No attention matrix, label, or future token is exposed. The
three-seed held-out run uses the same disjoint 128/32 split and exact 15%
budget; it beats fixed periodic on every seed:

| Seed | Fixed loss | Projection-aware loss | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 2.6150 | 2.5977 | 0.0172 |
| 2 | 2.5041 | 2.4924 | 0.0116 |
| 3 | 2.5654 | 2.5587 | 0.0067 |
| Mean | 2.5615 | **2.5496** | **0.0118** |

This three-seed table is retained for provenance only. The complete
five-seed projection-aware aggregate at the top of this report is
authoritative and supersedes it.
## Downstream controller generalization attempt (2026-08-03)

A deterministic minibatch variant was tested to address the graph-size failure
of fitting all 128 training sequences at once while avoiding the overfit of a
fixed 16-sequence subset. It was rejected at the authoritative held-out loss
gate. Across seeds 555-557, the minibatch controller scored `2.57390`,
`2.48651`, and `2.48697`, while fixed periodic scored `2.57338`, `2.48492`,
and `2.48614`, respectively. The controller was worse on every seed, so the
retained C6 result and implementation are unchanged.

## Pooled-training transfer check (2026-08-03)

An opt-in `--train-seeds` mode pooled two independent 128-sequence training
splits (555+556) and evaluated on seed 557. The learned controller reached
loss `2.48868` versus fixed periodic `2.48614`, so pooled training did not
transfer in this protocol and was not promoted. The retained single-split
4x-weighted controller remains the authoritative C6 configuration.

## MLP controller transfer check (2026-08-03)

The same held-out pooled-training protocol was run with the opt-in MLP
controller (`--controller mlp`, train seeds 555+556, evaluate seed 557). It
reached loss `2.49165` versus fixed periodic `2.48614`, so the extra model
capacity did not improve transfer and was rejected. The linear controller
remains the promoted policy. A downstream-loss optimization run did not emit a
machine-readable result before termination and is not treated as evidence.

## Bounded causal-teacher screen (2026-08-03)

The causal downstream-benefit teacher was made bounded with an explicit
`--causal-teacher-candidates` limit. Using 8 teacher sequences, 4 candidates
per sequence, and 300 linear distillation steps produced this three-way
held-out transfer screen:

| Train seeds | Eval seed | Fixed loss | Causal-teacher controller | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 555+556 | 557 | 2.48614 | **2.47866** | **0.00747** |
| 555+557 | 556 | 2.48492 | **2.48020** | **0.00472** |
| 556+557 | 555 | 2.57338 | 2.57674 | -0.00336 |
| Mean | | 2.51481 | **2.51187** | **0.00294** |

This is a promising but not yet promoted candidate: it wins two of three
splits, while the five-seed token-loss controller remains authoritative. The
candidate budget is now explicit so future teacher runs cannot silently become
unbounded.

### Hybrid extension across five held-out splits

Blending the causal downstream target with the token-loss target at weight
`0.5` was then evaluated with bounded settings (8 teacher sequences, 4
candidates per sequence, 300 steps). It won all five held-out splits:

| Eval seed | Fixed loss | Hybrid loss | Improvement |
| ---: | ---: | ---: | ---: |
| 555 | 2.57338 | **2.55588** | **0.01749** |
| 556 | 2.48492 | **2.47718** | **0.00775** |
| 557 | 2.48614 | **2.47114** | **0.01500** |
| 558 | 2.51163 | **2.50030** | **0.01133** |
| 559 | 2.45693 | **2.44589** | **0.01104** |
| Mean improvement | | | **0.01252** |

This is now the strongest transfer candidate: exact 15% rate, five wins, and
bounded teacher cost. The original five-seed token-loss result remains
unchanged as historical provenance. The hardened aggregate command has now
completed successfully, so this hybrid is the promoted C6 controller for the
next downstream evaluation.

The safe aggregate command is now
`PYTHONPATH=. .venv/bin/python scripts/hz0c_c6_hybrid_transfer_report.py`.
It runs splits sequentially and terminates the entire child process group on
timeout.
