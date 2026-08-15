# BlockBDH 100K-token packed-data pilot

Date: 2026-08-14. This follows the short active-fraction sweep in
`hz0h_blocksparse_packed_data_preflight_results.md`. It is a candidate-rejection
pilot, **not** a Transformer target-gate, trained-checkpoint quality study, or
peak-RAM result.

## Matched configuration

Dense BDH and experimental BlockBDH used the same packed byte train/validation
files, seed 7, MPS, BF16, eager execution, AdamW/cosine schedule (10 warmup
steps), batch 1 x 256 tokens, D=512, 8 recurrent levels, 8 heads, multiplier
32, and 100,096 actual tokens / 391 optimizer steps. Both have 25,427,968
parameters. BlockBDH used a 16-column block and 12.5% active fraction (16 of
128 blocks). Validation used a fixed four-sequence held-out batch at steps 100,
200,300,391.

| Arm | train seconds | tokens/s | speed ratio vs Transformer | best validation CE | MPS allocator snapshot |
|---|---:|---:|---:|---:|---:|
| Dense BDH | 114.829 | 871.70 | 0.189x | 2.828125 | 208,375,552 B |
| BlockBDH 12.5% | 32.209 | 3,107.66 | **0.674x** | 2.859375 | 206,569,984 B |
| Matched RoPE Transformer | 21.696 | 4,613.67 | 1.000x | 2.856395 | 199,647,488 B |

The Transformer has 25,343,488 parameters versus BlockBDH's 25,427,968
(parameter ratio 1.0033), and used the same packed files, BF16/MPS/eager
policy, batch tokens, token budget, seed, schedule, and held-out batch.
BlockBDH validation CE at the four checkpoints was 3.09375, 2.9375, 2.890625,
and 2.859375; dense BDH's was 3.078125, 2.890625, 2.953125, and 2.828125.
The final BlockBDH/Transformer difference (0.00298) on this tiny fixed
validation batch and only 100K tokens is not a quality-equivalence test. It
is enough to show neither loss diverged or became non-finite in this preflight.

## Fair-control diagnostic result

The dense-BDH speedup does **not** clear the actual objective at this shape:
BlockBDH achieved only 0.674x the matched Transformer's training throughput
and took 1.485x its wall-clock time. Its sampled MPS allocator ratio was
1.035x the Transformer, not <=0.70. Thus the current 12.5%-active derivative
fails both requested training-system thresholds in this MPS diagnostic.

## Depth and active-fraction boundary sweep

The same protocol was then used to test whether reducing recurrent work while
keeping the shared-weight parameter count fixed could clear the actual speed
threshold. Every BlockBDH row below has 25,427,968 parameters; the Transformer
has 25,343,488. These are short-run diagnostic values only.

| BlockBDH depth | active fraction | tok/s | speed / Transformer | best validation CE | allocator ratio / Transformer |
|---:|---:|---:|---:|---:|---:|
| 8 | 12.5% | 3,107.66 | 0.674x | 2.859375 | 1.035x |
| 4 | 12.5% | 4,106.65 | 0.890x | 2.593750 | 1.037x |
| 2 | 12.5% | 4,976.77 | 1.079x | 2.578125 | 1.038x |
| 1 | 12.5% | 5,533.00 | 1.199x | 2.562500 | 1.045x |
| 1 | 6.25% | 5,609.95 | **1.216x** | 2.578125 | 1.028x |
| target | n/a | >=5,997.77 | >=1.300x | quality-matched | <=0.700x |

The 6.25%-active depth-1 point is the fastest tested configuration, but it
still misses the speed target and has no sampled-memory win. Going from 12.5%
to 6.25% active at depth 1 gained only 1.4%, indicating substantial fixed
router/embedding/optimizer or remaining-attention cost. Do not infer that
further naive active-fraction reduction will reach 1.30x; any continuation
needs a concrete kernel or architectural change and a quality test.

The apparently favorable short fixed-batch CE values at shallow depth are not
a license to choose depth after observing validation: 100K tokens, one seed,
and four sequences are insufficient for a trained-quality conclusion.

## Cheap-proxy router sweep

The original activation router materializes a full latent solely to score
blocks. `router_method=cheap_proxy` instead pools the input and scores
encoder-block prototypes in O(D*N). It is an explicit routing-policy
experimental derivative; it is not evidence that the original activation
router has become cheaper.

| depth | active fraction | router | tok/s | speed / Transformer | best validation CE | allocator ratio |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6.25% | activation | 5,609.95 | 1.216x | 2.578125 | 1.028x |
| 1 | 6.25% | cheap_proxy | 5,734.50 | 1.243x | 2.546875 | 1.028x |
| 1 | 3.125% | cheap_proxy | 5,768.39 | **1.250x** | 2.578125 | 1.029x |
| target | n/a | n/a | >=5,997.77 | >=1.300x | quality-matched | <=0.700x |

The second cheap-proxy reduction gained only 0.6%; this is a plateau, not a
credible extrapolation to the 1.30x target. It also became highly sticky:
6.25% had 20 distinct route sets but 0.928 exact-repeat fraction; 3.125% had
11 sets and 0.962 exact-repeat fraction. Its apparent short fixed-batch CE is
not quality evidence. Do not use it for a full run absent a mechanism that
trains/regularizes the router and a pre-registered quality test.

## Fair compiler-policy sweep

Both the depth-1/6.25%-active BlockBDH arm and the matched Transformer arm were
then run with the same BF16/MPS/data/batch/token settings and `torch.compile`
policy. Reported end-to-end times include graph compilation startup, which is
appropriate for this short pilot; late-run steady intervals were examined
separately as a diagnostic.

| policy | BlockBDH tok/s | Transformer tok/s | overall speed ratio | Block/Transformer allocator snapshot |
|---|---:|---:|---:|---:|
| eager | 5,609.95 | 4,613.67 | 1.216x | 1.028x |
| compile `default` | 4,689.47 | 3,861.83 | 1.214x | 1.140x |
| compile `reduce-overhead` | 5,064.05 | 4,167.91 | 1.215x | 1.140x |

The default compiled late interval was 5,834 versus 4,727 tok/s (1.234x),
and reduce-overhead's was 5,781 versus 5,043 (1.146x). Neither comes close to
the required 1.30x. The validation losses remained finite and near their
uncompiled short-run values, but this is not quality evidence. Conclusion:
compilation is available and must remain fair, but it is not the intervention
that closes the current BlockBDH/Transformer gap on this backend.

## Router telemetry

The BlockBDH runner logged 79 distinct selected-block sets over 391 steps.
Mean consecutive-route Jaccard overlap was 0.916 (range 0.333--1.000). The
last 91 steps had 13 distinct sets and mean overlap 0.942. Thus routing became
sticky but was not a total immediate single-set lock-in; a longer run must
report these telemetry fields and may test the separately labelled balance-loss
ablation if actual quality/reassignment evidence shows collapse.

## Decision

Keep 12.5%-active BlockBDH as the first CUDA full-pilot candidate: it is well
above the 1.30x *dense-BDH* preflight speed threshold and did not fail the
short-loss/router screen. This says nothing about the requested target because:

- MPS sampled allocator memory is not a native peak metric; even its
  non-authoritative snapshot is higher than the Transformer's, not 30% lower;
- MPS is diagnostic rather than the target CUDA platform, and the short run
  cannot establish sustained training behavior;
- 100K/25M tokens, one seed, and four validation sequences cannot establish
  quality compatibility;
- CUDA can change `index_select`/GEMM crossover behavior materially.

Only the full matched CUDA three-arm protocol and
`scripts/hz0h_training_target_gate.py` can decide the training RAM/speed
objective.
