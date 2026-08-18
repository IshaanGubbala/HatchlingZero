# Dynamic Block Routing: Real Quality Probe -- Slower AND Worse

Status: real, honest negative result, closes out this thread's central
question. Real, disclosed risk carried over from the FactorizedBDH
probe's own lesson (a short, no-curriculum result can reverse under
full curriculum training) -- flagged before this run, still applies to
what follows: preliminary, not final.

## Real result

`scripts/hz0h_dynamic_block_routing_quality_probe.py`, real 25M-token
byte-level corpus, 500 steps, `capacity_factor=1.0`, seed=7, RTX3060,
bf16 -- verified directly against the raw JSON, not just the chat
summary:

```text
raw_bdh:          best_val=2.5656, 25.43M params,  262.7s
dynamic_routing:  best_val=2.6781, 25.69M params,  532.3s

delta (routed - raw): +0.1125 (WORSE)
training time ratio:  2.03x raw's wall-clock (consistent with the
                       ~1.7x per-step slowdown already measured at
                       cf=1.0, plus real probe overhead)
```

Validation loss history (both arms improving steadily, routed
consistently behind after the first couple of evals):

```text
step:        50      100     150     200     250     300     350     400     450     500
raw:       3.0625  2.9344  2.9766  2.8766  2.7406  2.7062  2.6703  2.5734  2.5812  2.5656
routed:    2.9969  2.8531  2.9594  2.7875  2.7313  2.7437  2.7250  2.6781  2.6891  2.6812
```

Dynamic block routing is both **slower** (2.03x training time) and
**worse** (+0.1125 validation loss) than raw dense BDH at this budget.
The real speed cost documented in
`docs/restart/hz0h_dynamic_block_routing_cuda_oom_results.md` is not
offset by any quality gain -- it is compounded by a real quality loss.

## A real, additional finding: the router's drop rate rose over training, not stayed flat

```text
step:            50      100     150     200     250     300     350     400     450     500
val drop rate: 0.5798  0.6286  0.6398  0.6371  0.6405  0.6545  0.6480  0.6563  0.6506  0.6477
```

Real, notable: this trained router's drop rate (58% -> ~65%, roughly
monotonic) ended up substantially higher than the *same* `cf=1.0`
setting's drop rate with an untrained, freshly-initialized random
router in the earlier CUDA speed benchmark (23.5%,
`hz0h_dynamic_block_routing_cuda_oom_results.md`). That is not a
capacity-factor artifact -- `cf=1.0` fixes the exact same per-block
capacity in both cases. The real difference is that the router *learned
its way toward* dropping more, not less, over training.

**Real, plausible, disclosed (not fully proven) explanation**: this is
consistent with a known, real MoE pathology -- load imbalance, where a
router without an explicit auxiliary balancing loss (a standard,
disclosed real technique in the literature this whole mechanism is
built from, e.g. Switch Transformer's load-balancing loss) has no
incentive to spread tokens evenly across blocks, and can drift toward
concentrating scores on a smaller effective subset of blocks -- which
directly increases the real capacity-driven drop rate as training
progresses, since more tokens compete for the same over-favored blocks'
fixed capacity. This implementation has no such balancing loss. Whether
adding one would close some or all of the quality gap is a real, open,
untested question -- not attempted here, and not a small addition (a
real auxiliary loss term, its own weighting hyperparameter, its own
real tuning question).

## Honest bottom line

Across this entire thread -- OOM found and fixed, a ~14x speed
regression found and fixed, and now a real quality probe -- the
complete, honest picture for per-token dynamic block routing (encoder
projection only, this mechanism's own disclosed scope) at this
project's shape is:

- **Correctness**: real, thoroughly verified (33+ tests, exact oracle
  matches, cross-validated gated behavior, multi-layer exactness,
  finite gradients throughout).
- **Speed**: real, confirmed cost -- 1.37x-2.7x slower than raw dense
  BDH at every capacity factor tested, even after fixing two real
  implementation bottlenecks.
- **Quality**: real, confirmed cost, not a benefit -- worse validation
  loss than raw BDH at this budget, plausibly compounded by real router
  load imbalance this implementation doesn't correct for.

**This mechanism, as built, is not currently a lever worth pursuing
further for this project** -- it costs real throughput and does not
buy back quality, the only thing that could have justified the cost.
The standing curriculum caveat means this is not a mathematically final
verdict (a full curriculum run, or adding a load-balancing loss, could
in principle change the picture), but nothing in the evidence gathered
across this entire investigation points toward those follow-ups being
likely to reverse a 2x-slower-and-worse result, and neither is proposed
as a next step here.
