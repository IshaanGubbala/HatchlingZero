# HZ-0I compact BDH knowledge-density training

The rank-256 factorized+tied 0.3B-profile model completed real adaptive
knowledge-density training on MPS:

- 500 steps / 32,000 tokens: loss `10.682 -> 6.495`, 655.5 tok/s, finite
- Domain counts: general 103, code 92, math 64, JSON 60, docs 77, terminal 104
- 200-step checkpointed continuation: loss `10.643 -> 8.183`, 644.0 tok/s,
  finite parameters, checkpoint saved as `outputs/hz0i_mps_knowledge_200.pt`

The adaptive sampler kept all six knowledge domains active while shifting
probability toward harder domains. These are still short continuation runs, not
pretraining-scale quality evidence, but they establish that the compact BDH can
train on a heterogeneous knowledge mixture at high MPS throughput.


Held-out per-domain evaluation of the 200-step checkpoint (16 sequences/domain)
measured CE: general `7.933`, code `7.531`, math `7.895`, JSON `6.907`, docs
`8.039`, terminal `7.474`. This is a diagnostic snapshot, not a mature
quality result; docs/general remain harder and should receive adaptive replay.


Batch scaling on MPS improved throughput further at sequence length 64:

- Batch 1: `655.5 tok/s` (500-step run)
- Batch 2: `722.5 tok/s` (200-step run)
- Batch 4: `796.5 tok/s` (100-step run)

All runs remained finite and sampled every knowledge domain. Batch 4 ended at
CE `8.366`; this is not directly comparable to the longer batch-1 run, but it
confirms the compact architecture benefits from device batching.


The MPS knowledge runner now saves optimizer state and supports resume. A 100-step
batch-2 continuation from the 200-step checkpoint resumed at CE `7.567` and
ended `7.520`, remaining finite at `691.6 tok/s`. This establishes a practical
continual-pretraining loop rather than isolated probes.


Adaptive sampler policy is now checkpointed alongside model and optimizer.
Resume restores domain weights, loss EMAs, and RNG state, preventing a
continual run from silently losing its learned knowledge-mixture policy.


The MPS runner now applies configurable gradient clipping (default norm 1.0)
for stable long continual runs. A 20-step batch-2 smoke was finite with loss
`10.727 -> 9.427`.


A longer 5,000-step batch-4 MPS continual-pretraining run is currently running
(`outputs/hz0i_mps_knowledge_5000.json`), with held-out per-domain evaluation
queued automatically after checkpoint completion.


The checkpointed batch-4 run completed 1,000 steps / 256,000 tokens:

- Loss `10.738 -> 7.040`
- 464.9 tok/s (long-run MPS throughput)
- Finite parameters
- Domain counts: general 670, code 722, math 647, JSON 644, docs 663,
  terminal 654

Held-out CE improved versus the 200-step checkpoint:

- General `6.806`
- Code `6.524`
- Math `6.705`
- JSON `5.610`
- Docs `7.002`
- Terminal `6.418`

The long run includes periodic `.latest.pt` checkpoints and preserved adaptive
sampler state.


At step 1,000 the adaptive policy remained intentionally balanced: weights were
within `0.163–0.168`, with JSON the easiest domain (EMA CE `6.376`) and docs
the hardest (`6.661`). The 5% floor prevented any domain collapse while still
allowing hardness-based movement.
