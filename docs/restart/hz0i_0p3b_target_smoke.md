# HZ-0I 0.3B target smoke

The immediate target is now the 0.3B BDH model before any 0.8B scale-up. The
profile (`d_model=768`, multiplier 144, 12 heads, 8 shared-weight layers,
vocab 24,576) contains exactly 292,552,704 BDH parameters.

Real CPU/Torch smokes passed:

- Base BDH forward: `(1,4,24576)` finite logits in 0.30s.
- Enhanced BDH (conditional attention + fast weights + MoE): 304,376,836
  parameters, finite logits in 0.30s; expert counts and trigger diagnostics
  emitted.
- Persistent streaming across chunks `[1,3]`: finite logits and exact state
  allocation of 2.718GB FP32 (1.359GB BF16).

These are bring-up gates, not training or Qwen comparison results. The next
correct step is a real 0.3B training continuation with the audited corpus and
held-out evaluation, while keeping Qwen 0.8B as the external comparison target.


## Real training probe

A real 20-step AdamW continuation on the packed corpus completed at the target
scale: loss `10.0960 -> 8.8468`, 640 tokens, 29.8 tok/s, finite parameters,
and measured peak RSS ~7.4GB. This confirms the 0.3B profile can train on the
current machine, but it is not yet a meaningful pretraining run.


A target-scale 1,000-step real-corpus continuation has now been launched as
`outputs/hz0i_0p3b_train_1000.json` (PID 49809 at launch). It is the first run
intended to produce a checkpoint useful for the Qwen3-0.6B comparison; until it
finishes, no Qwen comparison claim is possible.


A second target-scale run is queued behind the base control: the full BDH +
conditional-attention + fast-weight + MoE composition at the same 0.3B width.
This preserves a direct base-versus-capability-bundle comparison rather than
assuming the tiny-model gain transfers to target scale.
