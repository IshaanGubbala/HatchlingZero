# Next efficiency-candidate screen: GDN-2 Fix kernel prerequisite

Date: 2026-08-15. This is a feasibility decision after the matched RTX 3060
failure of BlockBDH, Direct Split-V, learned gating, and `chunk_gla`. It is not
an architecture promotion or an efficiency result.

## Why it is the only non-closed near-term candidate

A runner-compatible parameter match to the 25,343,488-parameter RoPE
Transformer is:

```bash
--vocab-size 256 --dim 512 --layers 6 --heads 4 --d-ff 1621 \
--architecture hybrid --mixer gdn2_fix --rope
```

The runner's fixed attention index makes this five GDN-2-Fix blocks plus one
attention block. It has 25,342,977 trainable parameters (ratio 0.999980), so
it satisfies the symmetric match constraint without padding or hidden smaller
capacity. Its recurrent state is materially smaller than BDH's 2,048-column
latent.

## Kernel blocker

Do **not** use `--fla-recurrence` for this arm. Original GDN-2 is affine and
can be expressed by the GLA scan. GDN-2 Fix updates
`S = decay(S) + (write*v - decay(S)@(erase*key))*key^T`; its key-dependent
state correction is not that affine recurrence. The runner intentionally
rejects `--mixer gdn2_fix --fla-recurrence`. Replacing it with old GDN-2 would
change the quality-supported architecture and cannot be presented as a
successor.

The existing quality signal for GDN-2 Fix is from a different 301M setup, not
this 25M byte corpus/B12×T256 target. Existing compiled sequential recurrence
also has no matched target measurement. Therefore it is not justified to
launch a full quality run or make an efficiency prediction.

## Required predecessor gate

Before any 25M GDN-2-Fix training or superiority experiment, implement a
strict CUDA delta-rule kernel (forward/backward) that exactly represents the
fixed update, and provide a raw-vs-kernel B12×T256 BF16 preflight with:

1. logits/loss/parameter-gradient tolerances and finite steps;
2. native `torch.cuda.max_memory_allocated` for candidate and the matched
   Transformer under identical optimizer/compile policy;
3. parameter ratio 0.9901–1.01;
4. ≥1.30× Transformer throughput and ≤0.70× Transformer peak RAM.

If that screen fails, close GDN-2 Fix as a training-efficiency candidate before
any long quality run. If it passes, run preregistered multi-seed byte-corpus
quality and frozen external evaluation. This preserves the frozen canonical
GDN-2 work: the proposed arm is explicitly the `gdn2_fix` derivative.
