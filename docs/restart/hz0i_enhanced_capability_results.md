# HZ-0I enhanced BDH capability composition

The BDH shell now has an explicit experimental composition that retains the
project premise: persistent BDH state plus optional explicit memory, triggered
attention, fast weights, and sparse MoE capacity. All components return
diagnostics (memory read rate, trigger rate, fast delta norm, expert counts).

## Structured task

At 300 steps, base BDH reached loss `0.0762`; all optional components reached
`0.00034`. Both remained finite.

## Real-corpus probe

At 100 steps on the audited packed corpus, base BDH reached `8.7654` from
`10.0989`; the enhanced composition reached `7.0582` from `10.1042`. Both were
finite. The enhanced model ran at 33,491 tok/s versus 39,791 tok/s for base
(the expected conditional/adapter overhead; this implementation is not yet
kernel-optimized).

This is the first direct evidence that the HZ capability bundle can improve the
BDH backbone on a real-corpus short probe. It is not yet a promotion result:
the probe uses a tiny 16-wide model, one seed, no held-out set, and a dense
attention implementation behind the trigger mask.


## Conditional-compute implementation

The triggered-attention primitive was upgraded from full attention plus masking
to vectorized triggered-query attention: Q is computed only at triggered
positions, while shared K/V are reused. The real-corpus probe remained finite
and improved `10.104 -> 7.058` versus base `10.099 -> 8.765`; throughput was
32,022 versus 38,536 tok/s (about 17% overhead in this tiny unoptimized
Torch path, substantially better than the earlier per-query loop).
