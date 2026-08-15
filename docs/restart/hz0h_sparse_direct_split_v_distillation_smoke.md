# Frozen-Transformer distillation smoke for sparse Direct Split-V: negative

Date: 2026-08-14. This was a 100,096-token MPS diagnostic, not a trained
comparison or systems measurement.

A frozen, locally available 25M matched RoPE Transformer checkpoint
`outputs/hz0h_phase_f_matched_transformer/seed7_from_pi/hz0h_phase_f_transformer_seed7_checkpoint.pt`
was validated first on the fixed 32-sequence byte batch: CE **1.74239**. A
randomly initialized depth-4 Direct-Split-V BlockBDH student (3.125% active,
cheap-proxy router) then trained for 391 steps using

```text
0.5 * next-token CE
+ 0.5 * T^2 * mean_{batch,time} KL(student/T || teacher/T), T=2
```

The token/time mean is essential. An initial scratch attempt used PyTorch
`batchmean`, which divides only by batch and inflated the sequence KL about
256x; it was discarded before evaluation. The corrected run logged ordinary
CE/KL at steps 100/200/300 as 6.594/4.867, 2.422/2.030, and 3.203/1.882. Its
final validation CE was **3.09375**, teacher KL 2.37739, far worse than the
frozen teacher and not an improvement over the preceding sparse pilots.

Decision: do not add distillation to a full sparse training run on this
recipe. A future distillation retry needs a specified teacher/student
alignment objective, a longer dense/soft-gate stage, and a controlled dense
student baseline; simply applying logits KL from random hard-sparse training
is not a quality fix.
