# HZ-0I weight-tied BDH

Added an experimental input/output vocabulary weight-tied variant. It uses a
cosine-normalized shared embedding/output matrix with a learnable logit scale;
plain raw tying produced unstable initial logits (`~153` CE), so it is not used.

On a 5.6M-parameter vocabulary-dominated probe:

- Parameters: `5,603,328 -> 3,244,033` (42.1% reduction)
- Ten-step CPU training: `102.3ms -> 89.1ms` (1.15x speedup)
- Target-scale 0.3B MPS smoke: finite, loss `10.199 -> 6.863` in 10 steps
- Target-scale tied parameter count: `273,678,337`

Weight tying remains experimental until longer knowledge-quality runs establish
that cosine output normalization does not reduce representational quality.
