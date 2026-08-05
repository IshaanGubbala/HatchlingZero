# HZ-0A/B/C/D Joint Evaluation

Date: 2026-08-05. Runner: `scripts/hz0abcd_joint_evaluation.py`.

The bounded evaluation composes the real frozen 301M HZ-0A checkpoint, HZ-0B
latent write/read memory, the HZ-0C deterministic 15% trigger schedule, and
the HZ-0D delta-prediction fast-weight update. It uses eight real corpus
sequences for adaptation and eight disjoint sequences for evaluation, across
seeds 555, 556, and 557. The update is applied after the first evaluation
call, then measured on the next call, preventing feedback from affecting the
same call that produced it.

| Seed | Baseline loss | Adapted loss | Delta (baseline - adapted) | Update norm | Finite |
|---:|---:|---:|---:|---:|:---:|
| 555 | 2.966108 | 2.991865 | -0.025757 | 7.359731 | yes |
| 556 | 3.099031 | 3.096072 | 0.002959 | 7.416341 | yes |
| 557 | 2.983381 | 2.980942 | 0.002439 | 7.500338 | yes |

Summary: mean loss delta `-0.006787` nats, two of three seeds improved, all
updates applied, and all logits/metrics were finite. The negative mean is an
honest mixed result: this proves the full composition executes and remains
bounded, but it does not establish a reliable quality gain from one short
adaptation window. Repeating the exact command produced byte-identical JSON.
