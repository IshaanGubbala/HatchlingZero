# HZ-0F F5: Fallback Isolation at the Full 3-Layer Joint Scope

Date: 2026-08-06. Direct validation of F4's `broad_only` finding at the
full 3-layer joint scope (27, 28, 30 trained together, matching E1's
real contract), per the explicit request: validate before locking it as
the HZ-0F default, and before combining with any other fix, so a
compound result can't obscure whether the fallback fix itself scales.

`reference/hz0e_f4_fallback_isolation.py::train_joint_moe_with_fallback_policy`
/ `train_joint_dense_baseline_with_general_eval` / `evaluate_joint_arm`.
Same three policies as F4 (`current`, `frozen`, `broad_only`), same
full `50/50/50`-step protocol, applied to all 3 target layers
simultaneously via one shared gradient step per batch (matching
`run_joint_multilayer_curriculum`'s own approach) -- not 3 independent
single-layer runs.

## Result: the single-layer win does NOT reliably generalize

| Seed | Arm | Domain win | General/OOD gap (MoE - dense) |
| --- | --- | :---: | ---: |
| 0 | current | 4/5 | +0.0233 |
| 0 | frozen | 4/5 | +0.0402 |
| 0 | broad_only | 4/5 | +0.0253 (WORSE than current) |
| 1 | current | 4/5 | +0.0193 |
| 1 | frozen | 4/5 | +0.0365 |
| 1 | broad_only | 4/5 | +0.0230 (WORSE than current) |
| 2 | current | 4/5 | +0.0214 |
| 2 | frozen | 4/5 | +0.0369 |
| 2 | broad_only | 4/5 | +0.0164 (better than current) |

**This is a real, honest, mixed result, reported exactly as measured.**
Two of F4's findings DO generalize cleanly to joint scope:

- **`frozen` is consistently worse than `current`** in every seed
  (`+0.0365` to `+0.0402` vs `+0.0193` to `+0.0233`) -- the same
  direction as the single-layer result, now confirmed at joint scope
  too.
- **Domain win count is unaffected** -- `4/5` in every arm, every seed,
  at joint scope (no degradation from any fallback policy).

**One finding does NOT generalize**: `broad_only` does NOT reliably
reduce the OOD gap at joint scope, let alone flip it net-positive the
way it did for layer 27 alone. It is WORSE than `current` in 2 of 3
seeds (`0.0253 > 0.0233`, `0.0230 > 0.0193`) and only better in 1 of 3
(`0.0164 < 0.0214`). None of the 3 seeds come close to F4's single-layer
result (`-0.001` to `-0.010`, a net MoE advantage).

## Ruled out: overflow-rate dilution across layers

One plausible explanation was checked directly and ruled out: the
original E6 integration measurement found layer 30 had `0%` overflow
under its own (different, untrained, differently-scaled) initial
weights -- if that held here, applying `broad_only` uniformly across a
low-overflow layer would dilute its effect. Checked directly on the
`current`-policy joint-trained model (seed 0): overflow is substantial
and comparable across ALL 3 layers at this training point (`46.1%`,
`49.6%`, `47.3%` for layers 27/28/30 respectively) -- not the dilution
this hypothesis predicted. The real explanation for why `broad_only`'s
benefit does not compose cleanly across layers is not identified here.

## Honest verdict: broad-only is NOT locked as the HZ-0F default

Per the stated gating condition ("lock it as the HZ-0F default if it
retains both ID and OOD wins" at joint scope) -- **it does not**, and is
therefore NOT adopted as a default here. The single-layer result
(F4) remains real, reproducible, and worth keeping as a documented
finding, but it does not extend to the full 3-layer contract without
further work. Per the same reasoning that motivated running this
validation before combining fixes: proceeding to add counterfactual-
utility routing on top of an unconfirmed fallback fix would risk
exactly the confound this step was designed to avoid, so that step is
NOT taken here either.

## Honest scope and open questions

Not identified: WHY the joint-scope composition differs from the
single-layer result -- candidate explanations not yet tested include
cross-layer interaction through the residual stream (layer 27's
residual feeds layer 28's input, etc.), the joint replay step updating
all 3 layers' fallbacks from the SAME general batch simultaneously
(versus F4's fully independent single-layer replay), or per-layer
differences in how much each layer's fallback quality actually matters
to the final logits. Any of these would be a real, testable follow-up,
not assumed here.
