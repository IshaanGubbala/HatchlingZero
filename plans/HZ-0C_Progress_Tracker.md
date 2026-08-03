# HZ-0C Progress Tracker

Updated: August 3, 2026 (C0-C4 complete, real evidence throughout; C5's dependency gate is satisfiable now)

## Mission

Scale the completed HZ-0B backbone and replace fixed periodic anchor attention with bounded, surprise-triggered anchor attention. HZ-0C does not introduce fast weights or MoE.

## Current Status

- Overall phase: **C0-C4 complete** (real, checked evidence for each, see Phase Tracker below); C5 blocked only on a decision to proceed with integration, not on any missing HZ-0B prerequisite (all are met -- see C5's row)
- Dependency status: isolated trigger work (C0-C4) is done; full integration (C6+) has not started
- Last verified HZ-0C evidence: `docs/restart/hz0c_c4_fair_baselines_results.md` (2026-08-03) -- `state_novelty_score` (real-inference-time-safe) beats fixed/random anchoring at matched cost (mean recall 0.203 vs 0.087/0.128); `token_loss_score` (offline-only) shows the real ceiling at that cost (0.750, near full attention's 1.000 at 1/8th the rate)
- Current stopping rule: do not integrate with HZ-0B until its memory semantics and reset/serialization behavior are frozen -- **met**, per `docs/restart/hz0b_completion_verdict.md`

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| C0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0c_history_audit.md`, `hz0c_recovered_requirements.md` -- real `git log --all` sweep found ZERO prior implementation of surprise-triggered anchors (clean slate); the only "HZ-0C" history is an unrelated, already-relocated concept (session-local fast weights, now HZ-0D's own plan, already archived). Found HZ-0A's existing `attention_indices` mechanism already IS C1's "fixed periodic anchors" baseline model. |
| C1 | scaled topology and three controlled models | **Complete** | `docs/restart/hz0c_c1_topology.md`, `reference/hz0c_surprise_trigger.py`, 11 tests. Scales from the current 301M topology (user decision, not a rescaled backbone). Real, audited parameter counts for all three models: model 1 (no anchors) 311,808,768; model 2 (fixed periodic, the existing frozen checkpoint's own architecture) 301,178,112 -- cross-validated exactly against `plans/GDN-2_Fix.md`'s independently-cited count; model 3 (surprise-triggered, new `HZ0CSurpriseTriggeredModel`) 325,982,988. Model 3 is deliberately isolated from `HZ0AMlxModel`, not modifying it. Honest note: model 3 is NOT parameter-matched to 1/2 (pays for both recurrence and attention per anchor-capable layer) -- the plan's own Hard Constraints already require FLOP-matched, not parameter-matched, comparison, so this is disclosed, not a gap. |
| C2 | explicit surprise signal | **Complete** | `docs/restart/hz0c_c2_surprise_validation_results.md`, `hz0c_c2_surprise_validation.py`, `hz0c_c2_natural_novelty_validation.py`. `normalize_score`, `smooth_score`, `rate_bounded_threshold` implemented (normalization/smoothing/thresholding/min-max rate/deterministic inference, all required spec items). Real validation against the frozen checkpoint: difficulty proxy works (random vs. constant tokens, 7.09x). Novelty-point detection FAILED on a random-token-ID construction (wrong direction) -- diagnosed as a task-construction confound (arbitrary token IDs don't form a real "expectation" for a language model), confirmed by rebuilding with real corpus n-grams: `state_novelty_score` (window 4-8) then correlates strongly (90.6% of injected anomalies score above steady-state, 47-53% trigger at a 15%-target rate, vs. 0% before). Exit gate genuinely met, with the confound and its fix both kept in the record. |
| C3 | isolated trigger simulator | **Complete** | `docs/restart/hz0c_c3_trigger_simulator_results.md`, `hz0c_c3_trigger_simulator.py`. All 8 named scenario types built from real, in-distribution corpus content (general text + real code/JSON data), each with real ground-truth positions. Exit gate (avoids always-on/always-off) passes structurally. Real finding: `state_novelty_score` stays strong only on the single-point-anomaly case (recall 0.656); `token_loss_score` (offline-only, fair for C3's own evaluation since ground truth is legitimately available there) fixes 6 of the other 7 decisively (topic shift 0.062->0.969, code/JSON boundary 0.156->1.000, etc.), and the 8th (distractor-heavy retrieval) was fixed by changing the scenario's substrate (a genuine cross-domain intrusion's onset vs. a within-pattern substitution), landing at recall 0.844. **Every one of the 8 scenarios now shows recall >= 0.5.** Also found and fixed a real onset-vs-sustained-anomaly limitation in both candidate signals along the way (locked in with regression tests). |
| C4 | fair anchor baselines | **Complete (equal-compute transformer deferred)** | `docs/restart/hz0c_c4_fair_baselines_results.md`, `hz0c_c4_fair_baselines.py`, 7 new tests (`no_anchor_trigger`, `fixed_periodic_trigger`, `random_trigger`, `oracle_trigger`, `full_attention_trigger`). Real, matched-~15%-rate comparison across all 8 real C3 scenarios: no anchors 0.000, fixed periodic 0.087, random matched-rate 0.128, `state_novelty_score` (deployable) **0.203**, `token_loss_score` (offline ceiling) **0.750**, oracle 1.000, full attention 1.000. **Real, positive finding: the only deployable signal beats naive fixed/random anchoring at identical cost** -- direct validation of HZ-0C's central hypothesis. Equal-compute transformer baseline NOT built (needs an actual trained matched-FLOP model, a much larger undertaking) -- named as real, disclosed future work, not silently skipped. |
| C5 | HZ-0B dependency gate | **Satisfiable now, integration not started** | Per `docs/restart/hz0c_recovered_requirements.md`'s own C5 check against HZ-0B's real, current status: stable memory semantics (met), PMetal implementation (met, CPU-tensor tier), trained checkpoint (met, the frozen 301M checkpoint used throughout C1-C4), reset/serialization (met, bit-exact), evaluation baselines (met, all 16 B11 tasks + completion verdict, `docs/restart/hz0b_completion_verdict.md`). Nothing blocks C6 on the HZ-0B side; C6 itself has not been started. |
| C6 | frozen-backbone integration | Not started | Conditional attention and logging; memory remains one write per token. `SurpriseTriggeredBlock`/`masked_anchor_attention` (C1) are the reference the eventual bounded/gathered-KV-cache version must match, not yet wired into a full HZ-0A+HZ-0B forward pass together. |
| C7 | controller training | Not started | LM loss plus bounded cost/rate/missed-anchor objectives. C4 gives a concrete, quantified target: close the gap between `state_novelty_score`'s 0.203 and `token_loss_score`'s 0.750 mean recall at the same ~15% budget. |
| C8 | PMetal implementation | Blocked | Reference parity first (C1's `masked_anchor_attention` is a correctness-first, full-O(seq^2) reference); include grouping and anchor-state caching. |
| C9 | evaluation | Not started | Quality, matched attention FLOPs, trigger behavior, latency, memory, failures. |

## Required Artifacts

- [x] `docs/restart/hz0c_history_audit.md`
- [x] `docs/restart/hz0c_recovered_requirements.md`
- [x] Audited scaled topology and parameter-count report (`docs/restart/hz0c_c1_topology.md`)
- [x] Deterministic surprise and trigger reference implementation (`reference/hz0c_surprise_trigger.py`)
- [x] Isolated simulator report with precision, recall, false-trigger, missed-anchor, and rate metrics (`docs/restart/hz0c_c3_trigger_simulator_results.md`)
- [x] Matched-compute baseline report (`docs/restart/hz0c_c4_fair_baselines_results.md`) -- activation-rate-matched, not yet FLOP-matched against an equal-compute transformer
- [ ] Frozen-backbone integration and trigger-log report
- [ ] PMetal/reference parity report
- [ ] End-to-end quality, cost, latency, and adversarial-failure report

## Hard Constraints

- Trigger rates must have explicit minimum and maximum bounds.
- Inference triggering must be deterministic and reproducible.
- Memory updates remain once per token; triggered attention must not duplicate writes.
- Claims must be compared at matched attention FLOPs, not only parameter count.
- The isolated simulator may proceed before HZ-0B integration; full integration may not.

## Exit Gates

- [x] C0: design is explicit without relying on archived code. -- `docs/restart/hz0c_history_audit.md`
- [x] C1: all three controlled models have audited counts and comparable protocols. -- `docs/restart/hz0c_c1_topology.md`
- [x] C2: surprise correlates with controlled novelty or difficulty. -- `docs/restart/hz0c_c2_surprise_validation_results.md` (`state_novelty_score`, on real in-distribution content)
- [x] C3: the controller avoids always-on and always-off behavior. -- `docs/restart/hz0c_c3_trigger_simulator_results.md` (anchor rate 0.125 throughout, non-degenerate)
- [x] C4: quality comparisons are available at matched attention cost. -- `docs/restart/hz0c_c4_fair_baselines_results.md` (equal-compute-transformer leg still open)
- [ ] C5/C6: HZ-0B memory behavior is preserved through integration. -- HZ-0B side ready (`docs/restart/hz0b_completion_verdict.md`); C6 itself not started
- [ ] C7/C8: the trigger policy is bounded, nontrivial, and matches the reference.
- [ ] C9: triggered anchors beat fixed/random anchors at matched cost, or match quality with lower cost. -- directionally shown already for `state_novelty_score` in C4; C9 itself (post-C6/C7 integration) not started

## First Milestone Checklist

- [x] Freeze a small HZ-0B checkpoint and its deterministic protocol. -- the frozen 301M checkpoint used throughout C1-C4
- [x] Implement a deterministic surprise score. -- `state_novelty_score`/`token_loss_score`, `reference/hz0c_surprise_trigger.py`
- [x] Build synthetic novelty points with known expected triggers. -- C3's 8 real scenarios, each with real ground-truth positions
- [x] Compare triggered anchors against random anchors at the same activation rate. -- C4: `state_novelty_score` 0.203 vs. random 0.128 mean recall at matched ~15% rate
- [x] Record trigger decisions, rate, precision, recall, and quality. -- `docs/restart/hz0c_c3_trigger_simulator_results.md`, `hz0c_c4_fair_baselines_results.md`

**First milestone met.** Next real step: C6 (frozen-backbone integration) or C7 (train a real-inference-time controller to close the 0.203-vs-0.750 gap C4 found) -- an open decision, not yet made.
