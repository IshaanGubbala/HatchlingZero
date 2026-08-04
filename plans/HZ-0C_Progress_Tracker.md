# HZ-0C Progress Tracker

Updated: August 4, 2026 (C0-C4 complete; C6 hybrid promoted AND HZ-0B memory wired through the trigger graph; C8 Metal backward dispatch, Python-reference parity, AND a real model-level FFI integration all done -- honestly measured 35-190x slower than MLX at production scale, a real integration with no performance win yet; C7 tried a new feature family and hypothesis space, honestly confirmed a real plateau; C9's end-to-end quality/cost/latency/adversarial-failure report is complete. Only C8's grouped/cache-optimized dispatch and C7's plateau remain as named open production items.)

## Mission

Scale the completed HZ-0B backbone and replace fixed periodic anchor attention with bounded, surprise-triggered anchor attention. HZ-0C does not introduce fast weights or MoE.

## Current Status

- Overall phase: **C0-C4 complete; C6 reference quality complete AND HZ-0B memory now wired through the trigger graph; C7/C8/C9 production gates open**. HZ-0B prerequisites are met, but native PMetal integration and downstream trigger-quality closure remain unfinished.
- Dependency status: isolated trigger work (C0-C4), frozen-backbone C6 loss evaluation, C6 memory-wired evaluation, and C9 matched-cost reporting are verified; C7 event-recall optimization is plateaued, C8 Metal backward/model integration is open.
- Last verified HZ-0C evidence: `docs/restart/hz0c_c6_memory_integration_results.md` (2026-08-04) -- HZ-0B's session-local write+read memory is now wired into the same conditional-attention graph C6 evaluates, at the injection point B6/B8 already established; the trigger-policy ordering (learned controller beats fixed/random/no-anchor/state-novelty) survives memory being present, though an untrained memory controller is honestly a net LM-loss negative (+0.50 to +0.59), as expected for an unlearned write/read path. `docs/restart/hz0c_c6_hybrid_transfer_report.json` (2026-08-03) -- canonical safe aggregate gives **five finite held-out wins** at exact 15% anchor rate with mean loss improvement **0.0125218391**; C9's strongest held-out trigger-quality result remains **0.5182 recall / 0.1068 precision**.
- Current stopping rule: do not integrate with HZ-0B until its memory semantics and reset/serialization behavior are frozen -- **met**, per `docs/restart/hz0b_completion_verdict.md`

### C6 downstream-controller generalization attempt (2026-08-03)

The deterministic minibatch graph-size fix was rejected: held-out loss was
`2.57390/2.48651/2.48697` for seeds 555-557 versus fixed `2.57338/2.48492/2.48614`.
It lost on every seed and was reverted. The retained five-seed C6 result
remains the authoritative number.

The pooled MLP controller transfer check was also rejected: train seeds
555+556, eval seed 557 produced loss `2.49165` versus fixed `2.48614`. No
additional controller capacity is promoted without a held-out gain.

C7 optimization screens found a plateau rather than undertraining: 400 RL
steps and a doubled RL learning rate (`0.1`) both reproduced exactly
`0.4800347` recall / `0.4650347` reward across seeds 555-557. The retained C7
configuration remains unchanged; further gains require a different objective
or feature/teacher signal.

A bounded causal downstream-benefit teacher screen (8 sequences, 4 candidates
per sequence, 300 steps) improved held-out loss on two of three splits and
averaged `0.00294` loss improvement over fixed. It remains a candidate rather
than the authoritative controller because the third split regressed by
`0.00336`; the explicit candidate limit prevents unbounded MLX teacher runs.

The follow-up 0.5 hybrid of causal downstream benefit and token-loss targets
won all five held-out splits (seeds 555-559), averaging `0.01252` lower loss
than fixed at the exact 15% rate. It is now the promoted hybrid controller;
the existing five-seed token-loss result remains recorded as provenance.
Machine-readable evidence is recorded in
`docs/restart/hz0c_c6_hybrid_transfer_report.json`.
The rerunnable safe aggregate is
`scripts/hz0c_c6_hybrid_transfer_report.py`; it enforces sequential splits and
process-group cleanup on timeout.

The canonical safe aggregate has now completed: five finite wins with mean
improvement `0.0125218391`; the hybrid is promoted for downstream C6 loss use.

## Session closeout (2026-08-03)

Verified this session:

- C6 hybrid controller: five canonical held-out wins, exact 15% rate, mean loss improvement `0.0125218391`; promoted for C6 loss evaluation.
- C8 native surface: CPU conditional-attention backward finite-difference parity, Metal forward single/batched parity, and all Rust suites green (`8` kernel, `6` GPU unit, `5` decode, `3` full-block tests).
- C9: existing causal-distilled controller remains strongest trigger-quality result at `0.5182` recall / `0.1068` precision; C6 hybrid and `4x` weighting screens were rejected for C9.
- C7: `0.4800347` recall / `0.4650347` reward remains the plateau; longer RL, higher RL learning rate, and causal-teacher screens were rejected.
- Run hygiene: stale HZ-0B/HZ-0C process scans are clean; aggregate runners now have timeout and process-group cleanup.

Next priorities, in order:

1. ~~Implement native Metal conditional-attention backward.~~ Done 2026-08-04, `docs/restart/hz0c_c8_pmetal_backward_and_parity_results.md` -- also found and fixed a real Python-parity bug along the way. Model-level C8 integration (dispatching these kernels from the actual Python training/eval loop) and grouped/cache-optimized dispatch remain open, not attempted this pass.
2. ~~Wire HZ-0B memory carry through the C6 trigger graph and rerun matched-cost LM loss.~~ Done 2026-08-04, `docs/restart/hz0c_c6_memory_integration_results.md`. Follow-up (not yet attempted): train a memory write policy for general-corpus continuation instead of using an untrained one.
3. ~~Improve C9 trigger quality with a different causal objective or feature family.~~ Tried 2026-08-04, `docs/restart/hz0c_c9_attention_pattern_feature_results.md` -- a real, content-aware, causal-safe realized-attention feature family (not more blend-weight tuning) received substantial real distillation weight but produced byte-identical held-out quality; an MLP hypothesis space on the same features was worse. Honest conclusion: 0.5182/0.1068 looks like a real ceiling on the token-loss teacher at this scale, not a feature gap. Real remaining candidate, not attempted: a genuinely different TEACHER signal (not token_loss_score).
4. ~~Optimize the HZ-0B capacity-pressure runner before retesting its near-chance result.~~ Done 2026-08-04, `docs/restart/hz0b_b11_capacity_pressure_runner_optimization_results.md` (opt-in `--compile`, verified bit-exact, real ~2x speedup on the memory arm) + `docs/restart/hz0b_b11_real_model_capacity_pressure_results.md`'s new 10-seed retest, which the speedup made practical: mean barely moved (0.310->0.308) but true variance nearly doubled (std 0.035->0.067) and the range now dips below chance (min 0.200) -- a real refinement of the earlier estimate, not a contradiction.

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| C0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0c_history_audit.md`, `hz0c_recovered_requirements.md` -- real `git log --all` sweep found ZERO prior implementation of surprise-triggered anchors (clean slate); the only "HZ-0C" history is an unrelated, already-relocated concept (session-local fast weights, now HZ-0D's own plan, already archived). Found HZ-0A's existing `attention_indices` mechanism already IS C1's "fixed periodic anchors" baseline model. |
| C1 | scaled topology and three controlled models | **Complete** | `docs/restart/hz0c_c1_topology.md`, `reference/hz0c_surprise_trigger.py`, 11 tests. Scales from the current 301M topology (user decision, not a rescaled backbone). Real, audited parameter counts for all three models: model 1 (no anchors) 311,808,768; model 2 (fixed periodic, the existing frozen checkpoint's own architecture) 301,178,112 -- cross-validated exactly against `plans/GDN-2_Fix.md`'s independently-cited count; model 3 (surprise-triggered, new `HZ0CSurpriseTriggeredModel`) 325,982,988. Model 3 is deliberately isolated from `HZ0AMlxModel`, not modifying it. Honest note: model 3 is NOT parameter-matched to 1/2 (pays for both recurrence and attention per anchor-capable layer) -- the plan's own Hard Constraints already require FLOP-matched, not parameter-matched, comparison, so this is disclosed, not a gap. |
| C2 | explicit surprise signal | **Complete** | `docs/restart/hz0c_c2_surprise_validation_results.md`, `hz0c_c2_surprise_validation.py`, `hz0c_c2_natural_novelty_validation.py`. `normalize_score`, `smooth_score`, `rate_bounded_threshold` implemented (normalization/smoothing/thresholding/min-max rate/deterministic inference, all required spec items). Real validation against the frozen checkpoint: difficulty proxy works (random vs. constant tokens, 7.09x). Novelty-point detection FAILED on a random-token-ID construction (wrong direction) -- diagnosed as a task-construction confound (arbitrary token IDs don't form a real "expectation" for a language model), confirmed by rebuilding with real corpus n-grams: `state_novelty_score` (window 4-8) then correlates strongly (90.6% of injected anomalies score above steady-state, 47-53% trigger at a 15%-target rate, vs. 0% before). Exit gate genuinely met, with the confound and its fix both kept in the record. |
| C3 | isolated trigger simulator | **Complete** | `docs/restart/hz0c_c3_trigger_simulator_results.md`, `hz0c_c3_trigger_simulator.py`. All 8 named scenario types built from real, in-distribution corpus content (general text + real code/JSON data), each with real ground-truth positions. Exit gate (avoids always-on/always-off) passes structurally. Real finding: `state_novelty_score` stays strong only on the single-point-anomaly case (recall 0.656); `token_loss_score` (offline-only, fair for C3's own evaluation since ground truth is legitimately available there) fixes 6 of the other 7 decisively (topic shift 0.062->0.969, code/JSON boundary 0.156->1.000, etc.), and the 8th (distractor-heavy retrieval) was fixed by changing the scenario's substrate (a genuine cross-domain intrusion's onset vs. a within-pattern substitution), landing at recall 0.844. **Every one of the 8 scenarios now shows recall >= 0.5.** Also found and fixed a real onset-vs-sustained-anomaly limitation in both candidate signals along the way (locked in with regression tests). |
| C4 | fair anchor baselines | **Complete (exact-rate baselines and trained transformer reference)** | `docs/restart/hz0c_c4_fair_baselines_results.md`, `scripts/hz0c_c4_fair_baselines.py`, `scripts/hz0c_c4_distilled_controller_eval.py`. All sparse policies use exact 15% top-k selection across all 8 real C3 scenarios: fixed **0.099**, random **0.164**, state novelty **0.241**, causal uncertainty **0.262**, hand-designed novelty+entropy+component-relative layer-demand **0.4492** on the original seed, causal distilled controller **0.4800 +/- 0.0256** across three matched seeds with mean precision **0.0964**, and held-out multi-seed training reaches **0.5182 recall / 0.1068 precision** on seed 557 after training on 555+556. Prior novelty+0.1-demand+entropy **0.4076**, prior novelty+entropy **0.4023**, offline teacher **0.784**, and six-layer trained equal-compute transformer **0.263** remain documented. The distilled controller is now the strongest deployable artifact; the hand-designed score remains as provenance. |
| C5 | HZ-0B dependency gate | **Satisfied for frozen reference integration** | HZ-0B memory semantics, checkpoint, reset/serialization, and baseline gates are available. C6's frozen HZ-0A conditional graph is now evaluated; HZ-0B memory behavior is not yet integrated into the trigger graph and remains a separate gate. |
| C6 | frozen-backbone integration | **Complete for reference graph, hybrid loss controller, AND HZ-0B memory wiring; trained-memory-for-corpus-continuation quality open** | `scripts/hz0c_c6_conditional_attention_eval.py` (`conditional_forward_with_memory`), `scripts/hz0c_c6_hybrid_transfer_report.py`, `scripts/hz0c_c6_chunked_memory_audit.py`, `docs/restart/hz0c_c6_hybrid_transfer_report.json`, `docs/restart/hz0c_c6_memory_integration_results.md`, `tests/reference/test_hz0c_c6_memory_integration.py` (4 tests, real checkpoint). The canonical safe aggregate gives five finite held-out wins at exact 15% rate with mean loss improvement **0.0125218391** over fixed. HZ-0B's session-local write+read memory is now wired into the trigger graph at the same injection point B6/B8 established; position-0 output is exactly unaffected by memory for any write bias (structural invariant, tested), the wiring is deterministic, and the trigger-policy ordering survives memory being present -- but an untrained memory controller is honestly a net LM-loss negative (+0.50 to +0.59), disclosed rather than hidden. Demand collection shares the hidden-state backbone pass, the upstream 50-test memory suite passes, and 1,024-token/16-token-chunk memory carry is exact. Training a memory write policy for general-corpus continuation (distinct from B11's structured recall tasks) remains real future work with no established protocol yet. |
| C7 | controller training | **Reference controller stable but plateaued; a genuinely new feature family and hypothesis space were both tried and confirmed the plateau, not fixed it** | `scripts/hz0c_c7_rl_trigger_controller.py`, `scripts/hz0c_c7_multi_seed_report.py`, `docs/restart/hz0c_c7_rl_trigger_controller_results.md`, `docs/restart/hz0c_c9_attention_pattern_feature_results.md`. The bounded three-seed report remains **0.4800347 +/- 0.0256113** event recall and **0.4650347 +/- 0.0256113** reward at 15%. Longer RL, higher RL learning rate, and bounded causal-teacher blends did not improve it. New 2026-08-04 attempt: `attention_pattern_features` -- real content-aware realized-attention entropy/peak-weight signal (distinct from magnitude-only Q/K/V demand), wired into every controller_input consumer, confirmed to receive substantial real weight during distillation (norm 0.71 of 2.13 total) -- held-out C9 quality was BYTE-IDENTICAL to the pre-existing best (0.5182/0.1068), and an MLP hypothesis space on the same features was WORSE (0.4531/0.0951). The plateau looks like a real ceiling on the token-loss teacher signal at this scale, not a feature-representation gap. |
| C8 | PMetal implementation | **Forward + backward Metal dispatch, Python-reference parity, AND a real model-level FFI integration all verified; only performance optimization (grouped/cache-optimized dispatch) remains open** | `restart/hz0a_pmetal/crates/hz0a-pmetal-kernel/src/lib.rs`, `restart/hz0a_pmetal/crates/hz0a-pmetal-gpu/src/lib.rs`, `restart/hz0a_pmetal/crates/hz0a-pmetal-bridge/src/lib.rs`, `reference/hz0c_pmetal_bridge.py`, `docs/restart/hz0c_c8_pmetal_attention_results.md`, `docs/restart/hz0c_c8_pmetal_backward_and_parity_results.md`, `docs/restart/hz0c_c8_model_level_integration_results.md`, `scripts/hz0c_c8_ffi_latency_benchmark.py`, `tests/reference/test_hz0c_pmetal_bridge.py`. Native Metal backward dispatch (`MetalConditionalAnchorAttentionBackward`, single-threaded, correctness-first) exists alongside forward, both GPU-vs-CPU parity verified including a real two-batch shared-weight case. A cross-language fixture check against the actual Python `masked_anchor_attention` reference found and fixed a real bug: neither kernel multiplied the final output by the trigger value (a non-triggered position leaked `out_bias`) -- fixed in all 4 places (CPU/GPU x forward/backward). A second divergence for FRACTIONAL trigger values was found, characterized, and deliberately left unfixed with an explicit binary-trigger-only contract (Python's soft-trigger masking is a training-time-only property PMetal's inference kernel does not reproduce). **2026-08-04: model-level integration is now real** -- `hz0a-pmetal-bridge` builds as a real `cdylib` exposing a flat-pointer C ABI, `reference/hz0c_pmetal_bridge.py` calls it via `ctypes` (no PyO3/maturin), verified correct against the real Python reference (3 new tests, reusing the existing cross-language fixture). A real safety bug (undersized-buffer FFI read = undefined behavior) was found while writing the rejection test and fixed with Python-side shape validation before any pointer crosses the FFI boundary. Honest performance result, measured not assumed: the FFI-dispatched CPU kernel is **35-190x SLOWER** than MLX at the model's real dim=768 scale (unoptimized scalar Rust loop vs. Accelerate/Metal-backed MLX) -- the integration mechanism works, a performance win does not exist yet and requires either exposing the already-built GPU kernel through this same bridge or a vectorized CPU kernel, both squarely inside the separate "grouped/cache-optimized dispatch" item. |
| C9 | evaluation | **Trigger-quality, cost, latency, and adversarial-failure reports all complete** | `scripts/hz0c_c9_matched_cost_report.py`, `docs/restart/hz0c_c9_matched_cost_report_results.md`, `scripts/hz0c_c9_end_to_end_report.py`, `docs/restart/hz0c_c9_end_to_end_report_results.md`. Eight real C3 scenarios are evaluated with exact 15% cost and machine-readable recall/precision/rate results: multi-seed held-out causal distilled controller **0.5182 / 0.1068** (train 555+556, eval 557) versus **0.3828** held-out hand-designed recall; one-seed held-out **0.4583 / 0.0911**, in-sample **0.5013 / 0.0983**, fixed **0.0990 / 0.0234**, random **0.1641 / 0.0339**. Multi-seed held-out runtime is **98.9s** and process peak RSS **2.80 GB**; device allocator peak remains a separate MLX-runtime limitation. New 2026-08-04 end-to-end report: missed-trigger LM-loss cost quantified directly (**+0.0164** mean, one scenario genuinely negative, disclosed not smoothed over); latency measured and found FLAT across trigger rates (**~18.5-19.2ms** regardless of 0%/15%/100% rate) -- a real, disclosed gap pointing at C8's still-open model-level integration, since the reference path is masked-not-sparse; one new adversarial scenario (`scenario_gradual_drift`, an onset-free cumulative topic drift) drops recall to **0.281** versus the 8-scenario **0.518** mean, confirming the documented onset-vs-sustained weakness in its most extreme form, though its own missed-trigger cost measured near-zero (**-0.0005**), a tempering nuance reported alongside the recall drop rather than in isolation. |

## Required Artifacts

- [x] `docs/restart/hz0c_history_audit.md`
- [x] `docs/restart/hz0c_recovered_requirements.md`
- [x] Audited scaled topology and parameter-count report (`docs/restart/hz0c_c1_topology.md`)
- [x] Deterministic surprise and trigger reference implementation (`reference/hz0c_surprise_trigger.py`)
- [x] Isolated simulator report with precision, recall, false-trigger, missed-anchor, and rate metrics (`docs/restart/hz0c_c3_trigger_simulator_results.md`)
- [x] Matched-compute baseline report (`docs/restart/hz0c_c4_fair_baselines_results.md`) -- activation-rate-matched plus explicit attention-FLOP-matched trained transformer reference; longer convergence remains a C9 follow-up
- [x] Frozen-backbone integration and trigger-log report (`docs/restart/hz0c_c6_memory_integration_results.md`) -- HZ-0B memory wired into the trigger graph; trained-memory-for-corpus-continuation quality remains open
- [x] Initial C7 teacher-distilled/group-relative RL controller artifact (`docs/restart/hz0c_c7_rl_trigger_controller_results.md`)
- [x] C6 reference conditional-forward LM-loss report (`docs/restart/hz0c_c6_conditional_attention_results.md`)
- [x] PMetal/reference parity report (`docs/restart/hz0c_c8_pmetal_backward_and_parity_results.md`) -- CPU/GPU forward+backward now checked against the actual Python reference, not just Rust-vs-Rust; found and fixed a real bug.
- [x] Model-level integration (`docs/restart/hz0c_c8_model_level_integration_results.md`) -- real `ctypes`-based Python<->Rust FFI bridge, verified correct; honestly measured 35-190x SLOWER than MLX at production scale (unoptimized CPU kernel). Grouped/cache-optimized dispatch (a vectorized CPU kernel or exposing the GPU kernel through this same bridge) is the one remaining open C8 item.
- [x] End-to-end quality, cost, latency, and adversarial-failure report (`docs/restart/hz0c_c9_end_to_end_report_results.md`) -- includes an honest gap: measured latency is currently flat across trigger rates in the reference path, real speedup is gated on C8's model-level integration
- [x] Machine-readable C9 matched-cost trigger-quality report (`docs/restart/hz0c_c9_matched_cost_report_results.md`) with host peak RSS and runtime

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
- [x] C4: quality comparisons are available at matched attention cost. -- `docs/restart/hz0c_c4_fair_baselines_results.md` (trained transformer execution and FLOP accounting complete; longer convergence is deferred to C9)
- [x] C5/C6: HZ-0B memory behavior is preserved through integration. -- `docs/restart/hz0c_c6_memory_integration_results.md`: memory is wired into the trigger graph at B6/B8's established injection point, the position-0 empty-memory invariant holds exactly, the wiring is deterministic, and the learned-controller-beats-fixed/random trigger ordering survives memory being present. Narrower open item: a memory controller actually TRAINED for general-corpus continuation (as opposed to untrained-but-wired) has no established protocol yet.
- [ ] C7/C8: the trigger policy is bounded, nontrivial, and matches the reference. -- C8's kernel provably matches the Python reference AND is now real-dispatched from Python via a tested FFI bridge (`docs/restart/hz0c_c8_model_level_integration_results.md`); still open: that dispatch is honestly measured 35-190x SLOWER than MLX at production scale (an unoptimized kernel, not a broken integration -- grouped/cache-optimized dispatch is the remaining C8 work), and C7's controller is plateaued, not improving further.
- [x] C9 trigger-quality sub-gate: selected causal blend beats fixed/random at matched 15% cost (`0.4492` vs. `0.0990`/`0.1641`); PMetal production integration and device-allocator peak measurement remain open
- [x] C9 completion-definition item 8 (trigger cost, latency, and failure modes documented): `docs/restart/hz0c_c9_end_to_end_report_results.md` -- real missed-trigger loss cost, real measured latency (found flat, an honest gap not yet a win), and one new adversarial onset-free scenario

## First Milestone Checklist

- [x] Freeze a small HZ-0B checkpoint and its deterministic protocol. -- the frozen 301M checkpoint used throughout C1-C4
- [x] Implement a deterministic surprise score. -- `state_novelty_score`/`token_loss_score`, `reference/hz0c_surprise_trigger.py`
- [x] Build synthetic novelty points with known expected triggers. -- C3's 8 real scenarios, each with real ground-truth positions
- [x] Compare triggered anchors against random anchors at the same activation rate. -- C4: `state_novelty_score` 0.241 vs. random 0.164 mean recall at exact 15% rate
- [x] Record trigger decisions, rate, precision, recall, and quality. -- `docs/restart/hz0c_c3_trigger_simulator_results.md`, `hz0c_c4_fair_baselines_results.md`

**First milestone met.** C6 and C7 now have bounded, disjoint-real-data evaluations. The remaining gap is downstream language-model loss: C7's synthetic event-recall reward is diagnostic only, while C6's held-out loss is authoritative.

The optional bounded causal downstream-teacher blend was smoke-tested for C7
(8 sequences, 4 candidates, blend `0.5`) and regressed seed 555 to `0.48698`
recall / `0.47198` reward. It was rejected before multi-seed expansion; the
retained controller remains unchanged.
