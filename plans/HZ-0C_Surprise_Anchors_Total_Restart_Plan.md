# HZ-0C Total Restart Plan

## Objective

HZ-0C scales the completed HZ-0B backbone and replaces fixed periodic anchor attention with **surprise-triggered anchor attention**.

It must answer:

> Can HZ spend quadratic attention only when the recurrent state encounters something unexpected, preserving or improving quality at lower average attention cost?

HZ-0C does **not** introduce fast weights or MoE.

## Starting point

Assume only the README, master plan, and Git history survive. No prior HZ-0C implementation is trusted until reproduced.

## C0 — Audit Git history

Recover and classify prior work on surprise signals, fixed and triggered anchor schedules, scale-up configurations, attention insertion points, latency and quality measurements, and known collapse modes.

Create:

- `docs/restart/hz0c_history_audit.md`
- `docs/restart/hz0c_recovered_requirements.md`

**Exit gate:** the design is explicit without depending on archived code.

## C1 — Freeze the scaled topology

Define target parameter count, hidden size, depth, recurrent/attention arrangement, inherited HZ-0B memory placement, context length, fixed-anchor baseline schedule, expected trigger rate, and compute budget.

Construct three controlled models:

1. scaled recurrence with no anchors
2. scaled recurrence with fixed periodic anchors
3. scaled recurrence with surprise-triggered anchors

**Exit gate:** all models have audited parameter counts and comparable protocols.

## C2 — Define surprise

Evaluate simple scalar signals such as recurrent-state prediction error, hidden-state delta norm, token-loss proxy, state novelty, recurrent/attention disagreement, and HZ-0B memory-read uncertainty.

Specify normalization, smoothing, thresholding, minimum and maximum trigger rates, and deterministic inference behavior.

**Exit gate:** surprise correlates with controlled novelty or difficulty.

## C3 — Isolated trigger simulator

Test repeated patterns with anomalies, topic shifts, long-range key reappearance, changed variable bindings, code or JSON boundaries, contradictions, rare-token bursts, and distractor-heavy retrieval.

Measure trigger precision and recall, false-trigger rate, missed anchors, and average anchor rate.

**Exit gate:** the controller avoids always-on and always-off behavior.

## C4 — Fair anchor baselines

Compare against no anchors, fixed anchors, random anchors at matched rate, oracle anchors, full attention, and an equal-compute transformer.

**Exit gate:** quality can be compared at matched attention FLOPs.

## C5 — Dependency gate

Full integration waits for a frozen HZ-0B with stable memory semantics, PMetal implementation, trained checkpoint, reset/serialization, and evaluation baselines.

The isolated trigger simulator may proceed earlier.

## C6 — Frozen-backbone integration

Add a surprise estimator, bounded trigger decision, conditional anchor-attention path, and trigger logging to frozen HZ-0B.

Memory updates remain once per token. Triggered attention must not duplicate memory writes.

**Exit gate:** trigger integration preserves HZ-0B memory behavior.

## C7 — Train the controller

Potential objectives include language-model loss, attention-cost penalty, trigger sparsity/rate penalty, missed-anchor penalty, and trigger entropy regularization.

Train with supervised synthetic labels first, then latent matched-compute routing.

**Exit gate:** the model learns a bounded, nontrivial trigger policy.

## C8 — PMetal implementation

Implement and validate surprise scoring, trigger decisions, conditional anchor attention, triggered-token grouping, and anchor-state caching.

Keep a slow reference implementation.

**Exit gate:** PMetal matches reference outputs and trigger decisions.

## C9 — Evaluation

Measure validation loss, long-context retrieval, code and structured-output quality, trigger rate, attention FLOPs, throughput, memory, latency, missed-trigger failures, and adversarial novelty behavior.

**Exit gate:** HZ-0C beats fixed or random anchors at matched attention cost, or matches quality with lower cost.

## Completion definition

HZ-0C is complete when:

1. The scaled topology is frozen and audited.
2. Surprise is explicit and reproducible.
3. Trigger behavior works on controlled novelty tasks.
4. The controller avoids collapse.
5. HZ-0B memory behavior is preserved.
6. Triggered anchors beat fair baselines.
7. PMetal matches the reference.
8. Trigger cost, latency, and failure modes are documented.

## First milestone

> A small frozen HZ-0B model with a deterministic surprise score that triggers anchor attention on synthetic novelty points and beats random anchors at the same activation rate.
