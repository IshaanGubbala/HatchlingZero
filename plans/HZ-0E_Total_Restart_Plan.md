# HZ-0E Total Restart Plan

## Starting Point

This plan assumes no working HZ-0E implementation exists.

Only the README, master plan, and Git history are available.

Everything related to micro-MoE must be rebuilt:

- expert architecture
- routing
- capacity handling
- load balancing
- PMetal kernels
- training curriculum
- evaluation
- integration rules

---

## Objective

HZ-0E adds a small mixture-of-experts layer to a completed HZ-0D model.

HZ-0E should answer:

> Can a small number of specialized experts increase model capacity and specialization without proportionally increasing per-token compute or destabilizing the recurrent, memory, fast-weight, and adaptive-compute systems?

HZ-0E should be a micro-MoE, not a giant sparse model.

Initial goals:

- modest expert count
- bounded routing
- deterministic inference
- low routing overhead
- no expert collapse
- Apple-Silicon-friendly execution

---

# Phase E0 — Git-history audit

Inspect Git history for:

- intended expert count
- expert placement
- top-k routing
- load-balancing losses
- capacity rules
- specialization goals
- prior performance assumptions
- failed implementations

Classify findings as confirmed, uncertain, or rejected.

## Deliverables

- `docs/restart/hz0e_history_audit.md`
- `docs/restart/hz0e_recovered_requirements.md`

## Exit gate

The micro-MoE design is fully specified without relying on archived code.

---

# Phase E1 — Define the expert contract

Specify:

- number of experts
- expert hidden size
- expert placement
- router input
- top-1 or top-2 routing
- capacity factor
- overflow behavior
- shared expert policy
- expert dropout
- deterministic inference
- parameter budget
- active-parameter budget

A reasonable first configuration:

```text
4 experts
top-1 routing
1 shared expert or shared dense path
experts replace selected MLP blocks
```

Do not begin with many experts or top-2 routing.

## Exit gate

Exact total and active parameter counts are known.

---

# Phase E2 — Build an isolated router simulator

Synthetic routing tasks:

1. domain classification
2. code versus prose
3. math versus dialogue
4. JSON versus natural language
5. tool planning versus completion
6. mixed-domain batches
7. imbalanced domains
8. unseen mixtures

Measure:

- routing accuracy
- expert utilization
- load balance
- overflow
- entropy
- collapse
- routing stability
- sensitivity to noise

## Exit gate

The router uses multiple experts and avoids collapse under imbalanced data.

---

# Phase E3 — Establish routing and balancing losses

Potential losses:

```text
language-model loss
router auxiliary load-balancing loss
router z-loss
capacity overflow penalty
expert diversity penalty
shared-expert regularization
```

Compare:

- top-1 routing
- top-2 routing
- soft mixture
- hash routing
- domain-supervised warm start
- fully latent routing

Start with supervised or weakly supervised routing before latent specialization.

## Exit gate

Routing remains balanced without overwhelming the task loss.

---

# Phase E4 — Establish fair baselines

Compare against:

- dense MLP with same active parameters
- dense MLP with same total parameters
- wider dense MLP
- low-rank adapters
- domain-specific adapters
- shared expert only
- static expert assignment

Report both:

- total parameters
- active parameters per token

## Exit gate

Micro-MoE benefits cannot be explained only by extra total capacity.

---

# Phase E5 — Wait for stable HZ-0D

Full integration waits until HZ-0D has:

- stable adaptive-compute policy
- deterministic PMetal execution
- trained checkpoint
- known latency and compute budgets
- reliable state semantics

The isolated router simulator may proceed independently.

## Exit gate

A frozen HZ-0D checkpoint is available.

---

# Phase E6 — Integrate with frozen HZ-0D

Begin by replacing only selected upper-layer MLPs.

Keep HZ-0D frozen.

Train:

- router
- experts
- shared expert
- load-balancing parameters

Do not initially place experts inside:

- GDN-2 recurrent state updates
- HZ-0B memory writes
- HZ-0C fast-weight update controllers
- HZ-0D halt controller

Keep the first MoE integration local to MLP computation.

## Exit gate

MoE improves target tasks without destabilizing stateful systems.

---

# Phase E7 — Specialization curriculum

Train on balanced domains:

- general prose
- code
- math
- technical documentation
- JSON and structured output
- tool schemas and planning

Stages:

1. supervised routing warm start
2. partially latent routing
3. fully latent routing
4. mixed-domain sequences
5. domain shifts within one sequence
6. adversarial imbalance

## Exit gate

Experts develop measurable specialization while remaining usable across domains.

---

# Phase E8 — PMetal implementation

Potential PMetal components:

```text
router_logits
top_k_route
token_expert_dispatch
expert_mlp
expert_output_combine
capacity_management
```

Apple-Silicon concerns:

- small expert batches
- dispatch overhead
- memory movement
- synchronization
- dynamic shapes
- expert weight residency

Optimize for few experts and coarse routing.

Possible execution strategies:

1. grouped tokens by expert
2. fixed-capacity expert buffers
3. fused dispatch and combine
4. shared dense fallback for overflow

Keep a slow reference implementation.

## Exit gate

PMetal matches the reference and provides end-to-end benefit after routing overhead.

---

# Phase E9 — Interaction with adaptive compute

Define the relationship between HZ-0D and HZ-0E.

Questions:

- Does each recurrent refinement step reroute?
- Is routing fixed per token?
- Can later steps choose a different expert?
- Does expert choice affect halting?

Safest first design:

> route once per layer per token, keep routing fixed across internal adaptive recurrence steps, and prevent MoE routing from directly controlling the halt decision.

This limits instability.

## Exit gate

Routing and adaptive compute do not form an uncontrolled feedback loop.

---

# Phase E10 — Evaluation

Measure:

- held-out language loss
- code performance
- math performance
- structured-output performance
- domain specialization
- expert utilization
- load balance
- overflow
- total parameters
- active parameters
- training throughput
- inference throughput
- p50/p95 latency
- memory residency
- routing overhead
- quality per active FLOP

## Exit gate

HZ-0E beats dense baselines at matched active compute or matched quality.

---

# HZ-0E completion definition

HZ-0E is complete when:

1. Expert and router semantics are explicit.
2. Router simulator avoids collapse.
3. Expert specialization is measurable.
4. Integration preserves HZ-0B, HZ-0C, and HZ-0D behavior.
5. It beats fair dense baselines.
6. PMetal execution provides net benefit after routing overhead.
7. Total and active parameter counts are reported separately.
8. Capacity overflow and worst-case latency are bounded.
9. Routing is deterministic at inference.
10. Limitations and failure modes are documented honestly.

---

# First concrete milestone

> A four-expert top-1 router with a shared fallback that stays balanced on mixed synthetic domains and beats a matched dense baseline after accounting for routing overhead.
