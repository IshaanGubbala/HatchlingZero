# HZ-0E Total Restart Plan

## Objective

HZ-0E adds **micro-MoE FFNs and sparse routing** to a completed HZ-0D model.

It must answer:

> Can a small number of specialized experts add capacity without proportional active compute or destabilizing recurrence, memory, triggered attention, and fast weights?

## Starting point

Assume no working MoE implementation survives. Rebuild routing, experts, capacity handling, PMetal execution, and evaluation from explicit specifications.

## E0 — Audit Git history

Recover and classify prior expert counts, placement, routing, balancing, capacity rules, and failures.

Create:

- `docs/restart/hz0e_history_audit.md`
- `docs/restart/hz0e_recovered_requirements.md`

## E1 — Define the expert contract

Specify expert count and size, placement, top-k policy, capacity factor, overflow behavior, shared fallback, total versus active parameters, deterministic inference, and whether HZ-0D fast weights may later modify expert adapters.

Start conservatively:

```text
4 experts
top-1 routing
shared dense fallback
selected upper MLP blocks only
```

**Exit gate:** exact total and active parameter counts are known.

## E2 — Isolated router simulator

Test routing on code, prose, math, JSON, tools, mixed domains, imbalance, domain shifts, and noisy inputs.

Measure utilization, balance, overflow, entropy, collapse, and stability.

**Exit gate:** multiple experts remain active without collapse.

## E3 — Routing objectives

Evaluate language-model loss, load balancing, router z-loss, overflow penalty, diversity regularization, and supervised warm starts.

**Exit gate:** balancing does not overwhelm task learning.

## E4 — Fair baselines

Compare with dense MLPs at matched active and total parameters, wider dense MLPs, domain adapters, static expert assignment, and shared-expert-only models.

Always report total and active parameters separately.

## E5 — Dependency gate

Full integration waits for a frozen HZ-0D with stable recurrence, HZ-0B memory, HZ-0C trigger behavior, HZ-0D snapshot/rollback, PMetal implementation, and a trained checkpoint.

The router simulator may proceed earlier.

## E6 — Frozen-backbone integration

Replace selected upper MLPs only.

Do not initially put experts inside GDN-2 updates, HZ-0B writes, the HZ-0C surprise controller, or the HZ-0D update controller.

**Exit gate:** MoE improves target tasks without disrupting stateful systems.

## E7 — Interaction rules

Initially:

- routing occurs once when each MoE layer executes
- HZ-0C surprise does not directly control router logits
- HZ-0D fast weights do not modify the router
- memory writes remain once per token
- inference routing is deterministic

**Exit gate:** no uncontrolled feedback loop exists among surprise, routing, memory, and fast updates.

## E8 — Specialization curriculum

Train progressively on balanced prose, code, math, technical documents, JSON, and tool tasks, then mixed-domain and adversarially imbalanced sequences.

**Exit gate:** experts show measurable specialization without becoming unusable elsewhere.

## E9 — PMetal implementation

Implement router logits, top-k routing, token dispatch, expert MLPs, output combining, capacity handling, and shared fallback.

Optimize for few experts, grouped tokens, fixed-capacity buffers, low dispatch overhead, and Apple-Silicon weight residency.

**Exit gate:** PMetal matches the reference and provides a net end-to-end benefit.

## E10 — Evaluation

Measure language/code/math/structured quality, expert utilization and specialization, overflow and balance, total and active parameters, throughput, memory, latency, dispatch overhead, quality per active FLOP, and interaction failures with HZ-0B/C/D.

**Exit gate:** HZ-0E beats fair dense baselines at matched active compute or matched quality.

## Completion definition

HZ-0E is complete when:

1. Router and expert semantics are explicit.
2. Routing avoids collapse.
3. Specialization is measurable.
4. HZ-0B, HZ-0C, and HZ-0D remain stable.
5. It beats fair dense baselines.
6. PMetal has net benefit after routing overhead.
7. Total and active parameter counts are distinct.
8. Overflow and tail latency are bounded.
9. Inference routing is deterministic.
10. Limitations are documented.

## First milestone

> A four-expert top-1 router with shared fallback that remains balanced on mixed synthetic domains and beats a matched dense baseline after routing overhead.
