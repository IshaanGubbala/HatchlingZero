# HZ-0E E0: Recovered Requirements

Date: 2026-08-05. Synthesizes `docs/restart/hz0e_history_audit.md`'s
findings into an explicit requirements list for E1's contract, per E0's
own exit gate (recover and classify prior expert counts, placement,
routing, balancing, capacity rules, and failures).

Unlike HZ-0D's D0 (which recovered real, reusable prior fast-weight
work under a relocated name) or HZ-0C's C0 (a clean slate with one
piece of directly reusable live infrastructure), HZ-0E's audit found
**no prior MoE work of any kind** (history audit finding 1) -- so this
document has nothing to warn against repeating and nothing broken to
avoid trusting. It instead synthesizes the plan's own E1-E10 text
against the one piece of real substrate the audit DID find (finding 3:
HZ-0A's existing per-block dense FFN), plus the cross-project
discipline established across HZ-0B/C/D that should carry forward as
working method, not as reusable code.

## Requirements carried forward from the plan (unconditional, E1's text)

1. Start conservative: **4 experts, top-1 routing, shared dense
   fallback, selected upper MLP blocks only** (the plan's own explicit
   starting point, not a choice this document is making).
2. E1's exit gate: exact total and active parameter counts must be
   known before anything else proceeds -- always reported SEPARATELY
   (E4, E10 both repeat this), never conflated.
3. Experts stay OUT of stateful internals initially (E6/E7): not inside
   the GDN-2 recurrent update, not inside HZ-0B memory writes, not
   inside the HZ-0C surprise controller, not inside the HZ-0D fast-
   weight update controller. Router logits are not directly controlled
   by HZ-0C surprise; HZ-0D fast weights do not modify the router;
   memory writes stay exactly once per token; routing itself happens
   once per MoE layer per token; inference routing is deterministic.
4. E5's dependency gate: full integration waits for a frozen HZ-0D
   (stable recurrence, HZ-0B memory, HZ-0C trigger behavior, HZ-0D
   snapshot/rollback, PMetal, a trained checkpoint) -- **already
   satisfied**, per `plans/HZ-0D_Progress_Tracker.md` (all of D0-D10
   complete, verified in `docs/restart/hz0d_d10_evaluation_results.md`)
   and confirmed further by the joint HZ-0A/B/C/D evaluation
   (`docs/restart/hz0abcd_joint_evaluation_results.md`, real checkpoint,
   real corpus, bounded and finite across seeds). The router simulator
   (E2) may proceed regardless, per the plan's own E5 text -- matching
   HZ-0D's own D2-before-D5 precedent.

## Requirements added from the real substrate found in the audit (Finding 3)

5. **The expert-candidate layer set is a real, concrete subset of a
   known 31-block model, not an abstract design choice.** Every block
   (6 full-attention layers at indices `4,9,14,19,24,29`, 25 GDN-2
   recurrent layers otherwise) has an IDENTICAL dense SwiGLU FFN shape:
   `gate`/`up`: `Linear(768, 2304)`, `down`: `Linear(2304, 768)`,
   `5,313,792` params per block. E1 must pick which specific blocks
   count as "upper" (most naturally the last N of 31, nearest the
   output, matching the plan's own "upper MLP blocks" language) and
   state the exact indices and per-expert size relative to this real
   `164,727,552`-param dense-FFN total -- not a placeholder range.
6. Whether MoE-replaced blocks should overlap with the 6 attention
   layers, the 25 GDN-2 layers, both, or be chosen independently of
   that split is explicitly OPEN -- E0 does not resolve it, E1 must.

## Requirements added from cross-project discipline (HZ-0B/C/D precedent, not code)

7. **Measure before choosing a mechanism, the same way D3 did.**
   HZ-0D's D3 phase did not pick gradient descent, Hebbian, or delta
   prediction by argument alone -- it built all named candidates, ran a
   real 4-way comparison, found and FIXED a real weakness (delta
   prediction's label-noise collapse) through three iterations, and
   only then selected a default, later revised again at real scale
   (D10) when new evidence (gradient descent's real-scale instability)
   emerged. E3's own text ("evaluate language-model loss, load
   balancing, router z-loss, overflow penalty, diversity regularization,
   supervised warm starts") should be treated the same way: build and
   measure the named routing objectives, not assume one a priori.
8. **A native/GPU tier is justified by a real benchmark, not built
   reflexively.** HZ-0B's B10 and HZ-0D's D9 both deferred a GPU kernel
   tier until a real, measured reason existed (D9: fast-weight overhead
   was already `<8%` of a real forward pass at the CPU-tensor tier).
   E9's PMetal work should apply the same standard -- build the CPU/
   reference tier first, benchmark real dispatch overhead at the chosen
   expert count/size, and justify (or explicitly defer) a GPU tier from
   that number, not from an a priori assumption that routing needs one.
9. **Disclose real limitations found along the way rather than smoothing
   them into the headline result.** D3's rank-misspecification limit,
   D6's real-scale learning-rate transfer problem, and D10's gradient-
   descent real-scale divergence were each investigated, quantified,
   and locked in as regression tests rather than hidden or silently
   patched. E2's "collapse" and E7's "uncontrolled feedback loop" exit
   gates should be held to the same bar: if a real failure mode is
   found (e.g. router collapse under a specific domain-imbalance
   pattern), it gets characterized and tested, not just avoided by
   picking easier evaluation conditions.

## Exit gate check

E0's exit gate: prior expert counts, placement, routing, balancing,
capacity rules, and failures are recovered and classified. Met -- the
honest classification is that there is nothing to recover (finding 1:
zero prior MoE work in this project's own code), so this document
instead grounds E1's upcoming contract in the one real piece of
substrate the audit found (HZ-0A's actual 31-block dense-FFN
architecture, measured exactly from the real checkpoint) and in the
working discipline already validated across HZ-0B, HZ-0C, and HZ-0D,
carried forward as method rather than as code.
