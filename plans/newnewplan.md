Absolutely. The old plan needs a real rewrite now because the 4/8 cached-evidence result changes the architecture direction materially.

# HatchlingZero vNext Plan — Revised after cached-evidence crux

## 0. Updated thesis

The previous vNext thesis was:

$$
\text{exact address} \rightarrow
\text{cheap compressed reasoning} \rightarrow
\text{occasional re-address}
$$

That was too aggressive.

The new evidence says:

1. **Exact re-addressing is useful every round.**
2. But reducing its frequency produces a **graceful quality/compute tradeoff**, not immediate collapse.
3. The catastrophic BDH-Δ regression came primarily from the new compressed belief/workspace/Think Cell system.
4. Therefore we should preserve BDH's existing full-dimensional representation and modify its dynamics **in-place**.

New thesis:

$$
\boxed{
\text{Keep BDH's full-state exact-address machinery intact,
but make its recurrent dynamics cheaper, more stable, and more adaptive.}
}
$$

vNext should evolve **from the current champion**, not replace its internal representation.

---

# 1. Canonical starting point

Start from the strongest validated architecture:

$$
\boxed{
\text{compound BDH}
+
\text{rank-64 decoder}
+
\text{single }g_1\text{ residual gate}
+
\text{torch.compile}
}
$$

Keep:

* exact high-rank addressing
* full \(D=2496\) recurrent state
* weight tying
* rank-64 output/value-side compression
* \(m=16\)-class reduced width
* BF16
* static/preallocated execution
* compiled training/inference
* existing depth curriculum where useful

Do **not** insert a new latent coordinate system between BDH rounds.

---

# 2. Kill the compressed world-model architecture

The following BDH-Δ components should be removed from the main architecture:

* 384-d belief bottleneck
* 8×96 separate workspace
* separate belief cell
* cross-token compressed belief carry
* large fresh Think Cell
* independent compressed latent dynamics
* predictor/corrector built around that bottleneck

Reason:

$$
1.7862-1.5125\approx0.274
$$

Most of BDH-Δ's damage came from the new machinery, not cached addressing.

The failure looks like a **representation/interface failure**, not evidence that adaptive recurrence itself is impossible.

---

# 3. Preserve full-state reasoning

Any new internal-computation mechanism now operates directly on:

$$
h_r\in\mathbb{R}^{2496}
$$

rather than:

$$
2496\rightarrow384\rightarrow\text{workspace}.
$$

So the same representational space that BDH already knows how to use remains intact.

This becomes a hard architectural principle:

$$
\boxed{\text{Don't force BDH through a new bottleneck unless evidence demands it.}}
$$

We've already repeatedly learned that BDH tolerates value/output compression much better than changes to representations involved in its internal computation.

---

# 4. Separate two concepts we previously bundled

There are really two different questions:

### Evidence refresh

$$
e_r=A(h_r,x)
$$

How often do we pay for exact addressing?

### State evolution

$$
h_{r+1}=U(h_r,e_r)
$$

How should the state change after receiving evidence?

These should remain conceptually separate even if they stay tightly coupled in implementation.

The cached-evidence experiment is measuring the first.

The next architecture work should focus on the second.

---

# 5. Evidence refresh becomes a tunable cadence

We now know:

$$
8/8 \approx 1.414\text{–}1.433
$$

and:

$$
4/8=1.5125.
$$

So exact refresh frequency is neither:

> totally redundant

nor:

> absolutely required every iteration.

It is a continuous quality/compute control.

Real, 2026-08-30 (RTX 5090, matched 25M-token budget, seed=7):

$$
8/8 \approx 1.414\text{–}1.433,\quad 6/8 = 1.4505,\quad 4/8=1.5125,\quad 2/8=\text{pending}.
$$

`final_g1` at 6/8 landed at 0.5366 -- same attractor family as 8/8's
0.583/0.586 and 4/8's 0.5748, now a fourth independent confirmation.

Once the frontier lands, define the production cadence from measured Pareto efficiency.

Potential outcomes:

```text id="qv9x4u"
8 refreshes = maximum quality (1.414-1.433)
6 refreshes = real, small cost (+0.017-0.036) -- looks like the efficiency knee
4 refreshes = real, moderate cost (+0.078-0.098) -- aggressive speed mode
2 refreshes = likely too stale, pending
```

Do not hard-code 4/8 as the vNext architecture.

The frontier decides.

---

# 6. Introduce the concept of **evidence lifetime**

The useful question is:

$$
\boxed{
\text{How many state transformations remain useful before evidence becomes stale?}
}
$$

Call that the **evidence lifetime**.

If 6/8 performs almost like 8/8, evidence can survive roughly one skipped refresh.

If 4/8 loses much more, two consecutive transformations on stale evidence is too much.

This should guide the architecture directly.

Instead of blindly:

```text id="g4xmnf"
refresh
think
think
refresh
think
think
```

we can design around the measured lifetime.

---

# 7. Full-state Adaptive Delta BDH

This is now the main architectural change I would pursue.

Current update approximately:

$$
h_{r+1}=\operatorname{LN}(h_r+y_r).
$$

We already learned the write should be scaled:

$$
h_{r+1}
=
\operatorname{LN}(h_r+g_1y_r)
$$

with:

$$
g_1\approx0.58.
$$

Now generalize that minimally.

Instead of a single global scalar:

$$
g_1
$$

use a **state-dependent gate**:

$$
g_r=g_\theta(h_r,e_r).
$$

Then:

$$
\boxed{
h_{r+1}
=
\operatorname{LN}
\left(
h_r+g_r\,y_r
\right)
}
$$

No new hidden representation.

No separate Think Cell.

No new workspace.

Just make the already-successful gate adaptive.

---

# 8. Keep the controller tiny

This controller should be deliberately tiny.

For example:

$$
q_r=
[
\operatorname{RMS}(h_r),
\operatorname{RMS}(y_r),
\cos(h_r,y_r),
\operatorname{RMS}(h_r-h_{r-1})
].
$$

Then:

$$
g_r=\sigma(W_2\phi(W_1q_r)).
$$

Maybe only tens or hundreds of parameters.

Alternatively a per-channel gate:

$$
g_r\in\mathbb{R}^{D}
$$

could be tested later, but start scalar or per-head.

The important point is:

$$
\boxed{\text{controller complexity} \ll \text{BDH state complexity}.}
$$

We don't want another 4.6M-parameter subsystem inventing a parallel coordinate system.

---

# 9. Protect initialization at the known-good solution

This is critical.

Initialize adaptive gating so:

$$
g_r\approx0.58.
$$

Not zero.

Not one.

Not random.

We now have remarkable reproducibility:

$$
0.583,\quad0.586,\quad0.5748.
$$

So ~0.58 looks like a genuine attractor of the existing dynamics.

Start vNext **exactly there**.

Then the new controller initially behaves approximately like the known-good single-gate architecture:

$$
g_\theta(h,e)\approx0.58.
$$

Training only has to learn deviations from a working solution.

This uses the protected-learning lesson correctly.

---

# 10. Add a bounded delta-update

The next minimal extension is to constrain update magnitude:

$$
u_r=g_r y_r.
$$

Then:

$$
u_r'
=
\alpha_r
\frac{u_r}
{\operatorname{RMS}(u_r)+\epsilon}.
$$

And:

$$
h_{r+1}
=
\operatorname{LN}(h_r+u_r').
$$

But this should be tested conservatively.

The purpose is to prevent:

$$
R=12,16
$$

from driving the representation off-manifold.

We want recurrence to behave more like:

$$
h_{r+1}=h_r+\delta_r
$$

than repeatedly performing unrestricted state rewrites.

---

# 11. No explicit round identity

Do not reintroduce round embeddings.

The controller should answer:

> How much should I update given my current state?

not:

> What round number am I on?

So all control signals come from state dynamics:

* change magnitude
* evidence disagreement
* residual magnitude
* confidence/convergence proxies

not \(r\).

This preserves extrapolation potential.

---

# 12. Evidence disagreement, but in full state

This idea survives BDH-Δ.

Calculate something cheap like:

$$
d_r=
\cos(P_hh_r,P_ee_r).
$$

Or even avoid projections initially:

$$
d_r=
\cos(h_r,e_r)
$$

if dimensions align meaningfully.

Feed it into the update gate:

$$
g_r=f(q_r,d_r).
$$

Interpretation:

* evidence agrees with state → smaller update
* evidence conflicts → stronger correction

This gives recurrence a primitive for:

$$
\boxed{\text{“what I believe” vs “what I just observed.”}}
$$

without creating a separate belief representation.

---

# 13. Adaptive refresh eventually, but not sample-level branching

Once the 8/6/4/2 frontier is known, we can consider a refresh-confidence score:

$$
\rho_r=f(h_r,e_r,h_{r-1}).
$$

Conceptually:

$$
\rho_r\rightarrow\text{“cached evidence is stale.”}
$$

But do not immediately implement:

```text id="5ksj7b"
if rho > threshold:
    run attention
else:
    skip
```

per token.

That would repeat the dynamic-routing hardware mistakes.

Instead:

### Training

Use fixed schedules.

### Inference

Potentially bucket whole sequences or batches into:

* refresh now
* reuse evidence

Only if profiling shows the branching pays.

Architecture can be adaptive mathematically without making CUDA irregular.

---

# 14. Multi-rate recurrence

If the frontier supports it, the final architecture becomes a **multi-rate recurrent system**.

Example if 6/8 wins:

```text id="z5ntwg"
Iteration 1: ADDRESS + UPDATE
Iteration 2: ADDRESS + UPDATE
Iteration 3: UPDATE using cached evidence
Iteration 4: ADDRESS + UPDATE
Iteration 5: ADDRESS + UPDATE
Iteration 6: UPDATE using cached evidence
Iteration 7: ADDRESS + UPDATE
Iteration 8: ADDRESS + UPDATE
```

So expensive evidence acquisition runs at one frequency, while state evolution runs at another.

This is substantially less radical than BDH-Δ but still genuinely new.

---

# 15. Preserve the original computation on cached rounds

The cached-evidence crux taught us another important thing.

When an address refresh is skipped, don't replace BDH's normal state computation with a new MLP.

Use the existing:

* `encoder`
* `encoder_v`
* ReLU
* multiplicative interaction
* rank-64 decoder
* gated residual

with cached \(e\).

That gives us cheap computation **inside the representation BDH already understands**.

This is much safer than:

$$
\text{cached evidence}\rightarrow\text{new Think Cell}.
$$

---

# 16. Speed architecture: design for a tiny number of big kernels

This stays central.

The architecture should intentionally map to:

$$
\text{large dense GEMM}
+
\text{large dense GEMM}
+
\text{fused reductions}
$$

rather than many conceptual kernels.

We already saw:

$$
1511\rightarrow109
$$

elementwise kernels and:

$$
2.21\times
$$

speed from compilation alone.

That is too large to treat implementation geometry as secondary.

---

# 17. Fuse same-input projections

Where dependencies permit:

$$
XW_1,\quad XW_2
$$

becomes:

$$
X[W_1|W_2].
$$

Candidates include compatible portions of:

* encoder
* encoder_v
* control statistics projection
* any future gate projections

The adaptive controller should preferably consume statistics already produced by the main kernels.

Don't introduce three tiny GEMMs just to decide a scalar gate.

---

# 18. Cached-round specialized kernel

If the frontier promotes reduced refresh cadence, there should eventually be **two compiled round types**:

### Refresh round

```text id="laa1xs"
encoder
exact attention/address
value path
decoder
adaptive residual
```

### Cached round

```text id="7oa4jw"
encoder
reuse evidence
value path
decoder
adaptive residual
```

The cached round should be substantially cheaper.

That gives us a predictable static execution pattern suitable for compile/kernel specialization.

---

# 19. Static schedules first

Possible schedule forms:

### Uniform

$$
\{1,3,5,7\}
$$

### Front-loaded

$$
\{1,2,3,5,7,8\}
$$

### Back-loaded

$$
\{1,3,5,6,7,8\}
$$

### Boundary-heavy

$$
\{1,2,4,6,7,8\}
$$

The 4/8 experiment only tests frequency, not necessarily optimal placement.

Because state changes may be largest early, refresh placement could matter.

But test this **after** the count frontier.

---

# 20. Revisit the depth curriculum with refresh curriculum

Training shouldn't necessarily start at the final sparse-refresh schedule.

A natural curriculum is:

```text id="g4e2as"
early training:
8/8 refresh

middle:
7/8 or 6/8

late:
target cadence
```

This is analogous to the successful depth curriculum.

Early training gets maximum fresh evidence while the representation is forming.

Later training learns to operate with stale evidence.

That might recover some of the 4/8 quality gap.

This is much more principled than asking freshly initialized compressed machinery to learn everything simultaneously.

---

# 21. Train for variable refresh count

Eventually sample:

$$
K\sim\{4,6,8\}
$$

during training.

But unlike the failed variable-R reasoning experiment, here there is already direct evidence that K is a useful compute knob.

This could produce one checkpoint with selectable modes:

```text id="x2h71v"
quality mode     8/8
balanced mode    6/8
speed mode       4/8
```

That would be genuinely useful for deployment.

---

# 22. Re-test reasoning only after recurrence dynamics improve

Don't immediately put synthetic world-model losses back in.

First require the architecture itself to show:

$$
A(R=1)<A(R=2)<A(R=4)<A(R=8)
$$

on genuinely multi-hop tasks.

Or at least:

$$
R_{\text{optimal}}
$$

should increase with task difficulty.

If full-state adaptive gating/delta recurrence changes the curve, then revisit reasoning objectives.

If it still doesn't, we have stronger evidence that BDH recurrence is fundamentally refinement rather than sequential reasoning.

---

# 23. Add a convergence diagnostic, not a training target

Track:

$$
\Delta_r=\|h_r-h_{r-1}\|
$$

$$
\cos(h_r,h_{r-1})
$$

$$
\|y_r\|
$$

$$
g_r
$$

and possibly evidence disagreement.

On ordinary LM and reasoning tasks.

We want to see whether:

$$
\Delta_r\rightarrow0
$$

as recurrence progresses.

Current BDH appears to have a preferred finite operating depth.

vNext should ideally exhibit controlled settling rather than late-depth destruction.

---

# 24. Full-state persistent carry: postpone

The previous cross-token carry stayed near zero.

So remove it for now.

Not permanently killed, but there is no reason to complicate the architecture until within-token recurrence itself works.

If revisited later, carry a projection of the existing full state rather than constructing a separate world-model state.

---

# 25. No MoE, no router, no sparse execution

Still hard no.

The architecture should remain:

$$
\boxed{\text{dense, regular, predictable}}
$$

because every attempt to exploit apparent sparsity has run into one of:

* insufficient stable support
* poor candidate recall
* slow gather/scatter
* GPU underutilization
* quality loss

The architecture should help Tensor Cores, not fight them.

---

# 26. Revised vNext architecture

The architecture now looks like:

```text id="k67fg7"
TOKENS
  │
  ▼
embedding / cached positional preparation
  │
  ▼
FULL D=2496 STATE
  │
  │
  ├───────────────────────────────────────────────────────┐
  │                                                       │
  ▼                                                       │
EXACT ADDRESS REFRESH                                     │
  │                                                       │
  ▼                                                       │
fresh evidence e                                          │
  │                                                       │
  ▼                                                       │
existing BDH value/write computation                      │
  │                                                       │
  ▼                                                       │
adaptive state-conditioned gate                           │
  │                                                       │
  ▼                                                       │
h ← LN(h + g(h,e,state_stats) · update)                   │
  │                                                       │
  ▼                                                       │
cached iteration? ───── yes ─► reuse e ───────────────────┘
  │
  no / scheduled refresh
  │
  └──────────────► exact address again

after R updates
  │
  ▼
rank-64 decoder path
  │
  ▼
logits
```

That's much simpler than old BDH-Δ.

And importantly, nearly every arrow is backed by something we've measured.

---

# 27. Revised principles

### Principle 1

$$
\boxed{\text{Exact addressing is valuable, but its frequency is negotiable.}}
$$

### Principle 2

$$
\boxed{\text{Preserve BDH's internal coordinate system.}}
$$

### Principle 3

$$
\boxed{\text{Change state dynamics before changing state representation.}}
$$

### Principle 4

$$
\boxed{\text{Controlled writes beat aggressive writes.}}
$$

### Principle 5

$$
\boxed{\text{Compression belongs after selection, not before it.}}
$$

### Principle 6

$$
\boxed{\text{Dense regular compute beats theoretically sparse irregular compute.}}
$$

### Principle 7

$$
\boxed{\text{Weight-tied computation remains the path to compute-per-parameter scaling.}}
$$

### Principle 8

$$
\boxed{\text{Hardware geometry is part of the architecture.}}
$$

---

# 28. Immediate experimental sequence

The next experiments now have a very clear order.

### A. Finish refresh frontier

$$
8,\quad6,\quad4,\quad2
$$

Measure:

* val loss
* training tok/s
* compiled tok/s
* decode tok/s
* wall-clock
* `g1`

This identifies the evidence-lifetime/refresh knee.

### B. Adaptive gate on full state

Replace:

$$
g_1=\text{global scalar}
$$

with:

$$
g_r=f(\text{current state statistics})
$$

initialized to 0.58.

Compare against the single-gate champion at 8/8 first.

### C. Stability test

Evaluate:

$$
R=2,4,8,12,16
$$

on LM loss and reasoning probes.

The question:

> Does adaptive gating prevent late-depth collapse?

### D. Combine only after B wins

Then test:

$$
\text{adaptive gate}
+
\text{best reduced-refresh schedule}.
$$

No bundled experiments before isolated wins.

### E. Compile/profile final candidate

Then attack:

* remaining GEMM utilization
* packed projections
* graph breaks
* refresh/cached specialized kernels

---

# 29. Success criterion

The real vNext target isn't merely lower loss.

We want something like:

$$
\boxed{
\begin{array}{c}
\text{quality}\leq\text{current champion}\\
\text{training throughput substantially higher}\\
\text{decode throughput substantially higher}\\
\text{stable recurrence beyond }R=8\\
\text{harder tasks benefit from more compute}
\end{array}
}
$$

Even getting the first three would already be a major architectural win.

The latter two would turn it into the more ambitious reasoning architecture we're trying to build.

---

# 30. New one-sentence architecture thesis

The old BDH-Δ idea was:

> retrieve sparsely and reason in a new compressed latent world model.

The revised vNext is:

$$
\boxed{
\textbf{Keep BDH's full exact representation, refresh evidence only as often as needed, and make each recurrent write a small state-dependent correction implemented as dense compiler-friendly computation.}
}
$$

That is much more consistent with what the experiments have actually taught us. 🐉
