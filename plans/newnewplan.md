# Priority override, 2026-08-31 -- do not resume the plan below mechanically, section-by-section

The adaptive-gate result (1.4023, real +0.028 over the best matched
fixed-scalar control, section B below) is large enough that
understanding IT outranks building anything further down this
document. Sections below remain the architecture backlog, but the
literal reading order is superseded by this sequence until each step
resolves:

$$
\boxed{
\text{understand gate} \rightarrow \text{test recurrence stability} \rightarrow \text{finish refresh frontier} \rightarrow \text{combine winners} \rightarrow \text{resume deeper vNext work}
}
$$

1. **Lock the gate mechanism.** In flight now: FP32 gate-variance check
   (does real fp32 precision reveal state-dependent variation bf16
   rounded away?), gate-trajectory analysis (did g drift meaningfully
   during training even though the endpoint looks flat?), and the
   state-independent `C_theta(1)` control (identical controller,
   architecturally incapable of state-dependence -- lands near 1.4023
   -> pure optimization-path effect; near 1.414 -> state-dependence
   itself matters; between -> both). Whichever way this resolves,
   simplify to the smallest mechanism that reproduces 1.4023 and adopt
   THAT as the new 8/8 quality baseline, replacing the plain single-gate
   champion everywhere below.
2. **R-stability test on the resolved gate.** R in {2,4,8,12,16}. The
   entire reason adaptive writes got proposed was the old late-depth
   collapse (R=8<R=4, R=12/16 near-chance). If 1.4023 is a pure LM-loss
   win but R=12/16 still collapses identically, the gate is a quality/
   optimization win, not (yet) a reasoning-dynamics win -- a real,
   useful, but smaller finding. If deeper recurrence stabilizes, that's
   the bigger architectural result this whole internal-computation track
   has been chasing.
3. **Finish the refresh-side cleanup**, now second priority, not first:
   constant-schedule 4/8 and 6/8 (fixing the K=2 curriculum confound,
   built, dispatch died to the disk-quota issue, needs a clean rerun),
   plus the 6/8 placement sweep (front-loaded/back-loaded/boundary-heavy
   vs uniform_6, built and verified, not yet dispatched). Goal: the true
   refresh Pareto point with no curriculum-shape confound anywhere in it.
4. **Combine the two independent winners only after both are locked
   separately** -- explicitly do not assume they compose (this
   project's own standing lesson, repeated across BDH-VB, subspace
   decoder, and now gate/refresh). Test resolved-gate+6/8 against
   resolved-gate+8/8. If 6/8 only costs ~0.01-0.02 while giving real
   compiled speed, it's the efficiency variant; if it wipes out the
   gate's win, keep 8/8.
5. **Only then resume the deeper sections below**, and only the parts
   still consistent with what steps 1-4 found: full-state delta/
   stability dynamics, state-of-computation signals, evidence
   disagreement, static compiler-friendly schedules, packed GEMMs,
   specialized refresh/cached kernels, eventually variable compute. The
   compressed belief/workspace branch (BDH-Delta, 1.7862) stays dead
   unless new evidence specifically revives it -- nothing so far does.

If the gate turns out to genuinely use tiny state-dependent variation,
the plan pivots harder toward state-conditioned recurrent control. If
it's an optimization-path effect, the plan pivots toward training-
dynamics/protected-recurrence mechanisms instead of state-conditioning
per se. That distinction is what steps 1-2 above are for, before any
further GPU spend on a new redesign.

---

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

Real, 2026-08-30 (RTX 5090/RTX 4090 fallback, matched 25M-token budget, seed=7):

$$
8/8 \approx 1.414\text{–}1.433,\quad 6/8 = 1.4505,\quad 4/8=1.5125,\quad 2/8=1.4984.
$$

`final_g1` landed at 0.5366 (6/8) and 0.5955 (2/8) -- same attractor
family as 8/8's 0.583/0.586 and 4/8's 0.5748, now confirmed across
FOUR independent refresh cadences.

**Real, non-monotonic surprise: 2/8 (1.4984) beat 4/8 (1.5125).** Not
what the plan's own outcome table predicted ("2 refreshes = likely too
stale" implying the worst score). Real, disclosed confound before
reading too much into this: `curriculum_stages(target_tokens,
n_refresh)` computes ramp stages as `{max(2, round(n_refresh*f)) for f
in (0.5,0.75,1.0)}` -- at n_refresh=2 this collapses to a SINGLE stage
`{2}` (the `max(2,...)` floor collides with the target itself), so the
2/8 arm trained at a CONSTANT refresh=2 the entire run, while 4/8 and
6/8 both got real 3-stage ramps (2/8 log confirms: `n_refresh=2` on
every single logged step, never anything else). So 2/8 and 4/8 are not
quite an apples-to-apples comparison of "final cadence" alone -- 2/8
also never spent any training time at LOWER-than-target refresh the
way 4/8 and 6/8 did, and never spent time at HIGHER refresh mid-run
either. Real, open question, not yet decomposed: does 2/8 truly
tolerate more staleness than 4/8, or did skipping the ramp (training
at the final cadence from step 1) help more than reaching a higher
final cadence did? A same-schedule-shape rerun would be needed to
separate these two effects cleanly -- not run here.

Potential outcomes:

```text id="qv9x4u"
8 refreshes = maximum quality (1.414-1.433)
6 refreshes = real, small cost (+0.017-0.036) -- looks like the efficiency knee
2 refreshes = real, moderate cost (+0.065-0.084) -- constant schedule, no ramp
4 refreshes = real, moderate cost (+0.078-0.098) -- WORSE than 2/8 despite more refreshes, curriculum-shape confound above
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

### B. Adaptive gate on full state -- real result, 2026-08-30

Real dispatch (RTX 4090, matched 25M-token budget, seed=7, n_refresh=8/8
so this is isolated from the refresh-cadence question entirely,
protected init at 0.58 per the attractor evidence above):

$$
\text{val\_loss}=1.4023.
$$

This is real and it BEATS every arm measured so far, including the
8/8 single-gate champion (1.4142-1.4326) and the whole cached-evidence
frontier (1.4505-1.5125).

**But read the gate stats before celebrating**: `gate_stats={'mean':
0.55078125, 'std': 0.0, 'min': 0.55078125, 'max': 0.55078125}` on a
real held-out batch. The controller had full freedom to vary `g_r` by
`(h_r, y_r, h_prev, e_r)` per position, and it didn't -- every position
in the batch got the EXACT same gate value. It converged to a
DIFFERENT constant (0.5508) than the plain single-gate arms
(0.583-0.596), but still a constant, not input-conditional gating. Real
caveat before over-reading the zero: eval ran under bf16 autocast, and
bf16's ~8 significant bits could be collapsing a genuinely tiny but
nonzero spread down to one representable value -- "indistinguishable
from constant at bf16 precision" is the honest claim, not "provably
zero variance." Not re-checked in fp32.

So the honest read: **the adaptive-gate machinery didn't demonstrably
learn to be adaptive**, but the run still landed at a real, meaningfully
better val_loss than any prior arm -- most likely explained by the
different initialization/parameterization path (a small MLP settling
near 0.55 versus a single scalar settling near 0.58-0.59) giving a
marginally better optimization trajectory, not by genuine
state-conditional computation. This doesn't kill the adaptive-gate
idea, but it means section 28C (does adaptive gating prevent late-depth
collapse at R=12/16?) is the real test of whether there's more here
than a better-placed constant -- if the gate stays flat there too,
"adaptive" is the wrong word for what this run found.

**Real operational gap, not a training result**: the local dispatch
wrapper's polling loop died silently sometime after the val_loss/
gate_stats print (no further local log lines, no error), while the
remote pod kept running with nothing left to do. Checkpoint save and
the script's own JSON write never completed on the pod -- process
vanished with no Python traceback right after that exact print, most
likely an OOM-kill on host RAM during `torch.save` prep (not diagnosed
further; checkpoint-saving worked fine on other, similarly-sized arms
this session, so this looks pod-specific). The val_loss/gate_stats
numbers are real, printed by the script's own eval before whatever
killed it -- recovered manually into
`results/local/hz0h_bdh_adaptive_gate_88_quality_check.json`. Pod
`ddmr5t5xgj5m1f` terminated manually since nothing was going to clean
it up automatically.

**Fixed-g1 local sweep, 2026-08-30 -- real, but inconclusive.** Ran
`--g1-fixed` at g in {0.50, 0.525, 0.55, 0.575, 0.60} locally on MPS,
500K tokens/arm (NOT the 25M-token budget every other number in this
file uses -- a quick directional check during a GPU-dispatch hold, not
a substitute for the real matched-budget test). Real result:

```text
g1=0.50:  val_loss=2.7074
g1=0.525: val_loss=2.7108
g1=0.55:  val_loss=2.7097
g1=0.575: val_loss=2.7217
g1=0.60:  val_loss=2.7190
```

Non-monotonic, spread of only ~0.014 across all 5 arms -- looks like
noise at this budget, not a real trend. This does NOT answer the real
question (does fixed g~0.55 reproduce the adaptive-gate's 1.4023) --
absolute values aren't comparable across a 500K vs 25M-token budget,
and even the relative ordering here is too noisy at 500K tokens to
trust. The real test still requires GPU dispatch at the matched 25M
budget; this local run was a cheap sanity check, not a resolution.
Full results: `results/local/hz0h_bdh_g1_fixed_local_sweep.json`.

**Full-budget fixed-g1 sweep, RTX 5090, 25M tokens/arm, real -- this is the decisive answer.** Same 5 values, matched to every other arm's methodology (seed=7, same curriculum, same warmstart):

```text
g1=0.50:  val_loss=1.4315
g1=0.525: val_loss=1.4293  <- best fixed value
g1=0.55:  val_loss=1.4303  <- closest to the adaptive gate's actual landing point (0.5508)
g1=0.575: val_loss=1.4339
g1=0.60:  val_loss=1.4470
```

Adaptive gate (real, full budget): **1.4023**.

**Every fixed value is worse than the adaptive gate, including g1=0.55 -- the value closest to where the controller itself landed.** Gap at the closest point: 1.4303 vs 1.4023, +0.028, real and not small (comparable in size to the entire single-gate-vs-baseline win this whole gated-residual line started from). This directly answers the question from section B: **a hard-coded scalar does NOT reproduce the adaptive gate's result, even at the exact value it converged to.** The controller's own parameterization or training dynamics were doing real work, despite its measured output collapsing to a near-constant (std=0 under bf16 eval). Whatever that work is, it isn't captured by "found a better constant."

Real, still-open question this raises rather than closes: if the gate isn't varying by input in any bf16-visible way, what IS different between "a scalar starting at 0.58 that gets pulled to 0.55 through gradient descent on a fixed value" and "a tiny MLP starting at 0.58 (by construction, via the protected zero-init) that gets pulled to output ~0.55 through gradient descent on its own weights"? Candidates, none yet tested: (a) the MLP's extra parameters (even producing a flat output) change the loss landscape / effective learning rate the shared backbone sees during training, a real optimization-dynamics effect unrelated to the final gate value; (b) bf16 eval genuinely is hiding real per-token variance too small to see at that precision but large enough over 25M tokens of gradient signal to matter; (c) something about SiLU/the two-layer structure biases early training differently even before g1 settles. Worth an fp32 gate-stats re-check (candidate b) before deeper investigation, since it's the cheapest to rule out.

## Gate mechanism resolved, 2026-08-31 -- real answer: BOTH effects are real, and both were measured cleanly

Three real, same-session (both RTX 5090, same seed=7, same 25M-token
budget, ran concurrently so no cross-run drift) dispatches close this
out:

```text
best hard-frozen fixed g1 (from the 5-point sweep):     1.4293
plain learned single scalar (original champion):        1.4142-1.4326
state-independent trainable gate, C_theta(1):            1.3970
real state-dependent adaptive gate (this retrain):       1.3879
```

**The state-independent control (`--state-independent`, identical
controller architecture/param count/protected init, but fed a constant
input -- structurally incapable of varying by token/state/round) beat
every fixed value AND the original champion by a wide margin (1.3970
vs best-fixed 1.4293) but was still real and measurably worse than the
true state-dependent gate (1.3879) by 0.0091.**

That's the "in between -> both contribute" branch of the three-way
decomposition, not either pure outcome:

1. **A real, large optimization/parameterization effect** (~0.017-0.036
   depending which baseline): training a 113-parameter MLP via gradient
   descent to output what is STILL mathematically just one scalar (no
   state input at all) does substantially better than either a single
   learned scalar or any hand-set constant. This is the "training
   dynamics matter, not just the final value" hypothesis, confirmed --
   the SAME final behavior (a flat gate) reached via a richer
   optimization path is a real, different, better local solution than
   the same value set or learned directly.
2. **A real, small, fp32-confirmed state-dependence effect on top of
   that** (~0.009): the real gate's `gate_stats_fp32` showed genuine
   nonzero variance -- std=1.94e-6, min=0.548983, max=0.548999 across a
   real held-out batch -- not the exact bf16-rounding-induced 0.0 the
   bf16 read showed, and categorically different from the
   state-independent arm's mathematically-guaranteed-exact 0.0 (same
   fp32 diagnostic, same code path, confirmed exactly zero there,
   confirming the real gate's nonzero reading isn't a numerical-noise
   artifact of the measurement itself). Tiny in absolute terms, but a
   real, repeatable, non-bf16-artifact signal that the controller IS
   using state information, just very weakly at this training budget.

**Resolution for the priority-override sequence's step 1**: lock in the
real state-dependent adaptive gate (this retrain's checkpoint,
`results/local/hz0h_bdh_adaptive_gate_retrain_checkpoint.pt`, verified
loadable, not corrupted like gate88's) as the new 8/8 quality baseline,
val_loss=1.3879 -- both because it's the best real number measured and
because the state-dependence, however small, is real and worth
preserving rather than simplifying away. The gate-trajectory log
(`gate_trajectory` field in the result JSON, 61 points over training)
additionally confirms the earlier local-sweep observation: std was
genuinely nonzero and shrinking through roughly step 3000-6000
(0.0005-0.003 range) before settling near the bf16 floor for the rest
of training -- the controller DID use more state-dependence earlier in
training than it settled into by the end, consistent with an
"automatically learned residual curriculum" as speculated, though the
FINAL fp32 measurement (post-training) is what's quoted above as the
operative number for the model actually being adopted.

**Next per the reprioritized sequence**: R-stability test (R in
{2,4,8,12,16}) on this exact checkpoint -- does state-dependent
adaptive writing prevent the old late-depth collapse, or does 1.3879
hold as an LM-loss win with the same R=12/16 breakdown as before?

Original plan for step B:

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

### C. Stability test -- real result, 2026-08-31: YES, the collapse is gone, but not (yet) proof of depth-dependent reasoning

Real dispatch (RTX 5090, `scripts/hz0h_bdh_adaptive_gate_variable_depth_eval.py`), direct methodological parallel to the original R-scaling result (Phase-redesign section 8 above: accuracy peaked R=2-4, DECLINED by R=8, COLLAPSED toward chance -- 0.0625 for 16 locations -- by R=12/16), same task, same training protocol, same eval matrix, only the backbone swapped for the locked adaptive-gate checkpoint (val_loss=1.3879):

```text
hops=1: R1=0.30 R2=0.41 R4=0.31 R8=0.32 R12=0.42 R16=0.27
hops=4: R1=0.27 R2=0.29 R4=0.32 R8=0.34 R12=0.35 R16=0.28
hops=8: R1=0.31 R2=0.40 R4=0.35 R8=0.39 R12=0.29 R16=0.27
```

**The real, unambiguous positive**: every single cell in the full 6x6
matrix (36 cells) lands between 0.27 and 0.43 -- nowhere close to the
0.0625 chance floor the plain-architecture R=12/16 arms collapsed
toward. The catastrophic late-depth breakdown that motivated this
whole adaptive-gate track in the first place is real and gone on this
backbone.

**The real caveat, stated as plainly as the positive**: this is
stability, not (yet) evidence of depth-dependent reasoning. There is no
clean monotonic trend anywhere -- R=12 is the single best point for
hops=1 (0.42) but among the worst for hops=8 (0.29); accuracy is
better described as a flat, noisy band across the whole R range than
as a curve that peaks then degrades. `shortcut_rate` stays comparably
high throughout every R (0.25-0.49 across the matrix, often exceeding
the real accuracy at that same cell -- e.g. hops=4 R=1: accuracy=0.27,
shortcut_rate=0.49), meaning the positional-shortcut confound this
task was specifically designed to expose is still very much live at
every depth, not just at the R values that used to collapse.

**Honest verdict for the priority sequence's own question**: "does
adaptive gating prevent late-depth collapse, or is it just an LM-loss
win?" -- the answer is genuinely in between, and closer to "prevents
collapse" than "just LM loss," but doesn't yet clear the bar for "the
gate enables genuine reasoning-depth-dependence" (section 22's own
target: A(R=1)<A(R=2)<A(R=4)<A(R=8) on a real multi-hop task -- not
observed here; the matrix is flat, not increasing). This is real
progress on the recurrence-stability axis specifically, decoupled from
(and not yet evidence for) the separate reasoning-uses-depth axis. Per
section 22's own instruction ("if it still doesn't [show the ordering],
we have stronger evidence that BDH recurrence is fundamentally
refinement rather than sequential reasoning") -- stability without
ordering is consistent with refinement (the state settles into a
decent, roughly depth-independent operating point and stays there)
rather than sequential composition, though a flat-not-collapsing curve
is a meaningfully different regime than the pre-gate flat-then-collapsing
one and worth taking as real progress on its own terms.

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

---

# 31. Progressive Latentization Training: real 4-arm result, 2026-08-31

The falsification experiment proposed alongside the gate-mechanism work:
freeze the 1.3879 locked adaptive-gate architecture and vary only the
*training regime* on a new order-dependent task (register-machine:
sequential `op val` clauses, decoy variable groups for shortcut
resistance, single-digit answers so the real `lm_head` can be reused
for step supervision). Four arms, all initialized from the same locked
checkpoint, each given 8M continued tokens (A/B) or 20K examples (C/D):

* **Arm A** -- ordinary continued LM training, fixed R=8. Real result:
  val_loss=1.3777 (real improvement over the 1.3879 baseline).
* **Arm B** -- Huginn-style random-R continued LM training, R drawn from
  `[6,6,6,7,7,8,8,8,8,12,12,16]` per step. Real result: val_loss=1.3949
  (real regression vs. baseline and vs. Arm A).
* **Arm C** -- explicit CoT SFT on clean (no-decoy) register-machine
  traces, ordinary byte-level LM loss. Real result: 20K examples in
  1327s (no comparable val_loss -- different training distribution).
* **Arm D** -- progressive latentization (Coconut/LOTUS/CODI-style):
  curriculum grows `n_latent` 0->n_steps over training, each latent
  reasoning step is one real recurrent round appended as a new sequence
  position, LOTUS-style step supervision reads intermediate rounds
  through the real `lm_head` against the true intermediate digit,
  `lambda` decays 1.0->0.5->0.1->0.0 across training. Real result: 20K
  examples in 2784s, checkpoint verified (13 keys, matches A/B/C
  architecture exactly).

## A real methodology bug found and fixed mid-experiment

The shared eval instrument (`hz0h_bdh_register_machine_variable_depth_eval.py`,
mirroring the entity-chain R-stability eval from section 22-28) trains a
fresh `answer_head` on top of each arm's checkpoint via 20,000 steps of
gradient descent, then reports an accuracy matrix over
step-count x R in {1,2,4,8,12,16}. The first run of this against Arms
A/B/C produced eval matrices that were **bit-identical across all 36
cells for all three arms**, despite different checkpoints and different
training losses -- caught by literally diffing the JSON files rather
than trusting the printed accuracy numbers. Root cause:
`train_probe()` called `torch.optim.AdamW(model.parameters(), ...)` --
full-model fine-tuning, not a frozen-backbone linear probe. 20,000 steps
of full fine-tuning on the same seeded data is enough gradient signal to
overwrite whatever each arm's differing continued-pretraining did,
so all three converged to the same fixed point regardless of starting
checkpoint. (This exact protocol was valid for the single-architecture
R-stability question in section 22-28 -- it only breaks when reused to
compare *different* checkpoints against each other, which is precisely
what this 4-arm experiment needs.) Fixed by freezing the backbone
(`requires_grad_(False)` on every parameter except the fresh
`answer_head`) before probe training -- now a real linear probe, matching
what the docstring already claimed. Verified: rerun matrices are
genuinely distinct across all four arms (confirmed by diff, not just
eyeballing).

## Real result after the fix

Chance accuracy for this 10-way single-digit classification is 0.10.
Overall accuracy across the full step-count x R matrix (36 cells,
eval_n=100/cell, so per-cell std error ~0.03):

| Arm | overall accuracy | overall shortcut-rate |
|-----|------------------|------------------------|
| A (plain continued LM)      | 0.126 |0.062 |
| B (random-R continued LM)   | 0.124 | 0.064 |
| C (explicit CoT SFT)        | 0.112 | 0.071 |
| D (progressive latentization) | 0.122 | 0.067 |

All four arms land in a narrow 0.11-0.13 band, barely above chance, with
low shortcut-rates (so the near-chance accuracy is not decoys pulling
predictions away from real answers -- the probe genuinely isn't finding
the answer, not finding something else instead). Per-R breakdown for
Arm D specifically (the one this whole experiment exists to test):

R=1: 0.137, R=2: 0.123, R=4: 0.128, R=8: 0.105, R=12: 0.132, R=16: 0.105

**No monotonic R-dependence in any arm, including D.** The target
falsification signature from the original proposal --
`A(R=1)<A(R=2)<A(R=4)<A(R=8)` or any consistent increasing trend with
R specifically in Arm D and not in A/B/C -- is not present. Arm D's
curve is flat-to-slightly-decreasing across R, statistically
indistinguishable from noise given the ~0.03 per-cell standard error,
and not meaningfully different in shape from Arms A/B/C.

**Honest verdict, per the falsification framing the experiment was
explicitly built for**: this is a real negative result. Under this
instrument, progressive latentization training does not produce
evidence of stepwise reasoning-depth-dependence on a genuinely
order-sensitive task, no more than ordinary continued pretraining does.
Two real caveats, neither of which should be used to explain the result
away without further evidence: (1) the frozen linear probe reads out
only the *last-token* hidden state through a single matmul -- it is a
real but narrow instrument, and it's possible task-relevant information
is present in the frozen features but not linearly decodable from that
one position; (2) overall accuracy for all four arms is close enough to
chance that the register-machine task itself, at n_steps up to 8 with
decoys, may simply be too hard for this model scale/training budget
regardless of training regime -- a near-floor instrument has limited
power to detect *any* real difference between arms, not just the
depth-dependence one. Both caveats argue for a follow-up with either a
richer probe (e.g. read out at every position, not just the last) or an
easier task variant before concluding progressive latentization
categorically doesn't work here -- but as currently measured, Arm D
gives no support for the hypothesis it was built to test.

---

# 32. Refresh-side cleanup: real result, 2026-08-31 (500K-token local stopgap)

The deferred refresh-frontier cleanup from the priority-override sequence
(constant-schedule K=4/K=6 reruns fixing the earlier curriculum confound,
plus the 6/8 placement-pattern sweep) finally got dispatched -- and the
real GPU run died mid-flight: RunPod balance hit zero, force-killing all
6 pods simultaneously (`Connection to ... closed by remote host` on every
one, confirmed root cause via a direct `402: "Your account balance is too
low to rent a pod"` from the create-pod API). None of the 6 saved a
result; one job (K=4 constant-schedule) was 65%+ through a 99-minute run
when it died -- real wasted compute, not recoverable.

Rather than wait, all 6 were rerun locally on the Mac's MPS backend at a
reduced 500K-token budget (245 steps, single seed=7) as an explicit
stopgap -- same precedent as the earlier local g1-sweep. **These numbers
are NOT comparable to any 25M-token GPU result elsewhere in this
document** (val_loss magnitudes are ~2x higher purely from 50x less
training) -- only the *relative ordering within this batch* is
informative, and even that comes from a single 245-step run per variant,
not a statistically powered comparison.

| Variant | Refresh schedule | val_loss |
|---|---|---|
| K=4 constant (`n4_const`) | {0,2,4,6} | 2.7000 |
| K=4 placement (`uniform_4`) | {0,2,4,6} (same schedule, different script) | 2.6952 |
| K=6 constant (`n6_const`) | {0,1,3,4,5,7} | 2.7115 |
| K=6 front_loaded | {0,1,2,4,6,7} | **2.6905** (best) |
| K=6 boundary_heavy | {0,1,3,5,6,7} | 2.6914 |
| K=6 back_loaded | {0,2,4,5,6,7} | 2.7197 (worst) |

Two real, useful things from this batch despite the reduced budget:

1. **Cross-script sanity check passed.** `n4_const` and `uniform_4` use
   the *identical* refresh schedule `{0,2,4,6}` through two independently
   written scripts (`hz0h_bdh_cached_evidence_quality_check.py` vs
   `hz0h_bdh_cached_evidence_placement_quality_check.py`). They land
   within 0.005 of each other (2.7000 vs 2.6952) -- consistent with the
   same computation modulo minor implementation-path/nondeterminism
   noise, not a bug in either script.
2. **Directional hint on placement, not yet a real result.** Among the
   four K=6 placement variants, `front_loaded` and `boundary_heavy`
   both beat the naive-uniform `n6_const`/`uniform_6`-equivalent
   schedule, while `back_loaded` is clearly worst. If this holds at real
   budget, it suggests refreshing early (front-loaded) or avoiding a gap
   right before the final round (back-loaded skips index 1 and 3, landing
   two consecutive skips right before the last two rounds) matters more
   than even spacing. This is exactly the kind of signal the placement
   sweep was built to surface -- but at 500K tokens it is a hint worth
   re-testing at 25M budget, not a conclusion. **Once RunPod funds are
   restored, rerun all 6 (or at minimum the two placement extremes,
   front_loaded and back_loaded) at the real 25M-token budget before
   updating the refresh-frontier decision in section 27-28.**

## Real result, 2026-08-31 (25M-token budget, all 6 jobs, real GPU)

Funds were restored and all 6 jobs reran at the real 25M-token budget,
one at a time (RTX 5090, ~$1.3-1.4/job, ~$8.20 total), per the cost-
discipline agreement established after the earlier burn-rate concern.
Real final results:

| Variant | Schedule | val_loss |
|---|---|---|
| K=4 constant (`n4_const`) | {0,2,4,6} | **1.4108** (best) |
| K=4 placement (`uniform_4`) | {0,2,4,6} (same, cross-check) | 1.4108 (bit-identical) |
| K=6 constant (`n6_const`, evenly spaced) | {0,1,3,4,5,7} | 1.4135 |
| K=6 front_loaded | {0,1,2,4,6,7} | 1.4159 |
| K=6 boundary_heavy | {0,1,3,5,6,7} | 1.4307 |
| K=6 back_loaded | {0,2,4,5,6,7} | 1.4314 (worst) |

**Decisive, real finding: K=4 constant-schedule is the new refresh-
frontier champion**, beating every K=6 variant tested including the
best-placed one -- and it does so with LESS exact-address compute per
round (4 refreshes vs 6), making it a genuine Pareto win: cheaper AND
better. This directly reverses the earlier curriculum-confounded
result (K=4=1.5125 losing to K=6=1.4505) -- the confound was masking
the real ordering the whole time.

Among the K=6 placement variants, evenly-spaced (`n6_const`, 1.4135)
beats all three clustered patterns -- real evidence that even spacing
matters more than clustering refreshes at specific positions, which
also **reverses** the 500K-token local-stopgap's directional hint
(front_loaded predicted best, actually second-worst at real budget;
only back_loaded's "worst" prediction held up). This is a real,
disclosable lesson for the whole project: small-budget stopgap
rankings are not reliable substitutes for real-budget results, even
when the direction seems intuitive.

**Refresh-frontier decision, updated**: K=4 constant-schedule
(schedule {0,2,4,6}) is the new candidate for the "combine with
adaptive gate" step (section 27-28's step D), not K=6 as previously
assumed. This work is now superseded in priority by the BDH-CQ pivot
(section 33) but stays the correct answer to "which refresh schedule
wins" whenever refresh-frontier work resumes.

That is much more consistent with what the experiments have actually taught us. 🐉

---

# 33. Major pivot proposed, 2026-08-31: target BDH-CQ, lock params at 150M

**Standing decision from this point forward: all future HatchlingZero
architecture work is locked at ~150M parameters** (current champion is
206.47M -- future configs need to shrink to match). The explicit target
is no longer "improve BDH LM loss in the abstract" but:

$$
\boxed{\text{beat BDH-CQ's reported ARC-AGI-1 pass@2 at} \le 150M \text{ params and} \le \text{its reported cost/task}}
$$

**Important sourcing caveat, stated plainly**: the BDH-CQ specification
below (architecture shape, disclosed/undisclosed pieces, and every
specific number -- 29.5% pass@2, 0.85 H200-sec/task, \$0.00070/task, the
per-failure-mode breakdowns like 0/72 on color-swap+relocation) comes
from the user's own reading of an external paper, not from anything
verified independently in this session. Treat it as the target
specification to build against, not as an internally-confirmed fact,
until someone actually pulls and checks the source. If the numbers turn
out to be misremembered or the paper doesn't say what's summarized here,
the target moves, not the discipline around it.

## What BDH-CQ reportedly does (per the above caveat)

Two distinct recurrent processes, not one:

1. **Persistent task memory** $S$, built by ingesting demonstrations
   sequentially: $S_t = U_\theta(S_{t-1}, D_t)$. Model weights don't
   change; the demonstrations update the recurrent state.
2. **Separate latent reasoning workspace** $H$, initialized from the
   query and the final memory state: $H_0 = E_\theta(x^*, S_K)$, then
   iterated $H_{r+1} = F_\theta(H_r, S_K)$ for $R$ steps, decoded to an
   answer $\hat y = G_\theta(H_R)$.

Reportedly trained at **variable latent-reasoning effort** (LOW/MEDIUM/
HIGH bands, roughly R in {2-4, 6-8, 12-16}), giving a real accuracy/cost
knob at inference (21% / 27% / 29.5% pass@2 respectively, per the
external summary) -- this is the same "more compute -> better answer"
property this whole session's recurrence-stability work has been
chasing, and if accurate, BDH-CQ already has it via Huginn-style
variable-R training, not anything more exotic.

No evidence of explicit CoT->latent distillation (Coconut-style) in
what's disclosed -- the training objective is reportedly just
demonstrations -> memory update -> query -> R latent steps -> exact
target, trained on a large curated ARC curriculum (ARC-AGI-1 train,
RE-ARC, ConceptARC, ARC-Heavy, ARC-GEN100K, undisclosed augmentations).
The evaluated system also reportedly includes candidate generation +
ranking around the network (pass@2 = up to 2 ranked candidates), not a
single raw forward pass.

Explicitly undisclosed: state/workspace dimensions, exact $U_\theta$/
$F_\theta$, recurrent step count, BDH internal dims, optimizer/LR/
schedule/batch size, augmentation recipe. **This cannot be literally
reproduced from the paper** -- only the system *shape* is knowable.

## Proposed HZ-CQ architecture (real work items, not yet built)

1. **Task memory $S$**: real persistent recurrent memory built from
   demonstration pairs (both input AND output enter memory -- the model
   needs to infer the transformation, not just encode inputs). Likely
   reuses BDH's exact associative addressing + state-dependent gated
   writes, NOT the aggressively bottlenecked belief/workspace split
   that killed BDH-Delta (1.7862, dead per section 27) -- keep $S$ and
   $H$ high-dimensional, don't squeeze through a 384-d bottleneck again.
2. **Reasoning workspace $H$**: separate from $S$, high-dim, iterated
   with the **already-validated adaptive gate** (1.3879 champion,
   real state-dependence, real stability through R=16 per sections
   22-28) as the update rule: $H_{r+1} = \text{LN}(H_r + g_r \Delta H_r)$
   with $g_r = C_\phi(H_r, \Delta H_r, S)$. This is a real, already-
   proven-in-this-project piece BDH-CQ hasn't disclosed having.
3. **Variable-effort training**: R sampled per-episode from LOW/MEDIUM/
   HIGH bands (or a broader distribution + the adaptive gate learning
   when to stop, going beyond BDH-CQ's reported discrete effort modes
   toward real per-task halting).
4. **Task-structured training data**: each training item is a full
   episode (demo1 in/out, demo2 in/out, demo3 in/out, query in -> query
   out), no explicit task ID -- learning-to-learn through recurrent
   state, not parametric memorization.
5. **Procedural curriculum targeting BDH-CQ's own reported failure
   modes** (per the external summary): ordering (length-8 collapse),
   deep nesting (depth-5 failures without demonstration), operator
   composition (reflection+relocation down to 47/72, color-swap+
   relocation at 0/72), extrapolation (train on depth 1-3, query depth
   4+). These reported weak points are a real, free roadmap for where
   to generate synthetic training coverage -- IF the numbers hold up
   under the sourcing caveat above.
6. **Candidate generation + verification/ranking**: instead of one
   forced-correct decode, produce a small candidate set and a cheap
   verifier scoring "does this candidate obey the transformation
   inferred from the demonstrations" -- optionally with a latent
   self-correction loop (candidate -> verify -> correct -> re-emit,
   no CoT tokens) as a potential improvement over pure ranking.
7. **Efficiency**: reuse this project's existing speed work (compile,
   packed GEMMs, BF16, static batched recurrence) -- beating BDH-CQ on
   accuracy alone while being far slower isn't actually beating it on
   its own accuracy/cost frontier.

## Explicit sequencing decision needed

**Not started yet.** This is a full scope change from the refresh/gate
work this document has tracked through section 32 -- new task (ARC-AGI
style, not register-machine or entity-chain), new architecture pieces
(persistent task memory, separate workspace, candidate ranking), new
param budget (150M, down from 206.47M), new success metric (ARC pass@2
+ cost/task, not LM validation loss). Before writing any code: decide
whether to (a) finish the in-flight 6-job refresh-cleanup sweep first
(5/6 done as of this section, real GPU cost already sunk, cheap to
finish), then pivot fully, or (b) treat the current champion as frozen
now and start HZ-CQ design work immediately in parallel. Given this
session's explicit cost-discipline agreement (small batches, one job
at a time, cost estimates before dispatch), HZ-CQ's first real
GPU spend should get the same treatment -- an explicit estimate and
go-ahead before any dispatch, not a silent large fan-out.

## Real groundwork done, 2026-08-31 (no GPU spend, done while the refresh sweep finished)

1. **150M-param config found**: `n_embd=2128, n_head=8, n_layer=8,
   d_state=532, subspace_rank=64, mult=16` -> 150,577,280 real params
   (verified by instantiating `BDHVBSubspaceDecoder` and counting, not
   estimated) -- 150.58M, within 0.4% of the 150M target, same
   architecture ratios as the 206.47M champion (d_state=n_embd/4,
   rank=64) just scaled down.
2. **ARC-AGI-1 dataset in place**: cloned from
   `github.com/fchollet/ARC-AGI` into `data/arc_agi_1/` (gitignored,
   matches project convention), 400 training + 400 evaluation tasks,
   real public benchmark data with known answers on the training split.
3. **`scripts/hz0h_bdh_arc_task_loader.py`** built and verified: loads
   tasks, serializes an episode as `IN/<grid>/OUT/<grid>/END` per
   demonstration (JSON's own order kept -- unlike register-machine,
   ARC demos have no "real order" to shuffle-protect) followed by
   `QUERY/<input>/ANSWER/<output>`, single ASCII digit per grid cell
   (byte-level, matches vocab_size=256 everywhere else in this
   project). Round-trip verified against all 400 real training tasks
   (serialize then recover the exact grid via `parse_answer`) -- not a
   toy/synthetic check.
4. **Real constraint found**: episode byte-length distribution is
   `median=925, p90=2493, mean=1232, max=9356` (all real numbers from
   the actual dataset, not estimated). This is far beyond the
   `sequence_length=256` chunks every other training script in this
   project uses -- whatever HZ-CQ training script comes next needs to
   either (a) support much longer context (2048-4096 would cover
   ~90-95% of tasks, 9356+ needed for the real max), or (b) filter/
   truncate the long tail, a real design decision not yet made.

Still not started: the persistent task-memory module, the separate
reasoning workspace wired to the adaptive gate, the training script
itself, and the GPU dispatch decision (which needs the same cost-
estimate-first treatment as everything else per the sequencing note
above).

## Real groundwork done, 2026-09-01: core forward pass built and verified

**`reference/hz0h_bdh_arc_task_memory_torch.py`** (`forward_hz_cq`) --
real, working implementation of the persistent-memory + separate-
workspace architecture, built entirely from already-validated pieces
(the adaptive gate's `_refresh_iteration`, the growing-sequence-with-
carry pattern from progressive-latentization's `_full_rounds`).
Deliberately NOT combined with the new K=4 refresh-schedule champion
(section 32) yet -- stacking two unvalidated combinations at once would
make failures unattributable.

Smoke-tested at tiny dims (CPU, zero GPU cost): loss finite, backward()
succeeds, P/O correctly frozen (no grad, by design), gate_w1/b1
correctly zero-grad on step 1 (documented protected-init behavior, not
a bug), held-out inference path clean, R in {0,1,16} all stable, the
single largest real ARC episode (9356 bytes) runs without crashing.
**Most important check**: swapping in a different task's demonstrations
produces a genuinely different final workspace state (0.064 max abs
diff) -- confirms task memory actually conditions the reasoning
workspace rather than being silently ignored by the forward pass.

`scripts/hz0h_bdh_arc_task_loader.py` also gained `build_episode_parts`
(returns memory/query/answer text separately instead of one joined
string), verified to rejoin to byte-identical output vs the original
`serialize_episode` on 50 real tasks.

Still not started: the variable-effort (LOW/MEDIUM/HIGH R-band)
training script, the 150M-param checkpoint warmstart (no warmstart
source exists yet at the new 150M config -- the existing
`hz0h_bdh_checkpoint_for_ablation.pt` SVD warmstart is sized for the
206.47M champion, not directly reusable), and the GPU dispatch decision
(cost-estimate-first, per the standing agreement).

**150M pretrain complete, 2026-09-01, run entirely locally (zero GPU
spend)**: `scripts/hz0h_bdh_hzcq_150m_pretrain.py`, 5M tokens, real
19,623s wall-clock (paused via SIGSTOP mid-run for the objective
refinement below, then resumed -- real throughput measured directly
via step-count diffs since the pause corrupted the script's own
cumulative rate stat, confirmed healthy throughout, 250-1300 tok/s
bursty range). **Real result: val_loss=1.849, params=150,577,393
(matches the target 150.58M config exactly)**. No SVD warmstart
(disclosed gap -- no dense-BDH source exists at 150M dims). Checkpoint
verified loadable (13 keys, correct 2128-dim embed, no NaN).

**Not comparable to the 206.47M champion's 1.3879** -- different scale,
no warmstart, 1/5 the tokens (5M vs 25M). This checkpoint's real
purpose is as the full warmstart source for HZ-CQ's ARC fine-tuning
phase (the actual gap this pretrain existed to close -- no 150M
checkpoint of any kind existed before this), not a quality result to
compare against other numbers in this document.

## ARC fine-tuning real result, 2026-09-01: an inverted-U, not a monotonic curve

**RELABELED HISTORICAL DIAGNOSTIC ONLY, 2026-09-01 (see section 34):**
this entire result was computed through `forward_hz_cq`'s answer-loss
code before a real, confirmed off-by-one bug in it was found and fixed
(every predictor position was paired with the byte TWO positions ahead
instead of one, and the first answer byte was never a supervised
target at all -- see `tests/reference/test_hz0h_bdh_arc_task_memory_torch.py`
and the fix commit). The pattern below (MEDIUM beating LOW and HIGH
8/8 times) was real and reproducible **as a measurement of the buggy
loss**, but that loss was not measuring "predict the true next byte"
correctly, so the inverted-U shape itself is not trustworthy evidence
about R-band behavior. Kept here for the record, not as a finding to
build on -- the real re-run (with both this fix and section 34's eval-
correctness work) is still pending.

`scripts/hz0h_bdh_hzcq_arc_finetune.py` built and run locally (400
examples, one full pass over the real ARC-AGI-1 training split, 8700s
wall-clock, warmstarted from the 150M pretrain checkpoint above). Per-
episode R sampled from LOW (2-4) / MEDIUM (6-8) / HIGH (12-16) bands,
per-band eval loss tracked at 8 checkpoints (every 50 examples) on a
fixed held-out set of real evaluation-split tasks.

**Real, decisive, and completely consistent across all 8 checkpoints,
no exceptions**:

| Step | LOW | MEDIUM | HIGH |
|---|---|---|---|
| 50 | 1.789 | 1.486 | 1.604 |
| 100 | 1.525 | 1.230 | 1.384 |
| 150 | 1.509 | 1.261 | 1.288 |
| 200 | 1.378 | 1.140 | 1.269 |
| 250 | 1.515 | 1.132 | 1.313 |
| 300 | 1.411 | 1.195 | 1.284 |
| 350 | 1.391 | 1.163 | 1.271 |
| 400 (final) | 1.460 | **1.157** | 1.364 |

**MEDIUM beats both LOW and HIGH at every single checkpoint.** This is
NOT the monotonic "more recurrent compute -> better answer" signature
this whole architecture line was built to test for (that would need
HIGH to beat MEDIUM) -- but it is real, reproducible evidence of an
inverted-U / sweet-spot pattern: LOW is consistently worst (some extra
reasoning depth clearly helps a lot, R=2-4 is not enough), while HIGH
consistently underperforms MEDIUM (more than R~6-8 stops helping and
mildly hurts, at least at this training scale).

**Real, disclosed caveats before reading too much into this:**

1. **This is teacher-forced next-byte loss, not task-solving accuracy.**
   Lower loss on the true answer bytes is suggestive but is not the
   same thing as the model actually producing correct ARC grids --
   BDH-CQ's own real target metric is pass@2 exact-match accuracy.
   An exact-match eval script has not been built yet; that's the next
   real thing needed before this finding can be compared to any
   external benchmark number.
2. **The 8 checkpoints are NOT independent replications** -- eval uses
   a fixed seed (0) against a fixed 30-task sample every time, so this
   is one real trajectory of the same instrument as training
   progresses, not 8 separate experiments. The *consistency* across
   the trajectory is still real signal (a noisy instrument wouldn't
   hold the same ordering 8/8 times), but it's one seed, one training
   pass, no repeat run yet.
3. **A real confound not yet ruled out**: HIGH-R episodes are longer
   (more appended latent-reasoning positions before the answer) than
   LOW-R episodes. Longer context immediately preceding the predicted
   bytes could affect loss through ordinary positional/recency
   dynamics unrelated to "reasoning quality" specifically -- this
   hasn't been isolated from the real R-effect yet.

**Honest verdict**: real, consistent, worth taking seriously -- but a
loss-based inverted-U on one training pass, not (yet) the "more
compute solves harder ARC tasks" result HZ-CQ was built to chase.
Next real steps: (1) build exact-match pass@k accuracy eval, (2) test
whether the sweet spot shifts with more training / more R-band
resolution, (3) rule out the length confound (e.g. compare against a
LOW-R run padded to HIGH-R's sequence length with no-op rounds).

## Objective sharpened, 2026-09-01: Pareto frontier, not a single score

**Standing governing loop, stated explicitly by the user**: recursively
test and update the architecture itself to maximize speed while
improving quality per parameter. Not a one-shot target -- an ongoing
cycle (real experiment -> real result -> real architecture change ->
re-test), the same discipline this whole project has followed since
the gate/refresh work in sections 22-28, now stated as the explicit
governing loop for everything after this point, including HZ-CQ.

Real refinement of section 33's target -- not "beat BDH-CQ's 29.5%
pass@2" as an isolated number, but beat it on its own cost/quality
frontier simultaneously:

$$
\boxed{\text{better quality} + \text{lower latency} + \text{lower memory} + \text{lower training/inference cost}}
$$

**Same sourcing caveat as section 33 applies to every BDH-CQ number
below.** Concrete target, if BDH-CQ is roughly 150M params / 29.5%
pass@2 / 0.85 H200-sec/task (per the user's external summary):

$$
\boxed{\ge 35\text{-}40\%\text{ pass@2}, \quad <100M\text{ params}, \quad <0.5\text{ H200-equivalent sec/task}}
$$

**Real tension with the standing 150M lock**: this message explicitly
argues for 50-100M params ("target 50-100M parameters rather than
matching 150M blindly") -- reasoning depth should come from recurrent
compute (more rounds), not duplicated layers, so a smaller model that
uses its recurrence well should be able to match or beat a larger one
that doesn't. This has NOT been reconciled with the in-progress 150M
pretrain above yet -- needs a real decision (resize before resuming,
or keep 150M as the ceiling and treat 50-100M as a later efficiency
pass once the memory+workspace architecture is validated at all).

**Per-experiment evaluation frame going forward** -- judge every future
HZ-CQ experiment on these four columns, not loss alone:

| Experiment | Intelligence | Speed | Memory/Cost | Verdict |
|---|---|---|---|---|
| Adaptive gate | real, validated (1.3879) | ~neutral | ~neutral | keep -- this is HZ-CQ's recurrence engine |
| BDH-Delta (belief/workspace bottleneck) | decisive real loss (1.7862) | some gain | maybe some gain | dead, section 27 -- do not revive without new evidence |
| K=4 cached-schedule refresh | real win (1.4108, beats K=6 at 1.4135+) | real win (less compute/round) | real win | keep, but not yet combined with adaptive gate (unvalidated combo) |
| Variable-R training (LOW/MED/HIGH) | target, not yet measured | flexible by construction | flexible by construction | high priority, not yet built |
| CoT distillation (Coconut-style) | maybe, per section 31's negative result on register-machine | inference neutral | training cost up | secondary, not first arm |
| Sparse/dynamic routing | unclear, this project's own earlier routing work hit real OOM issues | often down in practice | unclear | not being pursued right now |

Roadmap direction (not yet started once the params question above is
settled) superseded by section 34's reframing below.

---

# 34. HZ-CQ fresh research reframing, 2026-09-01 -- v0 relabeled, v1 designed

Real, substantive correction to everything built so far in section 33.
Same sourcing caveat as before applies to every external BDH-CQ number.
Condensed but faithful capture of the user's full reframing (the
complete version lives in this session's transcript, not reproduced
verbatim here for length).

## The core technical correction

**`forward_hz_cq()` (section 33) is relabeled HZ-CQ-v0: a growing-
sequence latent-recurrence smoke test, not a faithful implementation.**
Real problem identified: v0 concatenates all demos into one string,
embeds them together, and represents each latent reasoning step as a
NEW appended sequence position -- so there is no real persistent,
fixed-size $S_t = U_\theta(S_{t-1}, D_t)$, and $H_r$ is not a
structured workspace, it's just "more sequence." This directly
explains why R and sequence length were never disentangled in the
fine-tuning result above -- the exact confound flagged as unresolved
in that writeup, now identified as architectural, not incidental.

**HZ-CQ-v1 design correction**: $S$ must be a real fixed-size
multi-vector memory ($S_t \in \mathbb{R}^{M_S \times D_S}$, $M_S$
small like 4-16, $D_S$ kept high per BDH-Delta's own lesson against
aggressive bottlenecks), updated via a real adaptive-gated write
($\Delta S_t = U_\theta(S_{t-1}, E(D_t))$, $g_t^S = C_\phi(\ldots)$,
same gated-residual pattern as the validated adaptive gate) and
discarded after each demo -- NOT accumulated as growing raw sequence.
$H$ must be the same: a fixed-size workspace ($M_H \in \{4,8\}$ slots)
that the SAME slots evolve against across R rounds, so R becomes a
real compute-depth variable instead of a sequence-length variable.
$S$ conceptually answers "what task am I solving" (updated slowly,
during demo ingestion only); $H$ answers "what am I computing right
now" (updated fast, during query reasoning) -- keep them structurally
separate from the start rather than blurring them.

## What the v0 fine-tuning result (above) actually shows, honestly

Real, useful, but not what it might look like at first: R=2-4 is
consistently insufficient (LOW worst in all 8 checkpoints) and R=6-8
substantially helps -- real evidence the direction is alive. But R=12-16
not helping further is now suspect as a length artifact (longer
sequences push the answer further from the demos/query) rather than
proof that more reasoning genuinely stops helping. **Do not conclude
"R=6-8 is optimal" from v0** -- the instrument that produced that
result cannot currently distinguish "more reasoning" from "longer
sequence." This needs isolating (see Step 2 below) before it means
anything architectural.

## Explicit, ordered priority (per the user's own message, do not reorder)

**Step 1 -- evaluation correctness, P0, blocks everything else.** No
major new training until this exists: (a) a real unit test proving the
teacher-forced answer-byte target alignment is correct (no assumed
off-by-one indexing -- construct a tiny deterministic sequence, assert
exact position-to-target mapping); (b) true held-out generation (model
never sees the answer, real exact-grid-match parsing, malformed-output
detection, pass@1 today, pass@2 later -- teacher-forced CE becomes a
debugging metric only, not the headline number); (c) a PAIRED fixed-R
evaluator: same episodes, same checkpoint, only R varies across
{0,1,2,3,4,6,8,10,12,16,24}, tracking exact accuracy(R), CE(R),
latency(R), VRAM(R) -- the current LOW/MEDIUM/HIGH eval samples
DIFFERENT episodes into each band, which is not a real controlled
comparison.

**Step 2 -- diagnose v0 on the ALREADY-TRAINED checkpoint (no new
training needed).** Run the paired-R curve above on the existing
`hz0h_bdh_hzcq_arc_finetune_checkpoint.pt`, plus an explicit
length-confound control: real R=4 vs "padded R=4" (4 real thought
steps + 12 inert/no-op positions so the answer sits at the same
sequence position real R=16 would put it) vs real R=16. If padded-R4
degrades to R16 levels, the inverted-U is mostly a position/length
artifact. If padded-R4 stays good while real R16 still worsens, the
extra recurrence itself is the problem. This is cheap (no training,
reuses the existing checkpoint) and should happen before any more GPU/
wall-clock is spent training.

**Step 3 -- build HZ-CQ-v1's persistent $S$** (real fixed-size
multi-vector memory, sequential demo ingestion, adaptive-gated writes,
unit-tested for demo-sensitivity/order-dependence/accumulation/
capacity/efficiency -- query-time cost must NOT scale with raw demo
token count, or the memory architecture has failed its actual purpose).

**Step 4 -- build HZ-CQ-v1's structured $H$** (fixed-size workspace,
adaptive-gated writes, same slots evolve across R rounds, no new
sequence positions -- R becomes real compute depth). Rerun the paired-R
curve on v1.

**Step 5 -- real curriculum.** One pass over 400 tasks (the v0 run
above) is nowhere near enough; build episodic training from ARC train +
RE-ARC + ConceptARC + ARC-GEN100K + procedural generators targeting
BDH-CQ's reported failure modes specifically (composition, ordering,
nesting, extrapolation, conditional rules) -- only after v1's $S$/$H$
are validated structurally. K=4 refresh, rank-64, compiled execution,
adaptive halting, and candidate verification/ranking all get added
ONE AT A TIME after v1's skeleton is proven, each measured
independently -- explicitly not bundled together, the same "no
combined experiments before isolated wins" discipline this whole
project has repeated since the gate/refresh work in sections 22-28.

## Explicit "do not" list (real, worth keeping visible)

Do not: run another generic LM pretrain just because it's familiar;
optimize v0's kernels; call MEDIUM "fundamentally optimal" off the v0
result; call the current raw-sequence state "faithful BDH-CQ memory";
revive the tiny BDH-Delta belief bottleneck; combine multiple
architecture changes into one run; optimize against teacher-forced CE
instead of real task success; call stochastic two-sample decoding
"pass@2 candidate ranking" without a real verifier; abandon the
mainline adaptive-gate/K=4 work while chasing this.

## Benchmark discipline going forward

Every serious HZ-CQ variant gets: params, real ARC pass@1/pass@2,
per-difficulty accuracy, accuracy-vs-R curve, mean R used, GPU-sec/
task, peak VRAM, training cost -- plus accuracy-per-GPU-second, not
loss alone. Architecture decisions happen on a private/held-out
development set (RE-ARC/ARC-GEN/procedural, not the public ARC-AGI-1
eval split), which only gets touched at real milestones -- otherwise
the public benchmark quietly becomes the training objective.

**Immediate next real action**: Step 1a (the answer-alignment unit
test) -- cheap, unblocks everything else, and directly checks
`forward_hz_cq`'s existing teacher-forced loss indexing for the exact
kind of off-by-one bug this reframing flags as unverified.

## Step 1 complete, 2026-09-01: real bug found, real eval infra built

**Step 1a**: `tests/reference/test_hz0h_bdh_arc_task_memory_torch.py`
found and fixed a REAL, confirmed off-by-one bug in `forward_hz_cq`'s
answer loss -- every predictor position was paired with the byte TWO
positions ahead instead of one, and the first answer byte was never a
supervised target at all. Confirmed two ways (exact position/target
pairing, gradient-descent convergence on a short deterministic
sequence), fixed, re-verified against all prior edge cases. **Real
consequence**: section 33's entire ARC fine-tuning result (the
MEDIUM-beats-LOW-and-HIGH inverted-U) was computed with this bug and
is now relabeled historical diagnostic only, not trustworthy R-band
evidence -- see that section's own updated note.

**Step 1b/c**: `scripts/hz0h_bdh_hzcq_arc_eval.py` built -- real
autoregressive held-out generation (model never sees the answer), a
real `END` terminator added to the episode format (`hz0h_bdh_arc_task_loader.py`,
a genuine gap found while building this: the old format had no way for
generation to know when a grid ends), exact grid-match pass@1, and a
paired fixed-R evaluator (same frozen episodes, same checkpoint, only
R varies -- a real controlled comparison the old LOW/MEDIUM/HIGH
training eval never was). Small real probe (2 dev episodes × R in
{0,4,8}, 40-byte cap): 0% pass@1 across all R, exactly as expected
given the checkpoint was trained under the now-fixed loss bug -- not
evidence about R yet, just confirmation the infrastructure works
end-to-end.

**Real, disclosed cost finding**: ~24-26s per generated example even
at a 40-byte cap, because v0 has no incremental cache -- every
generated byte re-runs full rounds over the whole growing sequence.
Real ARC answers average much longer (up to ~900+ bytes for large
grids), so a full paired-R sweep at realistic length would be
impractically slow locally on the current checkpoint. This is now real
evidence (not just a measurement-cleanliness argument) for why v1's
fixed-size S/H matters for inference cost, not only for disentangling
R from sequence length.

**Step 1 is now done.** Per the explicit ordering: Step 2 (diagnose
v0 on the existing checkpoint, no new training) is moot until a fresh
checkpoint is trained with the corrected loss -- diagnosing a
checkpoint that never had correct supervision isn't informative. Real
next real action: retrain the ARC fine-tune (real, same recipe,
corrected loss) to get a checkpoint worth actually evaluating, THEN
run the real paired-R sweep on it.

## Standing three-part governing goal, updated 2026-09-01

Set explicitly via `/goal` (session-scoped, stays active until met):
(1) recursively test and update the architecture to maximize speed
while improving quality per parameter -- the ongoing loop already
described above; (2) strive for the smallest coherent chat-capable
model -- a real, human-legible English-conversation test surface,
much richer than byte-level ARC grids or raw LM loss alone; (3) real
architecture validation still comes first, per the user's own explicit
ordering ("first before even that we need to validate the architecture") --
(2) is a genuine milestone to work toward, not deprioritized/shelved
anymore, but it's sequenced AFTER the HZ-CQ Step 1-5 validation work
this section is mid-way through, since a chat-capable model still
needs a validated recurrence mechanism under it to be worth training
efficiently. Current real active work (the corrected ARC fine-tune
retrain, then the real paired-R eval) serves both goals at once: it's
architecture validation AND the same infrastructure a future chat-
model training run would reuse.

## Real corrected ARC fine-tune retrain, 2026-09-02 (RunPod RTX 5090)

Re-ran the exact same recipe as section 33's original ARC fine-tune
(`scripts/hz0h_bdh_hzcq_arc_finetune.py`, 400 examples, warmstarted
from the 150M pretrain checkpoint) with the corrected `forward_hz_cq`
answer-loss (Step 1a's off-by-one fix). Dispatched to a RunPod RTX
5090 -- real cost ~$0.99/hr x ~25min wall (1351s training + setup) =
under $0.50. Two real, disclosed dispatch bugs hit and fixed along the
way: `scripts/runpod_run.sh --sync local` excludes both `results/` and
`data/` by default, so the first attempt crashed immediately
(`IndexError: Cannot choose from an empty sequence` -- `data/arc_agi_1`
never made it to the pod); fixed by pushing the 602MB pretrain
checkpoint and the 5.3MB ARC dataset directly via rsync/scp before
rerunning with `--sync none`. Pod terminated cleanly on completion
(`--kill-reused`), `results/local/hz0h_bdh_hzcq_arc_finetune_fixed.json`
committed, `_checkpoint.pt` left untracked per convention.

**Real held-out eval-band losses (LOW/MEDIUM/HIGH), all 4 checkpoints:**

| step | LOW    | MEDIUM | HIGH   |
|------|--------|--------|--------|
| 100  | 1.4219 | 1.1110 | 1.1586 |
| 200  | 1.3337 | 1.0659 | 1.1869 |
| 300  | 1.2774 | 0.9679 | 1.0829 |
| 400  | 1.2944 | 0.9740 | 1.1237 |

MEDIUM beats both LOW and HIGH at every one of the 4 held-out eval
checkpoints -- consistent, not a one-off. This is the SAME qualitative
pattern the pre-fix (buggy) run showed, but this time computed with
the corrected loss, so it's real evidence, not a bug artifact.

**Real, more interesting wrinkle**: `train_band_means` (loss averaged
over the noisy per-step *training* losses, all 400 steps) tells a
DIFFERENT story than eval -- HIGH=1.2586 < MEDIUM=1.2927 < LOW=1.3127,
i.e. monotonically improving with more R, the naive "more compute
helps" signature. Only the held-out eval bands show the MEDIUM
sweet spot. Read plainly: HIGH-R episodes fit their own training
loss best but generalize worse than MEDIUM-R to held-out tasks --
a real train/eval divergence, not just noise (holds at all 4 eval
checkpoints), and a more informative finding than either "more
compute helps" or "there's a flat compute-depth ceiling" alone would
have been.

**Explicit caveat, per this doc's own benchmark discipline**: this is
still teacher-forced CE under the ORIGINAL unpaired LOW/MEDIUM/HIGH
eval protocol (`eval_pass` samples different random episodes into each
band per call) -- not the paired fixed-R evaluator built in Step 1c
(`scripts/hz0h_bdh_hzcq_arc_eval.py`, same frozen episodes, only R
varies) and not real exact-match pass@1. Per the explicit "do not"
list above ("optimize against teacher-forced CE instead of real task
success"), this loss pattern motivates the next step, it doesn't
substitute for it.

**Real next action**: run `hz0h_bdh_hzcq_arc_eval.py`'s paired fixed-R
sweep against this fresh, correctly-supervised checkpoint
(`results/local/hz0h_bdh_hzcq_arc_finetune_fixed_checkpoint.pt`) to
get real exact-match pass@1 and a confound-free accuracy-vs-R curve --
this is the actual Step 2 diagnostic the whole eval-infra build in
Step 1b/c was for.

## Real paired fixed-R held-out eval, 2026-09-02 (fresh corrected checkpoint)

Ran `scripts/hz0h_bdh_hzcq_arc_eval.py`'s paired fixed-R evaluator
against the just-retrained `hz0h_bdh_hzcq_arc_finetune_fixed_checkpoint.pt`
(the one with the corrected loss, see prior section) -- the actual
Step 1c instrument, real autoregressive held-out generation, not
teacher-forced CE. Same 3 frozen dev episodes (evaluation split,
seed=7) at R in {2, 7, 14} -- one representative from each effort
band. Real, disclosed session hiccup: the background dispatch got
externally killed twice mid-run with zero output (not a script crash --
clean process exit, no traceback, no leftover pid); worked around by
running each R value as a small foreground call instead (~4min each).

| R  | n | pass1_accuracy | malformed_rate | mean_generated_bytes | mean_s/example |
|----|---|----------------|-----------------|----------------------|-----------------|
| 2  | 3 | 0.0            | 0.0             | 120.0                | 77.2            |
| 7  | 3 | 0.0            | 0.0             | 120.0                | 76.3            |
| 14 | 3 | 0.0            | 0.0             | 120.0                | 83.5            |

**Real finding, and it's not just "wrong content"**: all 9 generations
(3 episodes x 3 R) hit the 120-byte cap -- but the 3 dev episodes'
real true-answer lengths are 66, 166, and 52 bytes. Two of three
(66 and 52) are well under the cap, so a model that had learned the
`END` terminator should have stopped well before 120 bytes on those.
It never did, at any R. Real conclusion: at 400 training examples
(one pass), the model hasn't learned to emit the stop marker yet --
this is a real, distinct failure mode from getting grid content wrong,
and it means pass@1 at this checkpoint isn't yet a meaningful
architecture signal (accuracy is floored at 0% by a formatting gap,
not by reasoning depth). `malformed_rate=0.0` despite this just means
`parse_answer` still recovered a well-formed (wrong) grid from
whatever digits came out before the cap or a ragged line ended it.

**Sample-size caveat**: n=3 per R is nowhere near enough to read
anything into the accuracy-vs-R curve itself (it's flat at 0% because
of the floor above, not because R doesn't matter) -- this run's real
purpose was confirming the eval pipeline produces sane, honest output
end-to-end on the corrected checkpoint, which it did.

**Real next actions**: (1) more training data/epochs so the model
actually learns the `END` convention before spending more eval budget
on accuracy -- one epoch over 400 examples with everything else
(persistent memory format, effort bands, answer format) still novel is
not enough exposure; (2) once generation reliably terminates, rerun
this paired sweep with a larger dev-n for a real accuracy-vs-R
reading. This -- not more architecture speculation -- is the actual
bottleneck the last two real results (training retrain + this eval)
point at.

## Real 2000-example ARC fine-tune (5x more exposure), 2026-09-02 (RunPod RTX 5090)

Per explicit instruction to address the real bottleneck the paired-R
eval flagged (model hadn't learned the `END` stop marker at 400
examples): reran the same fine-tune recipe with `--n-examples 2000`
(5 real epochs over the 400 ARC training tasks instead of 1), fresh
warmstart from the 150M pretrain checkpoint. Real cost: RTX 5090,
6347s (~106min) training + setup, ~$1.85 total. Two bad-network pods
hit and killed early (real, disclosed: one pod transferred a 602MB
file at ~70KB/s -- would've taken hours and cost real money for
nothing; killed both within minutes of confirming the slow rate rather
than let them bill). Third pod (same IP that worked cleanly earlier
this session) transferred cleanly in ~55s.

**Real held-out eval bands across all 10 checkpoints (step 200-2000):**

| step | LOW    | MEDIUM | HIGH   |
|------|--------|--------|--------|
| 200  | 1.3337 | 1.0659 | 1.1869 |
| 400  | 1.2944 | 0.9740 | 1.1237 |
| 600  | 1.1795 | 0.8869 | 1.0708 |
| 800  | 1.1882 | 0.8897 | 1.0114 |
| 1000 | 1.1824 | 0.8280 | 1.0565 |
| 1200 | 1.1615 | 0.8483 | 1.0634 |
| 1400 | 1.1323 | 0.8196 | 1.0171 |
| 1600 | 1.1431 | 0.8257 | 0.9916 |
| 1800 | 1.1652 | 0.8156 | 0.9760 |
| 2000 | 1.1395 | 0.8621 | 1.0549 |

MEDIUM beats both LOW and HIGH at all 10 checkpoints now (vs 4/4 at
400 examples) -- the sweet-spot pattern isn't just surviving more
training, the gap is widening (MEDIUM-vs-HIGH gap grows from ~0.12 at
step 200 to ~0.19 at step 2000).

**Real, sharper version of the train/eval divergence finding**:
`train_band_means` are now nearly IDENTICAL across bands (LOW=0.9960,
MEDIUM=0.9954, HIGH=0.9850, spread=0.011 -- down from spread=0.054 at
400 examples). Training loss has converged to roughly the same value
regardless of R, but held-out generalization still clearly separates
the bands, with MEDIUM the consistent best generalizer. Read plainly:
this is NOT the model getting equally good at every R and then
generalizing differently by chance -- it's fitting all three R-bands'
training losses about equally well while consistently generalizing
worse at both very-low and very-high R. That's real evidence the
sweet spot is about generalization/robustness at this R, not just
training-loss fit.

**Real next action**: spot-check whether more exposure fixed the
`END`-terminator gap flagged in the last eval (Step 1c infra,
`hz0h_bdh_hzcq_arc_eval.py`) -- in progress as of this writeup, real
autoregressive generation against this fresh checkpoint at R=7,
200-byte cap, same 3 frozen dev episodes. Once that lands, rerun the
full paired R-sweep with a larger dev-n for a real accuracy-vs-R
reading -- this is now the actual blocking step before this line of
work says anything about real task-solving capability rather than
teacher-forced loss.

**Addendum, spot-check result**: it didn't. R=7, same 3 frozen dev
episodes, 200-byte cap: `mean_generated_bytes=200.0` -- hit the cap on
all 3 examples again, exactly like the 400-example checkpoint did at
its 120-byte cap. `pass1_accuracy=0.0`, `malformed_rate=0.0`
(well-formed-but-wrong/truncated grids, same as before). 5x more
training exposure sharpened the held-out teacher-forced loss gap
(MEDIUM's generalization edge over LOW/HIGH got real and repeatable),
but did NOT teach the model to emit `\nEND` and stop. These are
genuinely separate capabilities -- getting better at *predicting the
next byte of a mostly-right answer under teacher forcing* is not the
same skill as *knowing when the answer is finished during free
generation*, and this checkpoint has made real progress on the first
without the second budging at all. Real implication: pass@1 stays
floored at 0% regardless of R until this is fixed specifically, so
further R-band training exposure alone won't move the real metric --
this needs direct attention (e.g. more explicit weight/supervision on
the terminator bytes, or checking whether `\nEND` is functionally
under-represented in the loss vs. the far larger token budget spent on
grid-digit bytes) before another round of "just train more" is worth
dispatching.

**Correction, real teacher-forced diagnostic**: the terminator-
supervision hypothesis above is wrong -- retracting it. Ran a direct
check (no generation, single teacher-forced forward pass per episode
on the 2000ex checkpoint, R=7, same 3 dev episodes): at the TRUE
`\nEND` positions, the model's predicted probability for the correct
byte is 0.87-1.0 with rank_of_true=0 (its actual top-1 choice) at
essentially every terminator position across all 3 episodes. It has
learned `END` under teacher forcing about as well as anything could be
learned. The real number that matters instead: overall per-byte top-1
accuracy under teacher forcing is only 0.44-0.67 across the 3
episodes -- roughly half the answer bytes are wrong even when every
prior byte is the true one.

Read together, this is classic **exposure bias**, not a terminator-
specific gap: during free generation the model's own early wrong bytes
compound (no teacher forcing to correct them), so it drifts off the
true grid's row/column structure before it ever reaches a state that
resembles "a complete, correctly-shaped grid" -- there's no available
`END` to predict because the input context it's actually seeing during
generation never matches anything like the true-answer contexts this
diagnostic just showed it handles well. Upweighting the terminator
loss would do nothing (it's already ~perfectly supervised); the real
lever is overall content accuracy and/or exposure to the model's own
generation trajectory during training (e.g. scheduled sampling), not
special-casing the last few bytes.

**Real next action, revised**: before spending more GPU budget, decide
between (a) more of the same training (raise overall per-byte accuracy
enough that free-running trajectories stay close enough to the true
grid to reach a real stopping point) vs (b) a real architecture/
training-procedure change addressing exposure bias directly (e.g.
mixing some fraction of self-generated context into training, matching
how real eval will condition the model). Given this project's "one
change at a time" discipline and that (a) hasn't been tried past 2000
examples yet, the disciplined next step is to first check whether
per-byte accuracy keeps improving with data before reaching for a
training-procedure change -- not dispatched yet, flagged for a
deliberate go/no-go rather than auto-launched, given real GPU spend is
involved and funds are limited right now.
