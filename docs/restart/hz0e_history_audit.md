# HZ-0E History Audit

Date: 2026-08-05. Per `plans/HZ-0E_Micro_MoE_Total_Restart_Plan_Corrected.md`'s
own E0 instruction: "Recover and classify prior expert counts, placement,
routing, balancing, capacity rules, and failures." Full `git log --all`
sweep performed (not just the current branch), matching HZ-0C's C0 and
HZ-0D's D0 precedent for this project.

## Finding 1: no prior implementation exists for micro-MoE/routing in this project's own code

`git log --all --oneline -i --grep="moe\|expert\|rout"` and a filename
sweep (`git log --all --diff-filter=A --name-only`) return **zero real
hits** for MoE/expert-routing work -- the only filename match across
all of history is `plans/HZ-0E_Micro_MoE_Total_Restart_Plan_Corrected.md`
itself. `git log --all --oneline -i --grep="mixture\|top-k\|top-1"`
returns unrelated hits: HZ-0A's A5 data-mixture-ratio work (training
corpus blending, not model routing) and HZ-0B's oracle memory-slot
routing (a different, already-completed mechanism, unrelated to expert
routing).

A broader filesystem sweep (`find` across `archive/src`, `reference`,
`scripts`, `restart` -- the project's own source trees, not installed
packages) also returns zero real hits. The only filesystem matches for
"moe"/"expert"/"router" are (a) third-party library source under
`archive/.venv/lib/.../mlx_lm/` and `transformers/` -- pip-installed
reference implementations of OTHER models' MoE architectures (Qwen,
GLM, Granite, etc.), never imported by this project's own code, and (b)
coincidental substring matches inside Rust build-artifact hash
filenames (`hz0a_pmetal_gpu-...7xmoepdpzbik1j7...`) with no semantic
relation to MoE at all.

**This is a genuine clean slate**, matching HZ-0C's C0 finding for
surprise-triggered anchors (not HZ-0D's D0 finding, which recovered
real, reusable prior fast-weight infrastructure under a different,
relocated name). E1-E10 start from nothing; no archived code needs
auditing for reuse or for repeating past mistakes.

## Finding 2: HZ-0E's own tracker exists but no phase work has started

`plans/HZ-0E_Progress_Tracker.md` was created at the repo restart
(`47b79a1`, "Restart repo layout and add progress trackers") and
touched twice more (`fb0673e`, `3d3eea5`) -- both administrative
updates (dependency-status notes reflecting HZ-0D's completion), not
real E0-E10 work. The tracker's own "Overall phase" field reads
`E0 not started` as of the most recent touch. Confirmed by reading the
tracker directly, not inferred from commit messages alone.

## Finding 3: HZ-0A's existing dense FFN is the concrete substrate E1 will need to specify against

`reference/hz0a_mlx_model.py::Block` gives EVERY layer -- whether a
GDN-2 recurrent mixer or a full-attention mixer -- an identical dense
SwiGLU FFN: `gate`/`up`: `Linear(768, 2304)`, `down`: `Linear(2304,
768)`. Verified directly against the real frozen checkpoint's own
weight shapes (not assumed from a script constant, which in this
project has drifted before -- see `hz0d_history_audit.md`'s own lesson
about trusting stated numbers over measured ones):

```
d_model = 768, d_ff = 2304, num_blocks = 31 (indices 0-30)
attention (full self-attention) layers: 4, 9, 14, 19, 24, 29 (6 of 31)
GDN-2 (recurrent) layers: the other 25
per-block dense FFN params (gate+up+down, weights+biases): 5,313,792
total dense FFN params across all 31 blocks: 164,727,552
```

This is real, concrete infrastructure E1's "replace selected upper MLP
blocks only" contract will specify against directly -- not abstract
design work. Every block (attention or recurrent) has the SAME FFN
shape, so "upper MLP blocks" is a real, well-defined subset choice
(e.g. the last N of 31 blocks' `gate`/`up`/`down`), not something that
needs new architecture invented first. Whether the MoE-replaced blocks
should overlap with the 6 attention layers, the 25 GDN-2 layers, or be
chosen independently of that split is an open E1 decision, not resolved
here.

## What this means for E1 (the expert contract)

E0's exit gate (an explicit design, real prior-work classification) is
met by findings 1-3 above: no archived MoE code exists to build on or
avoid repeating mistakes from (finding 1), no phase work has started
yet so there is no partial state to reconcile (finding 2), and the one
piece of relevant real infrastructure (finding 3, HZ-0A's existing
per-block dense FFN) is current, live, and precisely measured from the
real checkpoint. E1 can proceed directly to specifying the expert
contract (count, size, placement, top-k policy, capacity, fallback,
active-vs-total parameters) against this real substrate, per
`docs/restart/hz0e_recovered_requirements.md`.
