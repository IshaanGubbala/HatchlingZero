# HZ-0C History Audit

Date: 2026-08-02. Per `plans/HZ-0C_Surprise_Anchors_Total_Restart_Plan.md`'s
C0 instruction: "Assume only the README, master plan, and Git history
survive. No prior HZ-0C implementation is trusted until reproduced."
Full `git log --all` sweep performed (not just the current branch).

## Finding 1: no prior implementation exists for the current HZ-0C concept

`git log --all --oneline | grep -iE "anchor|surprise|trigger"` returns
**zero commits**. Nothing in this repository's history has ever
implemented surprise-triggered anchor attention, a surprise scalar
signal, or bounded/triggered attention insertion. This is a genuine
clean slate, not a "reproduce and verify" situation like B0 found for
HZ-0B (which recovered real prior memory-mechanism work). C1-C9 start
from nothing.

## Finding 2: "HZ-0C" is a reused name for a DIFFERENT, now-relocated concept

`git log --all --oneline | grep -i "hz-0c\|hz0c"` returns commits
about **session-local fast weights** (Phase 16: `9a52a59` prototype,
`2e6ecbe` full integration, `b5daab1` ICL benchmark eval, `4d93933`
production hardening, `fee890b` completion summary, `b6c8fd4` honest-
status correction). That work is unrelated to the current HZ-0C
definition (surprise-triggered anchor attention) -- it has already
been correctly relocated: `plans/HZ-0D_Fast_Weights_Total_Restart_Plan.md`
exists as its own restart plan, and the actual legacy code is already
moved to `archive/src/hz0/fast_weights/` (not live, not imported by
anything current). No action needed here beyond documenting the
rename so a future reader doesn't confuse the two "HZ-0C"s.

**Honest note on that legacy work's own history, since it's
informative context for HZ-0D later, not HZ-0C**: `b6c8fd4`'s own
commit message ("HZ-0C honest status, remove premature 'production-
ready' claims" -- "Was: Phase 14 ready to ship. Now: Phase 14
working, needs validation before production") is itself an instance of
this project's recurring pattern -- an early "done"/"production-ready"
claim later walked back to "infrastructure works, the actual
capability gain is not demonstrated." The same discipline that caught
this before should apply to HZ-0C (surprise-anchors) evaluation later:
infrastructure completeness is not evidence of capability.

## Finding 3: the top-level master plan/tracker are stale relative to the current restart plan

`plans/HATCHLING-ZERO Development Plan.txt` line 507 and
`plans/HATCHLING-ZERO_Progress_Tracker.md` line 22 both still label
"HZ-0C" as "Session-local fast weights" -- the OLD definition, not the
current one in `plans/HZ-0C_Surprise_Anchors_Total_Restart_Plan.md`
and `plans/HZ-0C_Progress_Tracker.md` (surprise-triggered anchor
attention). These top-level docs were not updated when HZ-0C/D/E were
restructured (fast weights -> HZ-0D, current HZ-0C -> surprise
anchors, and presumably HZ-0E -> whatever
`plans/HZ-0E_Micro_MoE_Total_Restart_Plan_Corrected.md` covers). Not
fixed in this pass -- these are pre-existing, currently-uncommitted
files this session has been instructed not to touch; flagged here so
whoever does touch them next has the context, and so a future reader
of the master plan doesn't get a stale picture of what HZ-0C actually
is.

## Finding 4: HZ-0A already has the concrete "fixed periodic anchors" baseline C1 needs

`reference/hz0a_mlx_model.py`'s `HZ0AMlxModel.__init__` takes
`attention_indices: tuple[int, ...]` and builds each `Block` with
`index in attention_indices` determining whether that layer is a full-
attention layer or a recurrent one. The frozen HZ-0A checkpoint used
throughout HZ-0B's B11 work
(`outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout`)
uses `ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)` out of 31 layers --
this IS, concretely, model 2 of C1's "three controlled models" (scaled
recurrence with FIXED periodic anchors), already built, trained, and
frozen. C1's model 1 (no anchors) and model 3 (surprise-triggered
anchors) do not yet exist, but model 2's real architecture pattern
does not need to be invented -- extending `attention_indices` from a
fixed schedule to a dynamically-triggered one is a direct, concrete
next step, not abstract design work.

## What this means for C1 (freeze the scaled topology)

C0's exit gate ("the design is explicit without depending on archived
code") is met by findings 1-3 above: no archived code is being relied
on, the one relevant piece of prior art (finding 4) is current, live,
already-tested infrastructure, not archived material. C1 can proceed
directly to defining the scaled topology, using HZ-0A's existing
`attention_indices` mechanism as the concrete substrate for the fixed-
anchor baseline, per `docs/restart/hz0c_recovered_requirements.md`.
