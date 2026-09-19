"""Hatchling World Language Nursery, Stages L0/L1 -- reference/
hz_language_model_torch.py. Reuses HZCQPersistentMemory and
HZCQReasoningWorkspace EXACTLY as validated (default LN recurrence,
M_H=32, D/2 value/write, exact Q/K) -- zero architecture changes, per
plans/Hatchling world.md section 2's "no new recurrence experiments."

Stage L0 (pure self-supervised LM, no grounding yet): H evolves ONE
STEP PER TOKEN via the tied `step()` operator -- H_{t+1}=F_theta(H_t,
S,x_t), x_t = the current token's embedding, S left at its untouched
init (no prior "lifetime" evidence exists for a single sentence).
After each step, a cross-attention readout over H predicts the NEXT
token. This is real teacher-forced next-token prediction, using the
SAME tied per-round operator the room-navigation agent uses per
reasoning round -- here "one round" = "one token" instead of "one
reasoning round per decision."

Stage L1 (grounded nouns/properties): objects are encoded as a fixed
per-object feature set (x_objects, analogous to the room agent's
per-room encoding); the instruction is ingested INTO persistent memory
S via `mem.update_sequence` (one token per real memory-update step,
same mechanism the room agent uses for post-action consequences); H
then reasons for R rounds over S (what the instruction said) and
x_objects (what's actually there), and the final readout is a real
cross-attention over x_objects (not a fixed classifier) producing a
distribution over WHICH object matches -- naturally scale-invariant to
the number of objects, and forces the model to use "the instruction I
just read" (S) to select from "what I'm looking at" (x), the exact
same S-vs-x semantic split as everywhere else in this project.

Stage L2 (verbs through consequences): same S/H split as L1, but the
readout now also PREDICTS THE VERB'S EFFECT -- the referenced object's
post-action (position, held, opened) -- from a soft-attention read over
its own real pre-action state. "Verb meaning" here is literally defined
as "the transition this instruction causes," per section 5: getting the
right object (L1's job) is necessary but not sufficient; L2 additionally
scores whether the model predicts the CORRECT resulting state, matching
plans/Hatchling world.md's "verb meaning is a learned state transition,
not co-occurrence in text."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import HZCQPersistentMemory, HZCQPersistentMemoryConfig
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig


class FactorizedObjectEncoder(nn.Module):
    """The promoted default object representation (plans/Hatchling
    world.md, composition-encoder ablation, 2026-09-04/05): each
    attribute gets its OWN embedding table, and the object's
    representation is their SUM -- composing "small" and "red" is
    structurally just vector addition, not something a single shared
    Linear layer (the old approach: concatenate one-hots, mix with one
    Linear) has to learn to keep separable on its own.

    Promotion rationale, from real experiments, not assumption: in
    ISOLATED L3 training this beat the old concat+Linear encoder by a
    wide, reproducible margin on held-out UNSEEN (size, color) combos
    (2 seeds: 92.3% vs 58.0%, and 55.7% vs 0.0% -- the old encoder
    sometimes converged to a systematic wrong answer, below its own
    chance floor). A follow-up 5-seed regression check training BOTH
    encoders jointly with L1/L4-logic/L5 found no difference at all --
    both reach 1.000 on every metric, every seed -- so joint training
    with L1 may independently close the same gap. Promoted anyway
    because it never underperforms the old encoder in either setting
    and was the deciding factor in the harder, isolated one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.type_embed = nn.Embedding(4, d_model)
        self.color_embed = nn.Embedding(4, d_model)
        self.size_embed = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(2, d_model)

    def forward(self, type_idx: torch.Tensor, color_idx: torch.Tensor,
                size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        return (self.type_embed(type_idx) + self.color_embed(color_idx)
                + self.size_embed(size_idx) + self.position_embed(position_idx))


class LocalCausalMixer(nn.Module):
    """Minimal causal local mixer for HZ block recurrence (Experiment 3,
    2026-09-16 AMX-execution-geometry pass, section 4). Deliberately the
    SIMPLEST possible causal local mixer per the plan's own instruction
    ("prefer something like: embedding -> large fused projection ->
    causal local attention -> MLP... do not build a complicated
    router"): one causal self-attention block (fused QKV, per the
    section-6 fusion philosophy) + one MLP, standard pre-norm residual
    structure. Entirely new, separate parameters from the per-token
    H-driven L0 path -- does not reuse or modify _lm_forward_step /
    HZCQReasoningWorkspace at all. H itself (the tied reasoning
    operator) stays exactly HZCQReasoningWorkspace, just invoked once
    per block instead of once per token (see HZLanguageModel.
    lm_forward_blocked)."""

    def __init__(self, d_model: int, n_heads: int = 4, mlp_mult: int = 4, num_sinks: int = 4):
        """num_sinks (2026-09-16, real evidence from arXiv:2608.28444
        "Sliding-window beats linear attention" -- verified real, see
        session notes): a small number of LEARNED key/value pairs, not
        derived from any token, that every real position can attend to
        regardless of the causal mask. `local_mixer`'s causal self-
        attention was already a de facto sliding-window (bounded to
        block_size K) even before this -- attention sinks are the one
        piece that paper's own evidence says such a window needs to
        match/beat linear-attention alternatives. num_sinks=0 reduces
        exactly to the prior (pre-sink) behavior -- see
        tests/test_hz_block_recurrence.py's regression test."""
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.num_sinks = d_model, n_heads, num_sinks
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model), nn.GELU(), nn.Linear(mlp_mult * d_model, d_model))
        if num_sinks > 0:
            # Two separate learned banks (not derived from qkv/any real
            # token) -- sinks are keys/values ONLY, never queried
            # themselves (no output position "is" a sink), matching the
            # standard attention-sink formulation.
            self.sink_k = nn.Parameter(torch.zeros(num_sinks, d_model).normal_(std=0.02))
            self.sink_v = nn.Parameter(torch.zeros(num_sinks, d_model).normal_(std=0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, k, D), k = current block's real length (<=block_size,
        the final block of a sequence may be shorter). Causal within the
        block ONLY -- position i attends to positions [0, i] of x, PLUS
        every sink (sinks are learned parameters, not derived from any
        specific token or example, so attending to them carries no
        future-token leakage regardless of position). No information
        from outside this call's own x reaches the output otherwise
        (H-conditioning, if any, must already be baked into x by the
        caller before this is invoked -- see lm_forward_blocked's
        h_cond addition)."""
        b, k, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, k, 3, self.n_heads, self.head_dim)
        q, key, v = qkv.unbind(2)
        q, key, v = (t.transpose(1, 2) for t in (q, key, v))  # (B, n_heads, k, head_dim)

        if self.num_sinks > 0:
            sink_k = self.sink_k.view(self.num_sinks, self.n_heads, self.head_dim)
            sink_v = self.sink_v.view(self.num_sinks, self.n_heads, self.head_dim)
            sink_k = sink_k.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)  # (B, n_heads, num_sinks, head_dim)
            sink_v = sink_v.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
            key_full = torch.cat([sink_k, key], dim=2)  # (B, n_heads, num_sinks+k, head_dim)
            v_full = torch.cat([sink_v, v], dim=2)
            positions_q = torch.arange(k, device=x.device)
            positions_kv = torch.cat([
                positions_q.new_full((self.num_sinks,), -1),  # sinks: always visible, never masked out
                positions_q])
            causal = (positions_kv[None, :] <= positions_q[:, None]) | (positions_kv[None, :] == -1)
            attn_out = F.scaled_dot_product_attention(q, key_full, v_full, attn_mask=causal)
        else:
            attn_out = F.scaled_dot_product_attention(q, key, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(b, k, d)
        x = x + self.out_proj(attn_out)
        return x + self.mlp(self.ln2(x))


class HZLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, memory_slots: int = 8,
                 workspace_slots: int = 32, gate_hidden: int = 16, n_rounds_l1: int = 8, n_qa_labels: int = 4,
                 n_read_labels: int = 2, n_arith_labels: int = 9, block_mixer_heads: int = 4,
                 block_mixer_sinks: int = 4, chat_only: bool = False):
        """chat_only=True (2026-09-16 AMX-execution-geometry pass, real
        chat/reasoning target sizing): skips building the L1-L6 Nursery-
        curriculum heads entirely (object_encoder, sel_rq/rk/rv,
        object_state_encoder, consequence_head, count_head, qa_rq/rk/
        qa_head, read_null_x, read_head, arithmetic_head) -- real,
        measured ~26% of total parameters at d_model=3072 (see
        scripts/hz_chat_model_sizing.py), and none of them are ever
        touched by lm_forward/lm_forward_blocked/generate (the only
        paths a deployed chat/reasoning model actually calls) -- dead
        weight in a chat checkpoint, not a training-time-only cost.
        Calling ground_forward/verb_forward/count_forward/qa_forward/
        read_forward/rule_forward/arithmetic_forward on a chat_only
        model raises AttributeError on the missing module, by design --
        no silent stub, no fallback."""
        super().__init__()
        self.D = d_model
        self.vocab_size = vocab_size
        self.n_rounds_l1 = n_rounds_l1
        self.chat_only = chat_only

        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(
            n_embd=d_model, memory_slots=memory_slots, gate_hidden=gate_hidden))

        value_dim = d_model // 2  # KEEP: D/2 value/write
        self.ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
            n_embd=d_model, workspace_slots=workspace_slots, gate_hidden=gate_hidden,
            allow_ablation_slots=workspace_slots > 8, value_dim=value_dim))
        # default config: identity_biased/bounded_residual/bounded_accumulating
        # all False -- the plain LN recurrence, per the plan's own KEEP list.

        # Experiment 3 (2026-09-16 AMX-execution-geometry pass, section 3):
        # block H recurrence -- used only by lm_forward_blocked, a SEPARATE
        # entry point from lm_forward (which stays completely untouched,
        # per-token H updates exactly as before). See LocalCausalMixer.
        self.local_mixer = LocalCausalMixer(d_model, n_heads=block_mixer_heads, num_sinks=block_mixer_sinks)

        # L0 next-token readout: cross-attention from H against H itself
        # (H is the only state available at a given token position),
        # then a classifier over the vocabulary.
        self.lm_rq = nn.Linear(d_model, d_model, bias=False)
        self.lm_rk = nn.Linear(d_model, d_model, bias=False)
        self.lm_rv = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # kept as three separate nn.Linear modules above (so any existing
        # checkpoint's state_dict keys/shapes are untouched); _packed_lm_read
        # below packs their WEIGHTS into one (3D, D) GEMM at call time --
        # same Parameter tensors, no copies, gradients flow identically to
        # lm_rq/lm_rk/lm_rv.weight either way. Real dispatch reduction
        # (2026-09-16 AMX-execution-geometry pass, section 6): this readout
        # runs once per token in both lm_forward's loop and generate's
        # decode loop, so 3 GEMM launches -> 1 is a real per-token saving,
        # not a one-time cost.

        if self.chat_only:
            return  # L1-L6 Nursery-curriculum heads below intentionally skipped -- see docstring above.

        # L1 object encoder + selection readout. FactorizedObjectEncoder
        # (promoted default, see its docstring) -- one embedding per
        # attribute (type/color/size/position), summed.
        self.object_encoder = FactorizedObjectEncoder(d_model)
        self.sel_rq = nn.Linear(d_model, d_model, bias=False)
        self.sel_rk = nn.Linear(d_model, d_model, bias=False)

        # L2 object+state encoder + selection/consequence readout.
        # Deliberately a SEPARATE encoder from L1's object_encoder rather
        # than widening it -- L2 adds held(2)+opened(2) to the feature
        # layout (16 total) and this keeps L1's already-validated weights
        # untouched (one change at a time). sel_rq/sel_rk are REUSED from
        # L1 (same "which object does the instruction mean" mechanism);
        # sel_rv + consequence_head are new, L2-only.
        self.object_state_encoder = nn.Linear(4 + 4 + 2 + 2 + 2 + 2, d_model, bias=False)
        self.sel_rv = nn.Linear(d_model, d_model, bias=False)
        # Takes [selected_pre_state ; pooled_H] -- pre-state ALONE can only
        # ever express "copy the object as-is"; pooled_H is what actually
        # carries the verb (it reasoned over S, which ingested the
        # instruction). Concatenating both is what makes "verb-conditioned
        # transform of pre-state" expressible at all, real fix for a real
        # bug (see hz_nursery_train.py run 2026-09-04: without this, the
        # model converged to the copy-pre-state baseline, 0.80 accuracy,
        # not real verb-consequence learning).
        self.consequence_head = nn.Linear(d_model * 2, 3, bias=True)  # [position_after, held_after, opened_after] logits

        # L4 numbers: verification head over pooled H. Reuses L1's
        # encode_objects (type/color/size/position) unchanged -- counting
        # needs no new object features, just a different readout that
        # AGGREGATES over the object set instead of pointing at one object.
        self.count_head = nn.Linear(d_model, 1, bias=True)

        # L5 teacher/student QA: recall a synthetic label that exists
        # ONLY in the teach utterance, never in encode_objects' features.
        # Reuses L1's object encoder (the question still needs to
        # resolve WHICH object) -- only the readout is new.
        self.qa_rq = nn.Linear(d_model, d_model, bias=False)
        self.qa_rk = nn.Linear(d_model, d_model, bias=False)
        self.qa_head = nn.Linear(d_model, n_qa_labels, bias=True)

        # L6 simple reading: no parallel object-feature-set input exists
        # for this task at all (every fact is language that was read,
        # nothing to point a cross-attention query at) -- read_null_x is
        # a small learned placeholder standing in for HZCQReasoningWorkspace.run's
        # required x_hidden argument, so H still reasons, just entirely
        # over what accumulated in S from reading the passage.
        self.read_null_x = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.read_head = nn.Linear(d_model, n_read_labels, bias=True)

        # School-0 (section 8.2): arithmetic gets its own head (a
        # different label space, sums 0-8); the logic/causal-rule task
        # reuses read_head as-is (same SIZES label space, see
        # rule_forward) -- no new parameters needed for that one.
        self.arithmetic_head = nn.Linear(d_model, n_arith_labels, bias=True)

    # ---- Stage L0: pure self-supervised next-token LM ----

    def _packed_lm_read(self, H: torch.Tensor) -> torch.Tensor:
        """Real fusion (2026-09-16 AMX-execution-geometry pass, section
        6): lm_rq(H), lm_rk(H), lm_rv(H) are three separate Linear(D,D)
        applications to the SAME H tensor -- pack their weights into one
        (3D, D) GEMM and split the result, same discipline as
        HZCQReasoningWorkspace._packed_q. Uses the exact same Parameter
        tensors as the three separate nn.Linear modules (no copies), so
        gradients flow to lm_rq/lm_rk/lm_rv.weight the same way as
        calling them separately. NOT bit-exact against calling them
        separately (verified: max abs diff ~1.8e-7 on logits, ~2.8e-7 on
        gradients, both well under atol=1e-5/rtol=1e-4) -- ordinary
        float32 GEMM accumulation-order noise from packing three D-wide
        matmuls into one 3D-wide matmul, not a correctness issue (unlike
        the x-attention single-source fast path in
        hz0h_bdh_hzcq_v1_reasoning_workspace_torch.py, which IS bit-exact
        since softmax(single logit)=1.0 identically regardless of
        matmul order)."""
        D = self.D
        W_packed = torch.cat([self.lm_rq.weight, self.lm_rk.weight, self.lm_rv.weight], dim=0)  # (3D, D)
        packed = F.linear(H, W_packed)  # (B, M_H, 3D)
        return packed[..., :D], packed[..., D:2 * D], packed[..., 2 * D:]

    def _lm_forward_step(self, H: torch.Tensor, K_S: torch.Tensor, V_S: torch.Tensor,
                          s_summary: torch.Tensor, token_id_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One token's worth of `lm_forward`'s per-token loop body,
        factored out so `torch.utils.checkpoint.checkpoint` can wrap it
        -- real, disclosed reason: this loop is the exact bottleneck
        Stage 0 measured on `combined_best` BDH before checkpointing
        (5x memory reduction there); `HZLanguageModel`'s own per-token
        recurrence retains one round of activations per token for
        backward with no checkpointing applied anywhere before this."""
        x_t = self.token_embed(token_id_t).unsqueeze(1)  # (B, 1, D) -- current token
        K_x, V_x = self.ws.read_x.project_kv(x_t)
        H = self.ws._step_with_cache(H, K_S, V_S, K_x, V_x, s_summary)
        q, k, v = self._packed_lm_read(H)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.D ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), v).mean(dim=1)
        return H, self.lm_head(read)

    def lm_forward(self, token_ids: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        """token_ids: (B, T). Returns logits (B, T-1, vocab_size) predicting
        token_ids[:, 1:] from token_ids[:, :-1], teacher-forced.

        `gradient_checkpointing=True` wraps each per-token step in
        `torch.utils.checkpoint.checkpoint` -- discards that step's
        activations and recomputes them during backward instead of
        retaining all T-1 steps' worth simultaneously. Default False,
        preserving every existing caller's exact behavior; verified
        bit-identical logits/gradients against the uncheckpointed path
        (`tests/test_hz_language_model_gradient_checkpointing.py`) --
        a pure memory/compute tradeoff, not an approximation, same
        discipline already validated for `combined_best` BDH.

        Real, confirmed, zero-risk fix (found while investigating HZ-
        Micro's real training-speed gap vs a matched transformer):
        S never changes across this whole call (never touched by
        `mem.update` -- "untouched, no prior lifetime evidence for one
        sentence" below), yet the original implementation called
        `ws.step(H, S, x_t)` every single token, which re-derives S's
        K/V projection and gate summary from scratch each time via the
        UNCACHED path -- exactly the "direct step callers (tests,
        diagnostics)" case `HZCQReasoningWorkspace._gate`'s own
        docstring warns is not the production path. `run()` (used by
        every OTHER production forward in this class -- qa_forward,
        read_forward, cs_program_forward, etc.) already caches this via
        `project_kv`/`_step_with_cache` (plan section 11.3); lm_forward
        alone never adopted it. Fixed here by caching S's K/V/summary
        ONCE (verified via torch.allclose against the old per-token
        implementation: bit-for-bit identical output, ~1.06x faster)
        while still recomputing x_t's own K/V fresh every step, since
        x_t genuinely changes each token (unlike S)."""
        B, T = token_ids.shape
        S = self.mem.init_state(B, device=token_ids.device)  # untouched -- no prior lifetime evidence for one sentence
        H = self.ws.init_state(B, device=token_ids.device)
        K_S, V_S = self.ws.read_s.project_kv(S)
        s_summary = S.mean(dim=1, keepdim=True)
        logits_seq = []
        for t in range(T - 1):
            if gradient_checkpointing and torch.is_grad_enabled():
                H, logits_t = torch.utils.checkpoint.checkpoint(
                    self._lm_forward_step, H, K_S, V_S, s_summary, token_ids[:, t], use_reentrant=False)
            else:
                H, logits_t = self._lm_forward_step(H, K_S, V_S, s_summary, token_ids[:, t])
            logits_seq.append(logits_t)
        return torch.stack(logits_seq, dim=1)

    def _lm_forward_block_step(self, H: torch.Tensor, K_S: torch.Tensor, V_S: torch.Tensor,
                               s_summary: torch.Tensor, block_ids: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
        """One block's worth of `lm_forward_blocked`'s per-block loop
        body, factored out so it can be wrapped in
        torch.utils.checkpoint.checkpoint -- same rationale as
        `_lm_forward_step`. block_ids: (B, k), k = this block's real
        length. Returns (H_b, logits_for_this_block (B, k, vocab)).

        Causal safety (the one property this whole experiment depends
        on -- see tests/test_hz_block_recurrence.py's explicit test):
        `H` passed in is H_{b-1}, fixed for the WHOLE block -- it is
        used to condition local_mixer's input but is NOT recomputed
        until after all of this block's logits are already produced.
        `block_summary` (used to derive the returned H_b) pools over the
        ENTIRE block including its LAST position, so H_b necessarily
        depends on tokens the earlier positions in this same block
        haven't seen yet -- that is exactly why H_b is returned
        SEPARATELY from `logits`, for the CALLER to use only in the
        NEXT block, never fed back into this block's own logits."""
        x_block = self.token_embed(block_ids)  # (B, k, D)
        # H-conditioning: simplest choice per the plan's own instruction
        # ("do not build a complicated router") -- a single pooled
        # H_{b-1} vector, additive, identical for every position in this
        # block (H_{b-1} itself does not vary within the block by
        # construction). A real, disclosed simplification: this does NOT
        # let different block positions read DIFFERENT parts of H_{b-1}
        # (e.g. via cross-attention into H's individual M_H slots) --
        # that richer mechanism is a natural follow-up, not implemented
        # in this first pass, consistent with "one architecture change
        # at a time."
        h_cond = H.mean(dim=1, keepdim=True)  # (B, 1, D), fixed across the block
        local_hidden = self.local_mixer(x_block + h_cond)  # (B, k, D), causal within the block
        logits_block = self.lm_head(local_hidden)  # (B, k, vocab)

        block_summary = local_hidden.mean(dim=1, keepdim=True)  # (B, 1, D) -- the block's "evidence" for H
        # block_summary has source length 1 -- Experiment 1's exact
        # single-source fast path (attend_with_q) fires here for free,
        # same as the per-token path did.
        K_x, V_x = self.ws.read_x.project_kv(block_summary)
        H_b = self.ws._step_with_cache(H, K_S, V_S, K_x, V_x, s_summary)  # ONE H update for this whole block
        return H_b, logits_block

    def lm_forward_blocked(self, token_ids: torch.Tensor, block_size: int = 16,
                           gradient_checkpointing: bool = False) -> torch.Tensor:
        """Experiment 3 (2026-09-16 AMX-execution-geometry pass, section
        2-3): H updates once per `block_size` tokens instead of once per
        token -- T/K H-transitions instead of T. SEPARATE entry point
        from `lm_forward`, which is completely unmodified (same
        signature, same behavior, still the per-token path) -- this is
        an opt-in alternative forward, not a replacement.

        Same teacher-forced convention as lm_forward: token_ids (B, T),
        returns logits (B, T-1, vocab_size) predicting token_ids[:, 1:]
        from token_ids[:, :-1]. S stays completely untouched (same
        init-only S as lm_forward -- Phase A/B separation, section 10:
        do not change S behavior in the same pass as block recurrence).

        Real, disclosed scope: does NOT claim a wall-clock speedup by
        itself (local_mixer's causal attention still runs real
        per-token work within each block) -- what changes is the COUNT
        of expensive H transitions (HZCQReasoningWorkspace._step_with_cache
        calls), from T-1 down to ceil((T-1)/block_size). See
        scripts/hz_block_recurrence_bench.py for the actual measured
        H-transition count and throughput comparison against lm_forward."""
        B, T = token_ids.shape
        S = self.mem.init_state(B, device=token_ids.device)
        H = self.ws.init_state(B, device=token_ids.device)
        K_S, V_S = self.ws.read_s.project_kv(S)
        s_summary = S.mean(dim=1, keepdim=True)

        x_ids = token_ids[:, :-1]  # (B, T-1) -- same predicting positions as lm_forward
        predict_len = x_ids.shape[1]
        logits_chunks = []
        for start in range(0, predict_len, block_size):
            end = min(start + block_size, predict_len)
            block_ids = x_ids[:, start:end]
            if gradient_checkpointing and torch.is_grad_enabled():
                H, logits_block = torch.utils.checkpoint.checkpoint(
                    self._lm_forward_block_step, H, K_S, V_S, s_summary, block_ids, use_reentrant=False)
            else:
                H, logits_block = self._lm_forward_block_step(H, K_S, V_S, s_summary, block_ids)
            logits_chunks.append(logits_block)
        return torch.cat(logits_chunks, dim=1)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 60,
                 eos_id: int | None = None, greedy: bool = True, temperature: float = 1.0) -> list[int]:
        """Real, first free-form generation capability for this class
        (HZ-Chat-Micro v0) -- every forward before this was either a
        classification readout or `lm_forward`'s teacher-forced,
        parallel-over-labels loss; nothing could actually produce open-
        ended text. `prompt_ids`: (1, T) -- batch size 1 only for now.

        Processes the prompt through the EXACT SAME per-token H-stepping
        `lm_forward` uses (same cached-S mechanism, same order), so a
        model trained via `lm_forward` on prompt+response sequences
        concatenated together sees IDENTICAL H dynamics at inference
        time -- after the last prompt token, `lm_forward`'s own indexing
        would next predict the token immediately following the prompt,
        which is exactly what happens here. Then continues
        autoregressively: sample one new token from the current
        readout, embed it, step H forward with it, repeat.

        Real, disclosed scope for v0: `S` stays at its untouched init
        state throughout (matching `lm_forward` exactly) -- persistent
        cross-turn memory via real `mem.update` calls is real, valuable
        follow-up work, not yet wired into generation."""
        B, T = prompt_ids.shape
        assert B == 1, "generate() supports batch size 1 for now"
        S = self.mem.init_state(B, device=prompt_ids.device)
        H = self.ws.init_state(B, device=prompt_ids.device)
        K_S, V_S = self.ws.read_s.project_kv(S)
        s_summary = S.mean(dim=1, keepdim=True)

        def step_with(token_id_tensor):
            x_t = self.token_embed(token_id_tensor).unsqueeze(1)
            K_x, V_x = self.ws.read_x.project_kv(x_t)
            return self.ws._step_with_cache(H, K_S, V_S, K_x, V_x, s_summary)

        for t in range(T):
            H = step_with(prompt_ids[:, t])

        generated: list[int] = []
        for _ in range(max_new_tokens):
            q, k, v = self._packed_lm_read(H)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.D ** 0.5)
            read = torch.matmul(F.softmax(scores, dim=-1), v).mean(dim=1)
            logits = self.lm_head(read)
            if greedy:
                next_id = logits.argmax(-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)
            next_id_val = int(next_id.item())
            generated.append(next_id_val)
            if eos_id is not None and next_id_val == eos_id:
                break
            H = step_with(next_id)
        return generated

    @torch.no_grad()
    def generate_blocked(self, prompt_ids: torch.Tensor, max_new_tokens: int = 60,
                         block_size: int = 32, eos_id: int | None = None,
                         greedy: bool = True, temperature: float = 1.0) -> list[int]:
        """Real fix (2026-09-18): `generate()` above uses the OLD per-token
        H-stepping path, but every checkpoint actually trained on this
        model used `lm_forward_blocked` (K=32 block recurrence) -- a real
        train/inference mismatch, plausibly a real contributor to the
        incoherent generation seen from the BPE-tokenizer checkpoint.
        This mirrors `_lm_forward_block_step`/`lm_forward_blocked` exactly:
        H is fixed (as H_{b-1}) for an entire block, local_mixer runs
        causally over the block-so-far with that fixed H-conditioning
        bias, and H only transitions once a block reaches `block_size`
        tokens (matching training's chunking -- a block only transitions
        early, at less than block_size tokens, if generation stops mid-
        block via `eos_id`, in which case H is simply never advanced past
        that partial block, matching how `lm_forward_blocked` would never
        see a not-yet-closed chunk either).

        prompt_ids: (1, T) -- batch size 1 only for now, same as generate()."""
        B, T = prompt_ids.shape
        assert B == 1, "generate_blocked() supports batch size 1 for now"
        S = self.mem.init_state(B, device=prompt_ids.device)
        H = self.ws.init_state(B, device=prompt_ids.device)
        K_S, V_S = self.ws.read_s.project_kv(S)
        s_summary = S.mean(dim=1, keepdim=True)

        current_chunk_ids: list[int] = []

        def process_token(token_id_int: int) -> torch.Tensor:
            nonlocal H, current_chunk_ids
            current_chunk_ids.append(token_id_int)
            chunk_tensor = torch.tensor([current_chunk_ids], device=prompt_ids.device)
            h_cond = H.mean(dim=1, keepdim=True)
            local_hidden = self.local_mixer(self.token_embed(chunk_tensor) + h_cond)
            logits_last = self.lm_head(local_hidden[:, -1:])
            if len(current_chunk_ids) == block_size:
                block_summary = local_hidden.mean(dim=1, keepdim=True)
                K_x, V_x = self.ws.read_x.project_kv(block_summary)
                H = self.ws._step_with_cache(H, K_S, V_S, K_x, V_x, s_summary)
                current_chunk_ids = []
            return logits_last

        logits = None
        for t in range(T):
            logits = process_token(int(prompt_ids[0, t].item()))

        generated: list[int] = []
        for _ in range(max_new_tokens):
            if greedy:
                next_id = logits[:, -1].argmax(-1)
            else:
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)
            next_id_val = int(next_id.item())
            generated.append(next_id_val)
            if eos_id is not None and next_id_val == eos_id:
                break
            logits = process_token(next_id_val)
        return generated

    # ---- Stage L1: grounded nouns/properties ----

    def encode_objects(self, type_idx: torch.Tensor, color_idx: torch.Tensor,
                        size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """Each *_idx: (B, N_obj) long. Returns (B, N_obj, D)."""
        return self.object_encoder(type_idx, color_idx, size_idx, position_idx)

    def ground_forward(self, instruction_ids: torch.Tensor, type_idx: torch.Tensor, color_idx: torch.Tensor,
                        size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """instruction_ids: (B, T). Returns selection logits (B, N_obj)."""
        B = instruction_ids.shape[0]
        x_objects = self.encode_objects(type_idx, color_idx, size_idx, position_idx)  # (B, N_obj, D)

        instr_hiddens = [self.token_embed(instruction_ids[:, t]).unsqueeze(1) for t in range(instruction_ids.shape[1])]
        S = self.mem.update_sequence(B, instr_hiddens)  # real: ingest the instruction into persistent memory

        H = self.ws.run(B, S, x_objects, n_rounds=self.n_rounds_l1)  # (B, M_H, D)

        q = self.sel_rq(H).mean(dim=1, keepdim=True)  # (B, 1, D) -- pooled query over the reasoning workspace
        scores = torch.matmul(q, self.sel_rk(x_objects).transpose(-1, -2)) / (self.D ** 0.5)  # (B, 1, N_obj)
        return scores.squeeze(1)

    # ---- Stage L2: verbs through consequences ----

    def encode_objects_with_state(self, type_idx: torch.Tensor, color_idx: torch.Tensor, size_idx: torch.Tensor,
                                   position_idx: torch.Tensor, held: torch.Tensor, opened: torch.Tensor) -> torch.Tensor:
        """Like encode_objects but with two extra real object-state bits
        (held, opened) that verbs actually change. held/opened: (B, N_obj)
        bool/long. Returns (B, N_obj, D)."""
        feat = torch.cat([
            F.one_hot(type_idx, 4).float(),
            F.one_hot(color_idx, 4).float(),
            F.one_hot(size_idx, 2).float(),
            F.one_hot(position_idx, 2).float(),
            F.one_hot(held.long(), 2).float(),
            F.one_hot(opened.long(), 2).float(),
        ], dim=-1)
        return self.object_state_encoder(feat)

    def verb_forward(self, instruction_ids: torch.Tensor, type_idx: torch.Tensor, color_idx: torch.Tensor,
                      size_idx: torch.Tensor, position_idx: torch.Tensor, held: torch.Tensor,
                      opened: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """instruction_ids encode "{verb} the {color} object". Returns
        (selection_logits (B, N_obj), consequence_logits (B, 3)) where
        consequence_logits are [position_after, held_after, opened_after],
        each a binary logit -- a real, structured prediction of the verb's
        EFFECT on the referenced object's pre-action state, not a fixed
        classifier over verb identity. Reuses the exact S-ingests-
        instruction / H-reasons-over-S-and-objects pattern as ground_forward:
        S carries "what the instruction said" (which verb, which object),
        x_objects carries "what's actually there right now" (the object's
        real pre-action state), and H fuses the two."""
        B = instruction_ids.shape[0]
        x_objects = self.encode_objects_with_state(type_idx, color_idx, size_idx, position_idx, held, opened)

        instr_hiddens = [self.token_embed(instruction_ids[:, t]).unsqueeze(1) for t in range(instruction_ids.shape[1])]
        S = self.mem.update_sequence(B, instr_hiddens)

        H = self.ws.run(B, S, x_objects, n_rounds=self.n_rounds_l1)  # (B, M_H, D)

        q = self.sel_rq(H).mean(dim=1, keepdim=True)  # (B, 1, D)
        sel_scores = torch.matmul(q, self.sel_rk(x_objects).transpose(-1, -2)) / (self.D ** 0.5)  # (B, 1, N_obj)
        attn = F.softmax(sel_scores, dim=-1)  # (B, 1, N_obj) -- soft pointer at the referenced object
        selected = torch.matmul(attn, self.sel_rv(x_objects)).squeeze(1)  # (B, D) -- its pre-action representation
        pooled_h = H.mean(dim=1)  # (B, D) -- carries the verb: H reasoned over S, which ingested the instruction

        consequence_logits = self.consequence_head(torch.cat([selected, pooled_h], dim=-1))  # (B, 3)
        return sel_scores.squeeze(1), consequence_logits

    # ---- Stage L4: numbers (counting verification) ----

    def encode_and_reason(self, instruction_ids: torch.Tensor, type_idx: torch.Tensor, color_idx: torch.Tensor,
                           size_idx: torch.Tensor, position_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared backbone computation for any task needing (x_objects, H)
        -- everything EXCEPT the final readout. Factored out so a readout
        ablation (see reference/hz_nursery_counting_readouts.py) can hold
        the backbone (token_embed/mem/ws/object_encoder) completely fixed
        and swap only what reads H, without touching this method or the
        underlying HZCQReasoningWorkspace recurrence at all."""
        B = instruction_ids.shape[0]
        x_objects = self.encode_objects(type_idx, color_idx, size_idx, position_idx)
        instr_hiddens = [self.token_embed(instruction_ids[:, t]).unsqueeze(1) for t in range(instruction_ids.shape[1])]
        S = self.mem.update_sequence(B, instr_hiddens)
        H = self.ws.run(B, S, x_objects, n_rounds=self.n_rounds_l1)  # (B, M_H, D)
        return x_objects, H

    def verify_count_forward(self, instruction_ids: torch.Tensor, type_idx: torch.Tensor, color_idx: torch.Tensor,
                              size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """instruction_ids encode "are there {number} {value} objects".
        Returns a single verification logit per batch element (B,) --
        does the stated number match the true count of matching objects?
        Uses the SAME S-ingests-instruction / H-reasons-over-S-and-objects
        pattern as ground_forward, but the readout AGGREGATES over the
        whole object set via pooled H instead of pointing at one object --
        a real test of whether the reasoning workspace can accumulate a
        quantity, not just select. This IS the "mean-pool" readout variant
        (see the counting-readout ablation) -- kept as the default/baseline
        head on the model itself since it's the one plans/Hatchling world.md
        already reports a real result for."""
        _, H = self.encode_and_reason(instruction_ids, type_idx, color_idx, size_idx, position_idx)
        pooled_h = H.mean(dim=1)  # (B, D)
        return self.count_head(pooled_h).squeeze(-1)  # (B,)

    # ---- Stage L5: teacher/student QA (one-shot novel-word recall) ----

    def qa_forward(self, teach_ids: torch.Tensor, question_ids: torch.Tensor, type_idx: torch.Tensor,
                    color_idx: torch.Tensor, size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """teach_ids/question_ids: (B, T) each, two REAL SEPARATE turns.
        Chains them into S via two sequential mem.update() calls (not one
        concatenated update_sequence call) so there is a genuine turn
        boundary: S after the teach utterance is exactly what a "student"
        would carry into the question, and the question's own tokens
        update that already-taught S further before H ever reads it.
        Returns label logits (B, n_qa_labels) -- the correct label exists
        ONLY in teach_ids, never in the object features, so this can only
        be solved by real recall through S, not grounding to x_objects.

        Each turn is ingested as ONE WHOLE-SENTENCE mem.update call
        (T_demo = turn length), not token-by-token -- promoting this
        session's own real, verified fix (softmax over T_demo=1 is
        always exactly 1.0, so token-by-token ingestion structurally
        forces delta_S identical across every slot, see the L5 memory-
        cliff diagnostic thread) into production. Whole-sentence
        ingestion alone moved 3-fact recall 24.5% -> 33.3% in that
        diagnostic; this was the one production forward method still
        using the old per-token loop."""
        B = teach_ids.shape[0]
        x_objects = self.encode_objects(type_idx, color_idx, size_idx, position_idx)

        S = self.mem.init_state(B, device=teach_ids.device)
        S = self.mem.update(S, self.token_embed(teach_ids))
        S = self.mem.update(S, self.token_embed(question_ids))

        H = self.ws.run(B, S, x_objects, n_rounds=self.n_rounds_l1)  # (B, M_H, D)
        q = self.qa_rq(H).mean(dim=1, keepdim=True)  # (B, 1, D)
        scores = torch.matmul(q, self.qa_rk(H).transpose(-1, -2)) / (self.D ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)  # (B, D)
        return self.qa_head(read)  # (B, n_qa_labels)

    # ---- Stage L6: simple reading (multi-sentence passage, selective recall) ----

    def read_forward(self, sentence_ids_list: list[torch.Tensor], question_ids: torch.Tensor) -> torch.Tensor:
        """sentence_ids_list: list of (B, T_k) tensors, one passage read
        one sentence at a time (real sequential turns into S, extending
        qa_forward's 2-turn chain to len(sentence_ids_list)+1 turns).
        question_ids: (B, T_q), the final turn. Returns label logits
        (B, n_read_labels). No object-feature-set input at all -- every
        fact is language that was read, so correctness depends entirely
        on S having retained (and H having selected) the ONE relevant
        sentence among several, not on grounding to a visible feature.

        Each sentence/question is ingested as ONE WHOLE-SENTENCE
        mem.update call (T_demo = sentence length), not token-by-token --
        promoting this session's own real, verified fix into production
        (see qa_forward's docstring for the underlying math; this method
        was the second of three production forwards still using the old
        per-token loop, alongside qa_forward and stress_recall_forward)."""
        B = question_ids.shape[0]
        S = self.mem.init_state(B, device=question_ids.device)
        for sentence_ids in sentence_ids_list:
            S = self.mem.update(S, self.token_embed(sentence_ids))
        S = self.mem.update(S, self.token_embed(question_ids))

        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)  # (B, M_H, D)
        pooled = H.mean(dim=1)
        return self.read_head(pooled)  # (B, n_read_labels)

    # ---- L5 memory stress test: multi-fact recall + distractor interference ----

    def stress_recall_forward(self, sequence_ids_list: list[torch.Tensor], question_ids: torch.Tensor) -> torch.Tensor:
        """Generalizes qa_forward's 2-turn (teach, question) chain to
        len(sequence_ids_list)+1 turns -- some of those turns are real
        taught facts, some are plain distractor sentences carrying no
        fact at all (see generate_l5_stress_episode), and S sees them in
        the SAME interleaved order a real multi-turn interaction would.
        Reuses qa_forward's readout (qa_rq/qa_rk/qa_head, same label
        space) and read_forward's turn-chaining + null-x mechanism
        (no object-feature-set input -- every fact is language read
        through S, same as L6).

        Each turn is ingested as ONE WHOLE-SENTENCE mem.update call
        (T_demo = turn length), not token-by-token -- promoting this
        session's own real, verified fix into production (see
        qa_forward's docstring; this was the third of three production
        forwards still using the old per-token loop)."""
        B = question_ids.shape[0]
        S = self.mem.init_state(B, device=question_ids.device)
        for sentence_ids in sequence_ids_list:
            S = self.mem.update(S, self.token_embed(sentence_ids))
        S = self.mem.update(S, self.token_embed(question_ids))

        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)  # (B, M_H, D)
        q = self.qa_rq(H).mean(dim=1, keepdim=True)
        scores = torch.matmul(q, self.qa_rk(H).transpose(-1, -2)) / (self.D ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
        return self.qa_head(read)  # (B, n_qa_labels)

    # ---- School-0: arithmetic and conditional-rule reasoning ----

    def arithmetic_forward(self, instruction_ids: torch.Tensor) -> torch.Tensor:
        """instruction_ids encode "{a} plus {b} equals". Single-turn
        ingestion into S (no teach/question split -- the whole problem
        is one utterance), reasoning over S and a null placeholder (no
        object-feature-set input, same as L6/L5-stress), classified into
        the sum via arithmetic_head."""
        B = instruction_ids.shape[0]
        S = self.mem.init_state(B, device=instruction_ids.device)
        for t in range(instruction_ids.shape[1]):
            S = self.mem.update(S, self.token_embed(instruction_ids[:, t]).unsqueeze(1))
        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)
        pooled = H.mean(dim=1)
        return self.arithmetic_head(pooled)  # (B, n_arith_labels)

    def rule_forward(self, rule_ids: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        """rule_ids encode a GENERAL conditional ("if an object is
        {color} then it is {size}"), question_ids ask about a specific
        instance identified by the rule's own premise. Structurally
        identical to qa_forward's 2-turn chain (teach, then question)
        -- the difference is semantic, not architectural: this is a
        RULE to apply to a query, not a FACT to retrieve verbatim.
        Reuses read_head (the same SIZES label space L6 already uses),
        no new parameters."""
        B = question_ids.shape[0]
        S = self.mem.init_state(B, device=question_ids.device)
        for t in range(rule_ids.shape[1]):
            S = self.mem.update(S, self.token_embed(rule_ids[:, t]).unsqueeze(1))
        for t in range(question_ids.shape[1]):
            S = self.mem.update(S, self.token_embed(question_ids[:, t]).unsqueeze(1))
        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)
        pooled = H.mean(dim=1)
        return self.read_head(pooled)  # (B, n_read_labels) -- same space as SIZES

    def cs_program_forward(self, statement_ids_list: list[torch.Tensor], question_ids: torch.Tensor) -> torch.Tensor:
        """School-0 Computer Science: "program execution" -- a real
        symbol table (2 variable assignments, "x is {a}", "y is {b}")
        must be tracked before their values can be substituted into
        "what is x plus y". Each statement is ingested as ONE WHOLE-
        SENTENCE chunk (T_demo = statement length, not token-by-token)
        -- applying this session's own real finding (a fixed
        mathematical fact: softmax over T_demo=1 is always exactly
        1.0, so token-by-token ingestion structurally forces delta_S
        identical across every slot) from the start here, rather than
        repeating the bug. Classifies via arithmetic_head (same label
        space as arithmetic_forward -- program execution's answer is
        also a sum)."""
        B = question_ids.shape[0]
        S = self.mem.init_state(B, device=question_ids.device)
        for stmt_ids in statement_ids_list:
            hidden = self.token_embed(stmt_ids)  # (B, T, D) -- whole statement, one mem.update call
            S = self.mem.update(S, hidden)
        question_hidden = self.token_embed(question_ids)
        S = self.mem.update(S, question_hidden)
        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)
        pooled = H.mean(dim=1)
        return self.arithmetic_head(pooled)  # (B, n_arith_labels)

    def physics_forward(self, teach_ids: torch.Tensor, scenario_ids: torch.Tensor,
                         question_ids: torch.Tensor) -> torch.Tensor:
        """School-0 Physics: teaches a comparative-magnitude rule ("a
        large object needs more force than a small object"), then a
        per-episode scenario naming which color is the large/small
        object, then asks which of two named objects needs more force.
        Real relational-inference test: the answer is which of TWO
        entities the rule picks out, not a single premise's conclusion
        (`rule_forward`'s task). Whole-sentence ingestion throughout
        (teach and scenario each as ONE mem.update call, not token-by-
        token), same discipline as `cs_program_forward`. Classifies via
        read_head over COLORS (n_read_labels must be >= len(COLORS) when
        this task is used -- unlike rule_forward's SIZES space, this is
        a genuinely different output space sharing the same head)."""
        B = question_ids.shape[0]
        S = self.mem.init_state(B, device=question_ids.device)
        for ids in (teach_ids, scenario_ids, question_ids):
            hidden = self.token_embed(ids)  # (B, T, D) -- whole sentence, one mem.update call
            S = self.mem.update(S, hidden)
        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)
        pooled = H.mean(dim=1)
        return self.read_head(pooled)  # (B, n_read_labels) -- indexed as a COLORS label here

    def entity_select_forward(self, statement_ids_list: list[torch.Tensor],
                               question_ids: torch.Tensor) -> torch.Tensor:
        """Real 2x2 diagnostic (plan Phase 9, Physics coreference
        ablation's successor): structurally IDENTICAL to
        `cs_program_forward` (same whole-sentence ingestion of a list of
        statements, then a question) -- the only difference is the
        OUTPUT: this classifies into {x, y} via read_head (a REFERENCE
        to which entity satisfies a named property), not a derived
        value via arithmetic_head. Isolates whether entity-selection
        specifically (not composition, not coreference across surface
        tokens) is what this architecture cannot represent."""
        B = question_ids.shape[0]
        S = self.mem.init_state(B, device=question_ids.device)
        for stmt_ids in statement_ids_list:
            hidden = self.token_embed(stmt_ids)  # (B, T, D) -- whole statement, one mem.update call
            S = self.mem.update(S, hidden)
        question_hidden = self.token_embed(question_ids)
        S = self.mem.update(S, question_hidden)
        x_null = self.read_null_x.expand(B, 1, self.D)
        H = self.ws.run(B, S, x_null, n_rounds=self.n_rounds_l1)
        pooled = H.mean(dim=1)
        return self.read_head(pooled)  # (B, n_read_labels) -- indexed as an {x, y} label here
