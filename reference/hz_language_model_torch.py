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


class HZLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, memory_slots: int = 8,
                 workspace_slots: int = 32, gate_hidden: int = 16, n_rounds_l1: int = 8, n_qa_labels: int = 4,
                 n_read_labels: int = 2, n_arith_labels: int = 9):
        super().__init__()
        self.D = d_model
        self.vocab_size = vocab_size
        self.n_rounds_l1 = n_rounds_l1

        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(
            n_embd=d_model, memory_slots=memory_slots, gate_hidden=gate_hidden))

        value_dim = d_model // 2  # KEEP: D/2 value/write
        self.ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
            n_embd=d_model, workspace_slots=workspace_slots, gate_hidden=gate_hidden,
            allow_ablation_slots=workspace_slots > 8, value_dim=value_dim))
        # default config: identity_biased/bounded_residual/bounded_accumulating
        # all False -- the plain LN recurrence, per the plan's own KEEP list.

        # L0 next-token readout: cross-attention from H against H itself
        # (H is the only state available at a given token position),
        # then a classifier over the vocabulary.
        self.lm_rq = nn.Linear(d_model, d_model, bias=False)
        self.lm_rk = nn.Linear(d_model, d_model, bias=False)
        self.lm_rv = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

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

    def lm_forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T). Returns logits (B, T-1, vocab_size) predicting
        token_ids[:, 1:] from token_ids[:, :-1], teacher-forced.

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
            x_t = self.token_embed(token_ids[:, t]).unsqueeze(1)  # (B, 1, D) -- current token
            K_x, V_x = self.ws.read_x.project_kv(x_t)
            H = self.ws._step_with_cache(H, K_S, V_S, K_x, V_x, s_summary)
            q = self.lm_rq(H)
            scores = torch.matmul(q, self.lm_rk(H).transpose(-1, -2)) / (self.D ** 0.5)
            read = torch.matmul(F.softmax(scores, dim=-1), self.lm_rv(H)).mean(dim=1)
            logits_seq.append(self.lm_head(read))
        return torch.stack(logits_seq, dim=1)

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
