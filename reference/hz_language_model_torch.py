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
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import HZCQPersistentMemory, HZCQPersistentMemoryConfig
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig


class HZLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, memory_slots: int = 8,
                 workspace_slots: int = 32, gate_hidden: int = 16, n_rounds_l1: int = 8):
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

        # L1 object encoder + selection readout.
        # Feature layout: type(4) + color(4) + size(2) + position(2) = 12,
        # matches hatchling_world.language.tokenizer's NOUNS/COLORS/SIZES/POSITIONS.
        self.object_encoder = nn.Linear(4 + 4 + 2 + 2, d_model, bias=False)
        self.sel_rq = nn.Linear(d_model, d_model, bias=False)
        self.sel_rk = nn.Linear(d_model, d_model, bias=False)

    # ---- Stage L0: pure self-supervised next-token LM ----

    def lm_forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T). Returns logits (B, T-1, vocab_size) predicting
        token_ids[:, 1:] from token_ids[:, :-1], teacher-forced."""
        B, T = token_ids.shape
        S = self.mem.init_state(B, device=token_ids.device)  # untouched -- no prior lifetime evidence for one sentence
        H = self.ws.init_state(B, device=token_ids.device)
        logits_seq = []
        for t in range(T - 1):
            x_t = self.token_embed(token_ids[:, t]).unsqueeze(1)  # (B, 1, D) -- current token
            H = self.ws.step(H, S, x_t)
            q = self.lm_rq(H)
            scores = torch.matmul(q, self.lm_rk(H).transpose(-1, -2)) / (self.D ** 0.5)
            read = torch.matmul(F.softmax(scores, dim=-1), self.lm_rv(H)).mean(dim=1)
            logits_seq.append(self.lm_head(read))
        return torch.stack(logits_seq, dim=1)

    # ---- Stage L1: grounded nouns/properties ----

    def encode_objects(self, type_idx: torch.Tensor, color_idx: torch.Tensor,
                        size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """Each *_idx: (B, N_obj) long. Returns (B, N_obj, D)."""
        feat = torch.cat([
            F.one_hot(type_idx, 4).float(),
            F.one_hot(color_idx, 4).float(),
            F.one_hot(size_idx, 2).float(),
            F.one_hot(position_idx, 2).float(),
        ], dim=-1)
        return self.object_encoder(feat)

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
