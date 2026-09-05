"""HZ-World agent, plans/Hatchling world.md Phase 2 (HZ adapter).

Wires the validated S+H architecture -- HZCQPersistentMemory (S) and
HZCQReasoningWorkspace (H) -- to Hatchling World's real observations
and fixed action space. Per the plan's own Phase 2 checklist (section
22): original LN recurrence only, M_H=32, D/2 value/write, exact Q/K,
persistent S updated after observing REAL action consequences, fixed
action head, NO new recurrence experiments. Reuses both modules
exactly as validated on the FSM task family -- zero architecture
changes, only new glue code around them.

Real per-environment-step loop, matching section 3.2/3.3:

    H_{t,0}   = H_init                              (fixed, from ws)
    H_{t,r+1} = F_theta(H_{t,r}, S_t, o_t)           (ws.run, R rounds)
    a_t       = Policy(H_{t,R}, o_t)                 (act())
    -- environment executes a_t, returns (o_{t+1}, r_t, done) --
    S_{t+1}   = U_theta(S_t, o_t, a_t, r_t, o_{t+1})  (update_memory())
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import HZCQPersistentMemory, HZCQPersistentMemoryConfig
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig
from hatchling_world.state import WorldConfig, WorldState


class HZWorldAgent(nn.Module):
    def __init__(self, world_config: WorldConfig, d_model: int = 64, memory_slots: int = 8,
                 workspace_slots: int = 32, gate_hidden: int = 16, n_rounds: int = 8):
        super().__init__()
        self.world_config = world_config
        self.D = d_model
        self.n_rounds = n_rounds
        R, C = world_config.n_rooms, world_config.n_colors

        # Real, fixed-shape observation encoding -- one-hot agent/goal
        # room, flattened door adjacency/lock state, one-hot key color
        # per door pair (C+1 classes: -1..C-1 shifted to 0..C so "no
        # door" is representable), flattened room keys, inventory.
        self.obs_dim = 2 * R + 2 * R * R + R * R * (C + 1) + R * C + C
        self.obs_encoder = nn.Linear(self.obs_dim, d_model, bias=False)

        # Consequence encoder for the real post-action S update
        # (section 3.3): (o_t, a_t, r_t, o_{t+1}) -> D.
        self.action_embed = nn.Embedding(world_config.n_actions, d_model)
        self.reward_proj = nn.Linear(1, d_model, bias=False)
        self.consequence_encoder = nn.Linear(3 * d_model, d_model, bias=False)

        self.mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(
            n_embd=d_model, memory_slots=memory_slots, gate_hidden=gate_hidden))

        value_dim = d_model // 2  # KEEP: D/2 value/write, real confirmed quality-preserving efficiency win
        self.ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
            n_embd=d_model, workspace_slots=workspace_slots, gate_hidden=gate_hidden,
            allow_ablation_slots=workspace_slots > 8, value_dim=value_dim))
        # default config: identity_biased/bounded_residual/bounded_accumulating
        # all False -- the plain LN recurrence, undefeated across five real
        # controlled comparisons this project has run. No new recurrence
        # experiments here, per the plan's own Phase 2 rule.

        self.rq = nn.Linear(d_model, d_model, bias=False)
        self.rk = nn.Linear(d_model, d_model, bias=False)
        self.rv = nn.Linear(d_model, d_model, bias=False)
        self.action_head = nn.Linear(d_model, world_config.n_actions, bias=False)

    def init_memory(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return self.mem.init_state(batch_size, device=device, dtype=dtype)

    def encode_observation(self, state: WorldState) -> torch.Tensor:
        R, C = self.world_config.n_rooms, self.world_config.n_colors
        B = state.batch_size()
        agent_oh = F.one_hot(state.agent_room, R).float()
        goal_oh = F.one_hot(state.goal_room, R).float()
        adj = state.door_adj.float().view(B, -1)
        locked = state.door_locked.float().view(B, -1)
        color_idx = state.door_key_color + 1  # -1..C-1 -> 0..C ("no door" = class 0)
        color_oh = F.one_hot(color_idx, C + 1).float().view(B, -1)
        room_keys = state.room_keys.float().view(B, -1)
        inventory = state.inventory.float()
        flat = torch.cat([agent_oh, goal_oh, adj, locked, color_oh, room_keys, inventory], dim=-1)
        return self.obs_encoder(flat)  # (B, D)

    def act(self, state: WorldState, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Real reasoning step: observation -> R rounds of H -> action
        logits. Returns (logits (B, n_actions), obs_embed (B, D) for
        the caller to reuse in the later memory update)."""
        obs_embed = self.encode_observation(state)
        x_hidden = obs_embed.unsqueeze(1)  # (B, 1, D) -- T_query=1, the current observation
        B = state.batch_size()
        H = self.ws.run(B, S, x_hidden, n_rounds=self.n_rounds)
        q = self.rq(obs_embed).unsqueeze(1)
        scores = torch.matmul(q, self.rk(H).transpose(-1, -2)) / (self.D ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), self.rv(H)).squeeze(1)
        logits = self.action_head(read)
        return logits, obs_embed

    def update_memory(self, S: torch.Tensor, obs_embed: torch.Tensor, action: torch.Tensor,
                       reward: torch.Tensor, next_state: WorldState) -> torch.Tensor:
        """Real persistent-memory update AFTER observing consequences,
        section 3.3: S_{t+1}=U_theta(S_t,o_t,a_t,r_t,o_{t+1})."""
        next_obs_embed = self.encode_observation(next_state)
        a_embed = self.action_embed(action)
        r_embed = self.reward_proj(reward.unsqueeze(-1))
        consequence = self.consequence_encoder(torch.cat([obs_embed, a_embed, r_embed], dim=-1))
        combined = consequence + next_obs_embed
        return self.mem.update(S, combined.unsqueeze(1))
