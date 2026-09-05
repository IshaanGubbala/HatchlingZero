"""Real structural tests for the HZ-World adapter, plans/Hatchling
world.md Phase 2/section 22 checklist: fixed shapes, gradients reach
policy/memory/readout/recurrent-state paths (section 1.2.B's own
rescue-ladder check, verified directly rather than assumed)."""
from __future__ import annotations

import torch

from hatchling_world.curriculum import generate_school_worlds
from hatchling_world.transition import step as env_step
from reference.hz_world_agent_torch import HZWorldAgent


def _agent(school_level="S1_short_composition", seed=0):
    torch.manual_seed(seed)
    state, config = generate_school_worlds(school_level, batch=1, episode_seed=0)
    agent = HZWorldAgent(config, d_model=32, memory_slots=8, workspace_slots=16, n_rounds=4)
    return agent, state, config


def test_act_returns_correct_shapes():
    agent, state, config = _agent()
    S = agent.init_memory(1)
    logits, obs_embed = agent.act(state, S)
    assert logits.shape == (1, config.n_actions)
    assert obs_embed.shape == (1, agent.D)


def test_memory_update_changes_state_and_preserves_shape():
    agent, state, config = _agent()
    S = agent.init_memory(1)
    logits, obs_embed = agent.act(state, S)
    action = logits.argmax(-1)
    next_state, reward, done = env_step(state, action, config)
    S2 = agent.update_memory(S, obs_embed, action, reward, next_state)
    assert S2.shape == S.shape
    assert not torch.allclose(S, S2)


def test_gradients_flow_to_every_parameter():
    agent, state, config = _agent()
    S = agent.init_memory(1)
    logits, obs_embed = agent.act(state, S)
    action = logits.argmax(-1)
    next_state, reward, done = env_step(state, action, config)
    S2 = agent.update_memory(S, obs_embed, action, reward, next_state)
    loss = logits.sum() + S2.sum()
    loss.backward()
    for name, p in agent.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_multi_step_episode_bptt_is_finite():
    """Real check that backprop through a full multi-step episode
    (the actual BC training pattern) stays numerically stable."""
    agent, state, config = _agent(school_level="S2_multi_step")
    S = agent.init_memory(1)
    total_loss = torch.zeros(())
    for _ in range(6):
        logits, obs_embed = agent.act(state, S)
        action = logits.argmax(-1)
        next_state, reward, done = env_step(state, action, config)
        S = agent.update_memory(S, obs_embed, action, reward, next_state)
        total_loss = total_loss + logits.sum()
        state = next_state
        if done.item():
            break
    total_loss.backward()
    for p in agent.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_agent_uses_d_over_2_value_write_and_default_ln_recurrence():
    """Real check that the adapter follows the plan's own Phase 2
    constraints -- KEEP: D/2 value/write, default LN recurrence, no
    new recurrence experiments."""
    agent, state, config = _agent()
    assert agent.ws.config.value_dim == agent.D // 2
    assert agent.ws.config.identity_biased is False
    assert agent.ws.config.bounded_residual is False
    assert agent.ws.config.bounded_accumulating is False
