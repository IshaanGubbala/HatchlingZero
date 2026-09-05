#!/usr/bin/env python3
"""Real behavior-cloning training for the HZ-World agent, plans/Hatchling
world.md Phase 3 (W2) -- delivered right after Phase 2's adapter since
an untrained adapter has nothing worth watching. Trains via teacher-
forced imitation of the oracle's real plans: at each step the agent
reasons over the CURRENT observation via S+H, is scored against the
oracle's real action (cross-entropy), and the environment is then
stepped with the ORACLE's action (teacher forcing) so persistent
memory S gets updated on the real, correct trajectory -- standard BC
practice, and it means credit assignment backprops through the whole
episode's S recurrence AND every step's H rounds (section 1.2.B's own
"are gradients reaching policy, memory, readout, and recurrent state
paths?" check, verified directly here, not just assumed).

Periodically runs a REAL self-driven evaluation episode -- the
model's OWN actions, no oracle forcing -- and streams it live to
scripts/hz_world_live_view.py's shared state file with agent_type='hz'.
Open http://localhost:8765 while this runs to watch actual competence
emerge (or not) over training, not just a loss curve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_world_agent_torch import HZWorldAgent
from hatchling_world.actions import decode_action
from hatchling_world.curriculum import SCHOOL_LEVELS, generate_school_worlds
from hatchling_world.oracle import solve
from hatchling_world.state import WorldConfig, WorldState
from hatchling_world.transition import step as env_step

HISTORY_LEN = 60


def snapshot(state: WorldState, config: WorldConfig, episode: int, step_idx: int, plan_len: int,
             last_action: dict | None, last_reward: float, episode_return: float,
             success_history: list[int], return_history: list[float], agent_type: str,
             school_level: str, prev_room: int) -> dict:
    R = config.n_rooms
    doors = []
    for a in range(R):
        for b in range(a + 1, R):
            if state.door_adj[0, a, b]:
                doors.append({"a": a, "b": b, "locked": bool(state.door_locked[0, a, b]),
                              "color": int(state.door_key_color[0, a, b].item())})
    return {
        "kind": "room",
        "episode": episode, "step": step_idx, "plan_len": plan_len, "school_level": school_level,
        "n_rooms": R, "n_colors": config.n_colors,
        "agent_room": int(state.agent_room[0].item()), "prev_room": prev_room,
        "goal_room": int(state.goal_room[0].item()), "doors": doors,
        "room_keys": state.room_keys[0].tolist(), "inventory": state.inventory[0].tolist(),
        "last_action": last_action, "last_reward": last_reward, "episode_return": episode_return,
        "success_history": success_history[-HISTORY_LEN:], "return_history": return_history[-HISTORY_LEN:],
        "recent_success_rate": (sum(success_history) / len(success_history)) if success_history else 0.0,
        "agent_type": agent_type,
    }


def bc_train_step(agent, opt, school_level: str, episode_seed: int):
    state, config = generate_school_worlds(school_level, batch=1, episode_seed=episode_seed, split="train")
    plan = solve(state, config, index=0)
    if not plan:
        return None
    S = agent.init_memory(1)
    opt.zero_grad(set_to_none=True)
    total_loss = torch.zeros(())
    correct = 0
    for a in plan:
        logits, obs_embed = agent.act(state, S)
        target = torch.tensor([a])
        total_loss = total_loss + F.cross_entropy(logits, target)
        correct += int((logits.argmax(-1) == target).item())
        next_state, reward, done = env_step(state, target, config)
        S = agent.update_memory(S, obs_embed, target, reward, next_state)
        state = next_state
    total_loss = total_loss / len(plan)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()
    return total_loss.item(), correct / len(plan)


def run_live_eval_episode(agent, school_level: str, episode_seed: int, state_file: Path, step_delay: float,
                           ep_idx: int, success_history: list[int], return_history: list[float]):
    """Real self-driven episode -- the model's OWN argmax actions, no
    oracle forcing. split='test' -- these worlds are never trained on."""
    state, config = generate_school_worlds(school_level, batch=1, episode_seed=episode_seed, split="test")
    plan_len = len(solve(state, config, index=0) or [])
    S = agent.init_memory(1)
    episode_return = 0.0
    prev_room = int(state.agent_room[0].item())

    state_file.write_text(json.dumps(snapshot(state, config, ep_idx, 0, plan_len, None, 0.0, episode_return,
                                               success_history, return_history, "hz", school_level, prev_room)))
    time.sleep(step_delay)

    success = False
    with torch.no_grad():
        for step_idx in range(1, config.max_steps + 1):
            logits, obs_embed = agent.act(state, S)
            action = logits.argmax(-1)
            action_info = decode_action(int(action.item()), config)
            prev_room = int(state.agent_room[0].item())
            next_state, reward, done = env_step(state, action, config)
            episode_return += reward.item()
            S = agent.update_memory(S, obs_embed, action, reward, next_state)
            state = next_state
            state_file.write_text(json.dumps(snapshot(state, config, ep_idx, step_idx, plan_len, action_info,
                                                        reward.item(), episode_return, success_history,
                                                        return_history, "hz", school_level, prev_room)))
            time.sleep(step_delay)
            if done.item():
                success = state.agent_room.item() == state.goal_room.item()
                break

    success_history.append(int(success))
    return_history.append(episode_return)
    success_history[:] = success_history[-HISTORY_LEN:]
    return_history[:] = return_history[-HISTORY_LEN:]
    return success, episode_return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--school-level", type=str, default="S1_short_composition", choices=list(SCHOOL_LEVELS))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-step-delay", type=float, default=0.35, help="seconds/step during LIVE eval episodes")
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/hz_world_live_state.json"))
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _, world_config = generate_school_worlds(args.school_level, batch=1, episode_seed=0)
    agent = HZWorldAgent(world_config, d_model=args.d_model, memory_slots=args.memory_slots,
                          workspace_slots=args.workspace_slots, n_rounds=args.n_rounds)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[hz_world_bc] school_level={args.school_level} n_params={n_params} n_actions={world_config.n_actions}",
          flush=True)
    opt = torch.optim.AdamW(agent.parameters(), lr=args.lr)

    success_history: list[int] = []
    return_history: list[float] = []
    recent_acc: list[float] = []
    eval_ep_idx = 0

    for ep in range(args.train_episodes):
        result = bc_train_step(agent, opt, args.school_level, episode_seed=args.seed + 1 + ep)
        if result is None:
            continue
        loss, acc = result
        recent_acc.append(acc)
        recent_acc[:] = recent_acc[-200:]

        if (ep + 1) % 50 == 0:
            print(f"[hz_world_bc] train_episode={ep+1}/{args.train_episodes} loss={loss:.4f} "
                  f"recent_action_acc={sum(recent_acc)/len(recent_acc):.3f}", flush=True)

        if (ep + 1) % args.eval_every == 0:
            success, ret = run_live_eval_episode(agent, args.school_level, episode_seed=eval_ep_idx,
                                                  state_file=args.state_file, step_delay=args.eval_step_delay,
                                                  ep_idx=eval_ep_idx, success_history=success_history,
                                                  return_history=return_history)
            eval_ep_idx += 1
            recent_rate = sum(success_history[-20:]) / len(success_history[-20:])
            print(f"[hz_world_bc] LIVE EVAL episode {eval_ep_idx}: success={success} return={ret:.3f} "
                  f"recent_success_rate(20)={recent_rate:.3f}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), args.save_checkpoint)
        print(f"[hz_world_bc] saved checkpoint to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
