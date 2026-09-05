#!/usr/bin/env python3
"""Real, continuously-running Hatchling World rollout that feeds
scripts/hz_world_live_view.py's shared state file after every step --
run this alongside the viewer to watch an agent act in the sandbox
live in a browser.

Today this drives the ORACLE (the BFS planner from hatchling_world.oracle)
-- a real, always-succeeding agent, useful to prove the whole
env-to-browser pipeline works end to end before the actual HZ policy
(Phase 2 of plans/Hatchling world.md) exists. Once the HZ agent lands,
it plugs into this exact same state-file schema (see `snapshot()`
below) and the viewer needs zero changes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchling_world.curriculum import SCHOOL_LEVELS, generate_school_worlds
from hatchling_world.oracle import solve
from hatchling_world.state import WorldConfig, WorldState
from hatchling_world.transition import step as env_step


def action_to_str(a: int, config: WorldConfig) -> str:
    if a < config.n_rooms:
        return f"MOVE({a})"
    if a < config.n_rooms + config.n_colors * config.n_rooms:
        color, target = config.decode_use_key(a)
        return f"USE_KEY(color={color}, door->{target})"
    if a == config.action_pickup:
        return "PICKUP"
    return "INSPECT"


def snapshot(state: WorldState, config: WorldConfig, episode: int, step_idx: int,
             last_action: int | None, last_reward: float, episode_return: float,
             recent_successes: list[bool], agent_type: str) -> dict:
    R = config.n_rooms
    doors = []
    for a in range(R):
        for b in range(a + 1, R):
            if state.door_adj[0, a, b]:
                doors.append({
                    "a": a, "b": b,
                    "locked": bool(state.door_locked[0, a, b]),
                    "color": int(state.door_key_color[0, a, b].item()),
                })
    return {
        "episode": episode, "step": step_idx,
        "n_rooms": R, "n_colors": config.n_colors,
        "agent_room": int(state.agent_room[0].item()),
        "goal_room": int(state.goal_room[0].item()),
        "doors": doors,
        "room_keys": state.room_keys[0].tolist(),
        "inventory": state.inventory[0].tolist(),
        "last_action_str": action_to_str(last_action, config) if last_action is not None else None,
        "last_reward": last_reward,
        "episode_return": episode_return,
        "recent_success_rate": (sum(recent_successes) / len(recent_successes)) if recent_successes else 0.0,
        "agent_type": agent_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/hz_world_live_state.json"))
    parser.add_argument("--school-level", type=str, default="S2_multi_step", choices=list(SCHOOL_LEVELS))
    parser.add_argument("--step-delay", type=float, default=0.4, help="seconds between steps, for watchability")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    recent_successes: list[bool] = []
    print(f"[hz_world_rollout] school_level={args.school_level}, writing live state to {args.state_file}", flush=True)

    for ep in range(args.episodes):
        state, config = generate_school_worlds(args.school_level, batch=1, episode_seed=args.seed + ep)
        plan = solve(state, config, index=0)
        episode_return = 0.0

        # write an initial "episode start" snapshot before any action
        args.state_file.write_text(json.dumps(
            snapshot(state, config, ep, 0, None, 0.0, episode_return, recent_successes, "oracle")))
        time.sleep(args.step_delay)

        if plan is None:
            print(f"[hz_world_rollout] episode {ep}: UNSOLVABLE (real bug if this ever prints)", flush=True)
            continue

        success = False
        for step_idx, a in enumerate(plan, start=1):
            state, reward, done = env_step(state, torch.tensor([a]), config)
            episode_return += reward.item()
            args.state_file.write_text(json.dumps(
                snapshot(state, config, ep, step_idx, a, reward.item(), episode_return, recent_successes, "oracle")))
            time.sleep(args.step_delay)
            if done.item():
                success = state.agent_room.item() == state.goal_room.item()
                break

        recent_successes.append(success)
        recent_successes[:] = recent_successes[-50:]
        print(f"[hz_world_rollout] episode {ep}: success={success} return={episode_return:.3f} "
              f"steps={len(plan)}", flush=True)


if __name__ == "__main__":
    main()
