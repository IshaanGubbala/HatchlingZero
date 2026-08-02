"""Matched synthetic before/after benchmark for the exact GDN-2 fix.

This isolates recurrence semantics from model training. It compares the frozen
HZ additive recurrence, scalar-beta KDA-like correction, exact vector-gated
GDN-2, and a full-history matching control on overwrite/interference tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_gdn2_fix_reference import gdn2_fix_step


def old_step(state, query, key, value, alpha, erase, write):
    state = alpha[None, :] * (1.0 - erase)[None, :] * state + write[:, None] * value[:, None] * key[None, :]
    return state @ query, state


def kda_step(state, query, key, value, alpha, beta):
    decayed = alpha[None, :] * state
    old = decayed @ key
    state = decayed + beta * (value - old)[:, None] * key[None, :]
    return state @ query, state


def attention_lookup(history, query):
    keys, values = history
    scores = np.asarray(keys) @ query
    return values[int(np.argmax(scores))]


def run_case(kind, keys, values, updates, *, dim, decay=0.99):
    state = np.zeros((dim, dim), dtype=np.float32)
    history_keys, history_values = [], []
    elapsed = 0.0
    final_output = None
    for key_index, value_index in updates:
        key = keys[key_index]
        value = values[value_index]
        query = key
        start = time.perf_counter()
        if kind == "old":
            final_output, state = old_step(state, query, key, value, np.full(dim, decay), np.full(dim, 0.01), np.ones(dim))
        elif kind == "kda":
            final_output, state = kda_step(state, query, key, value, np.full(dim, decay), 0.5)
        elif kind == "gdn2_fix":
            final_output, state = gdn2_fix_step(state[None, None], query[None, None], key[None, None], value[None, None], np.full((1, 1, dim), decay), np.ones((1, 1, dim)), np.ones((1, 1, dim)), normalize_key=False)
            final_output, state = final_output[0, 0], state[0, 0]
        elif kind == "attention":
            history_keys.append(key)
            history_values.append(value)
            final_output = attention_lookup((history_keys, history_values), query)
        else:
            raise ValueError(kind)
        elapsed += time.perf_counter() - start
    if kind == "attention":
        state = np.zeros((dim, dim), dtype=np.float32)
        for key, value in zip(history_keys, history_values):
            state[:, int(np.argmax(key))] = value
    return {"state": state, "output": final_output, "elapsed_seconds": elapsed}


def benchmark(seed=7, trials=128, dim=16):
    rng = np.random.default_rng(seed)
    kinds = ("old", "kda", "gdn2_fix", "attention")
    results = {
        kind: {
            "target_squared_errors": [],
            "untouched_squared_errors": [],
            "state_norms": [],
            "elapsed_seconds": 0.0,
        }
        for kind in kinds
    }
    for _ in range(trials):
        keys = np.eye(dim, dtype=np.float32)
        values = np.eye(dim, dtype=np.float32)
        first_key, second_key = rng.choice(dim, 2, replace=False)
        first_value, second_value = rng.choice(dim, 2, replace=False)
        cases = {
            "overwrite": [(first_key, first_value), (second_key, second_value), (first_key, second_value)],
            "interference": [(i, i) for i in range(dim)] + [(first_key, second_value)],
            "contradiction": [(first_key, first_value), (first_key, second_value)],
        }
        for updates in cases.values():
            expected_state = np.zeros((dim, dim), dtype=np.float32)
            touched = set()
            for key_index, value_index in updates:
                expected_state[:, key_index] = values[value_index]
                touched.add(key_index)
            untouched = np.asarray([i for i in range(dim) if i not in touched])
            for kind in kinds:
                result = run_case(kind, keys, values, updates, dim=dim)
                actual_state = result["state"]
                target_error = np.mean((actual_state - expected_state) ** 2)
                untouched_error = (
                    np.mean(actual_state[:, untouched] ** 2) if untouched.size else 0.0
                )
                results[kind]["target_squared_errors"].append(float(target_error))
                results[kind]["untouched_squared_errors"].append(float(untouched_error))
                results[kind]["state_norms"].append(float(np.linalg.norm(actual_state)))
                results[kind]["elapsed_seconds"] += result["elapsed_seconds"]
    total = trials * len(cases)
    for kind, result in results.items():
        result["total_cases"] = total
        result["mean_target_mse"] = float(np.mean(result.pop("target_squared_errors")))
        result["mean_untouched_mse"] = float(np.mean(result.pop("untouched_squared_errors")))
        result["mean_state_norm"] = float(np.mean(result.pop("state_norms")))
        result["cases_per_second"] = total / max(result["elapsed_seconds"], 1e-12)
    return {"seed": seed, "trials": trials, "dimension": dim, "results": results, "finite": all(np.isfinite(v["mean_state_norm"]) for v in results.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.seed, args.trials, args.dim)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
