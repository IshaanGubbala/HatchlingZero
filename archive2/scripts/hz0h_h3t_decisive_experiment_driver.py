"""HZ-0H H3-T: driver for the decisive 10-30M-scale experiment. Shells
out to scripts/hz0h_h3t_decisive_experiment_worker.py once per
(condition, seed) combination -- genuine process isolation, so each
run's peak RSS is a fair, independent measurement (fixing the earlier
disclosed same-process limitation for real). Aggregates results into one
summary JSON + printed table.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path("outputs/hz0h_h3t_decisive_experiment")
CONDITIONS = ["true_bptt", "sg_global", "sg_global_calibrated"]
SEEDS = [0, 1, 2]
STEPS = 150
WARMUP_STEPS = 30
SYNTHETIC_FRACTION = 0.5
EVAL_EVERY = 15


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total_runs = len(CONDITIONS) * len(SEEDS)
    run_i = 0
    overall_start = time.perf_counter()

    for condition in CONDITIONS:
        for seed in SEEDS:
            run_i += 1
            out_path = OUT_DIR / f"{condition}_seed{seed}.json"
            print(f"=== [{run_i}/{total_runs}] {condition} seed={seed} starting ===", flush=True)
            t0 = time.perf_counter()
            cmd = [
                sys.executable, "scripts/hz0h_h3t_decisive_experiment_worker.py",
                "--condition", condition, "--seed", str(seed),
                "--steps", str(STEPS), "--warmup-steps", str(WARMUP_STEPS),
                "--synthetic-fraction", str(SYNTHETIC_FRACTION), "--eval-every", str(EVAL_EVERY),
                "--out", str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.perf_counter() - t0
            if proc.returncode != 0:
                print(f"=== [{run_i}/{total_runs}] {condition} seed={seed} FAILED (exit {proc.returncode}) in {dt:.1f}s ===")
                print(proc.stdout[-2000:])
                print(proc.stderr[-2000:])
                continue
            print(proc.stdout.strip())
            print(f"=== [{run_i}/{total_runs}] {condition} seed={seed} done in {dt:.1f}s ===", flush=True)
            with open(out_path) as f:
                results.append(json.load(f))

    total_dt = time.perf_counter() - overall_start
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_wall_clock_s": total_dt}, f, indent=2)

    print(f"\n=== ALL RUNS DONE in {total_dt:.1f}s ({total_dt/60:.1f} min) ===")
    print(f"{'condition':>22s} {'seed':>5s} {'final_train_loss':>17s} {'final_held_out_ce':>18s} {'final_grad_cosine':>18s} {'mean_prod_ms':>13s} {'peak_rss_mb':>12s}")
    for r in results:
        cos = r["final_grad_cosine"]
        cos_str = f"{cos:.4f}" if cos is not None else "n/a"
        prod = r["mean_production_step_ms"]
        prod_str = f"{prod:.1f}" if prod is not None else "n/a"
        print(f"{r['condition']:>22s} {r['seed']:>5d} {r['final_train_loss']:>17.4f} {r['final_held_out_ce']:>18.4f} {cos_str:>18s} {prod_str:>13s} {r['peak_rss_mb']:>12.1f}")


if __name__ == "__main__":
    main()
