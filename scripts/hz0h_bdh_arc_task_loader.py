#!/usr/bin/env python3
"""Real ARC-AGI-1 task loader + byte-level episode serializer for the
HZ-CQ groundwork (plans/newnewplan.md section 33). Turns a task's
demonstration pairs + query into a single byte string matching this
project's existing byte-level LM convention (vocab_size=256), in the
episode shape the plan calls for: demo1 in/out, demo2 in/out, ...,
query in -> query out, no explicit task ID.

Dataset: data/arc_agi_1/data/{training,evaluation}/*.json (cloned from
https://github.com/fchollet/ARC-AGI, 400 tasks per split, real ARC-AGI-1
public benchmark). Each task JSON has "train" (demonstration pairs) and
"test" (query pairs, held out at real eval time -- output present here
since this is the public training-split data with answers included).

Grid cells are single digits 0-9 (10 ARC colors) -- serialized as their
literal ASCII digit, one byte per cell, so a real byte-level model can
read/write them directly without a separate tokenizer.
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

ARC_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "arc_agi_1" / "data"

Grid = list[list[int]]


def load_arc_tasks(split: str = "training") -> dict[str, dict]:
    """split: 'training' or 'evaluation'. Returns {task_id: task_dict}."""
    root = ARC_DATA_ROOT / split
    tasks = {}
    for path in sorted(glob.glob(str(root / "*.json"))):
        task_id = Path(path).stem
        tasks[task_id] = json.loads(Path(path).read_text())
    return tasks


def _serialize_grid(grid: Grid) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def _parse_grid(text: str) -> Grid:
    return [[int(ch) for ch in line] for line in text.split("\n") if line]


def serialize_episode(task: dict, rng: random.Random, held_out_query: bool = False) -> tuple[str, Grid]:
    """Builds the full episode text for one ARC task.

    Demonstration order is NOT shuffled -- unlike the register-machine
    task's shortcut-resistant shuffling, ARC demonstrations don't have
    a "real order" to preserve or obscure; they're an unordered set of
    examples of the same transformation, so using the JSON's own order
    is fine and matches how a human would read them.

    held_out_query=True omits the query's true output from the text
    (real eval mode); False appends it (training mode). Always returns
    the true query output grid separately so the caller has ground
    truth either way.
    """
    demos = task["train"]
    query = rng.choice(task["test"]) if len(task["test"]) > 1 else task["test"][0]

    parts = []
    for demo in demos:
        parts.append(f"IN\n{_serialize_grid(demo['input'])}\nOUT\n{_serialize_grid(demo['output'])}\nEND")
    parts.append(f"QUERY\n{_serialize_grid(query['input'])}")
    if not held_out_query:
        parts.append(f"ANSWER\n{_serialize_grid(query['output'])}\nEND")

    return "\n".join(parts), query["output"]


def build_episode_parts(task: dict, rng: random.Random) -> tuple[str, str, str, Grid]:
    """Same content as serialize_episode but split into the three
    phases the HZ-CQ forward pass needs staged separately: demonstration
    text (-> persistent task memory), query text (-> conditions the
    reasoning workspace init), and answer text (-> teacher-forced
    target during training, never fed to the model at real eval time).
    Always returns the true answer text/grid; the caller decides
    whether to embed it (training) or hold it out (eval)."""
    demos = task["train"]
    query = rng.choice(task["test"]) if len(task["test"]) > 1 else task["test"][0]

    demo_parts = [f"IN\n{_serialize_grid(d['input'])}\nOUT\n{_serialize_grid(d['output'])}\nEND" for d in demos]
    memory_text = "\n".join(demo_parts)
    query_text = f"QUERY\n{_serialize_grid(query['input'])}"
    answer_text = f"ANSWER\n{_serialize_grid(query['output'])}\nEND"
    return memory_text, query_text, answer_text, query["output"]


def parse_answer(generated_text: str) -> Grid | None:
    """Extracts the grid following the last 'ANSWER\\n' in generated text.
    Returns None if the model didn't produce a well-formed answer block
    (missing marker, ragged rows, non-digit characters) -- real failure
    modes to expect from a model that hasn't learned the format yet. A
    trailing 'END' marker (if present, real generation should produce
    one -- see hz0h_bdh_arc_eval.py's stopping criterion) is accepted
    but not required, so this still parses teacher-forced text from
    before the END marker existed."""
    marker = "ANSWER\n"
    idx = generated_text.rfind(marker)
    if idx == -1:
        return None
    body = generated_text[idx + len(marker):]
    lines = body.split("\n")
    grid_lines = []
    for line in lines:
        if not line or line == "END" or not all(ch.isdigit() for ch in line):
            break
        grid_lines.append(line)
    if not grid_lines:
        return None
    width = len(grid_lines[0])
    if any(len(row) != width for row in grid_lines):
        return None
    return _parse_grid("\n".join(grid_lines))


def grids_equal(a: Grid, b: Grid) -> bool:
    return a == b


if __name__ == "__main__":
    # Real round-trip verification against the actual downloaded dataset.
    tasks = load_arc_tasks("training")
    assert len(tasks) == 400, f"expected 400 training tasks, got {len(tasks)}"
    eval_tasks = load_arc_tasks("evaluation")
    assert len(eval_tasks) == 400, f"expected 400 evaluation tasks, got {len(eval_tasks)}"

    rng = random.Random(7)
    checked = 0
    max_episode_bytes = 0
    for task_id, task in tasks.items():
        text, true_output = serialize_episode(task, rng, held_out_query=False)
        assert all(ord(c) < 256 for c in text), f"{task_id}: non-byte char in episode"
        parsed = parse_answer(text)
        assert parsed is not None, f"{task_id}: parse_answer failed to recover the answer we just wrote"
        assert grids_equal(parsed, true_output), f"{task_id}: round-trip mismatch"
        max_episode_bytes = max(max_episode_bytes, len(text.encode("utf-8")))
        checked += 1

    print(f"[arc_task_loader] round-trip verified on {checked}/400 training tasks")
    print(f"[arc_task_loader] max episode length: {max_episode_bytes} bytes")

    # build_episode_parts: verify the three pieces concatenate back to
    # exactly serialize_episode's output (same content, just staged).
    for task_id, task in list(tasks.items())[:50]:
        rng2 = random.Random(7)
        whole_text, _ = serialize_episode(task, rng2, held_out_query=False)
        rng3 = random.Random(7)
        mem_text, query_text, answer_text, true_out = build_episode_parts(task, rng3)
        rejoined = "\n".join([mem_text, query_text, answer_text])
        assert rejoined == whole_text, f"{task_id}: build_episode_parts doesn't match serialize_episode"
        assert true_out is not None
    print("[arc_task_loader] build_episode_parts verified against serialize_episode on 50 tasks")

    # held-out mode sanity: no ANSWER marker should appear, parse_answer -> None
    text_held, true_out = serialize_episode(tasks[next(iter(tasks))], rng, held_out_query=True)
    assert "ANSWER" not in text_held
    assert parse_answer(text_held) is None
    print("[arc_task_loader] held-out query mode verified (no answer leakage)")
