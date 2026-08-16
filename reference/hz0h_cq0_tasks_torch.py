"""Real synthetic task generators for CQ-0's own decisive gate
(`plans/Deep Reserach Plan.md`'s "Synthetic task ladder" section):
`d(accuracy)/d(R) > 0`, growing with task dependency depth, evaluated
on the SAME trained `HZCQ0` checkpoint at different `R`.

## Reassignment

Real, well-motivated first choice: this project already has strong
prior evidence (H5 experiment, an earlier investigation of exact BDH's
own streaming state) that BDH's undecayed running-sum state tracks
"latest write wins" cleanly (0.984-1.00 real-state accuracy across 3
seeds on an analogous task). This task tests whether `HZCQ0`'s own
`S`/`H` split preserves that property once wrapped in the multi-slot
reasoning loop, and whether reasoning depth `R` helps more as the real
dependency depth (`num_overwrites`) increases.

Byte-level encoding (matches this project's own byte-vocab convention,
`vocab_size=256`, no separate tokenizer needed):

  key   = single uppercase ASCII letter, `ord('A') + i`, `i` in `[0, num_keys)`
  value = single ASCII digit, `ord('0') + v`, `v` in `[0, 10)`
  `=`   = byte 61 (assignment)
  `;`   = byte 59 (event separator)
  `?`   = byte 63 (query marker)

One demonstration is a sequence of fixed-length `"X=Y;"` (4 bytes)
assignment events: `num_overwrites` events for the TARGET key (the one
that will be queried, real values assigned in order, the correct
answer is whichever value was assigned LAST) interleaved with
`num_distractor_keys` single-binding events for other keys (real noise,
a second, independent difficulty axis from `num_overwrites`). Total
demo length is always `4 * (num_overwrites + num_distractor_keys)` --
fixed given fixed generator args, so a whole batch shares one length
with no padding needed. The query is always `"X?"` (2 bytes, the
target key). The target is the single correct current value byte.

Real, disclosed task-design choice: `num_overwrites` is the intended
real dependency-depth axis for the R-scaling gate (harder = more
overwrites to track correctly through); `num_distractor_keys` is a
separate, secondary difficulty axis (more irrelevant information to
filter, not more dependency depth) -- vary them independently when
building the CQ-0-gate's own `A(R, d)` plot (see the plan doc's own
"Synthetic task ladder" section for that plot's exact spec).
"""
from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class ReassignmentTaskConfig:
    num_keys: int = 6          # real alphabet size the target/distractor keys are drawn from
    num_overwrites: int = 3    # real dependency-depth axis: how many times the target key is (re)bound
    num_distractor_keys: int = 3  # secondary axis: how many OTHER keys get one binding each


def _key_byte(i: int) -> int:
    return ord("A") + i


def _value_byte(v: int) -> int:
    return ord("0") + v


def demo_length(config: ReassignmentTaskConfig) -> int:
    return 4 * (config.num_overwrites + config.num_distractor_keys)


def query_length() -> int:
    return 2  # "X?"


def generate_reassignment_batch(
    batch_size: int,
    config: ReassignmentTaskConfig,
    generator: torch.Generator,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(demo_tokens, query_tokens, targets)`:
    `demo_tokens`: `(B, demo_length(config))`, `query_tokens`: `(B, 2)`,
    `targets`: `(B,)` -- one correct-value byte per example. All real
    `long` dtype token IDs (0-255), ready to feed directly into
    `HZCQ0.forward` (`targets` needs `.unsqueeze(-1)` to match its own
    `(B, M)` convention when `M=1`, deliberately left to the caller so
    this generator stays decoder-shape-agnostic)."""
    if config.num_keys < 1 + config.num_distractor_keys:
        raise ValueError("num_keys must be large enough for 1 target key + all distractor keys, with no overlap")
    if config.num_overwrites < 1:
        raise ValueError("num_overwrites must be >= 1 (the target key needs at least one binding)")

    T_demo = demo_length(config)
    demo_tokens = torch.zeros((batch_size, T_demo), dtype=torch.long, device=device)
    query_tokens = torch.zeros((batch_size, query_length()), dtype=torch.long, device=device)
    targets = torch.zeros((batch_size,), dtype=torch.long, device=device)

    for b in range(batch_size):
        key_ids = torch.randperm(config.num_keys, generator=generator)[: 1 + config.num_distractor_keys].tolist()
        target_key = key_ids[0]
        distractor_keys = key_ids[1:]

        events = []  # list of (key_id, value_id)
        last_value = None
        for _ in range(config.num_overwrites):
            value_id = int(torch.randint(0, 10, (1,), generator=generator).item())
            events.append((target_key, value_id))
            last_value = value_id
        for dk in distractor_keys:
            value_id = int(torch.randint(0, 10, (1,), generator=generator).item())
            events.append((dk, value_id))

        # Real, important detail: shuffle events so the target key's LAST
        # assignment is not always in a fixed position -- otherwise the
        # model could learn a positional shortcut instead of actually
        # tracking "most recent write per key", which would defeat the
        # whole point of the task. Overwrites of the SAME key must stay
        # in their real relative order (can't shuffle a key's own writes
        # relative to each other), so shuffle a stable-sort-safe ordering:
        # assign each event a random priority, but keep target-key events
        # in original (chronological) order relative to each other.
        target_events = [e for e in events if e[0] == target_key]
        other_events = [e for e in events if e[0] != target_key]
        # Interleave by inserting target_events (in order) at random
        # positions among the shuffled other_events.
        perm = torch.randperm(len(other_events), generator=generator).tolist()
        other_events = [other_events[i] for i in perm]
        insert_positions = sorted(torch.randint(0, len(other_events) + 1, (len(target_events),), generator=generator).tolist())
        merged = list(other_events)
        for offset, (pos, ev) in enumerate(zip(insert_positions, target_events)):
            merged.insert(pos + offset, ev)

        byte_seq: list[int] = []
        for key_id, value_id in merged:
            byte_seq.extend([_key_byte(key_id), ord("="), _value_byte(value_id), ord(";")])
        assert len(byte_seq) == T_demo

        demo_tokens[b] = torch.tensor(byte_seq, dtype=torch.long, device=device)
        query_tokens[b] = torch.tensor([_key_byte(target_key), ord("?")], dtype=torch.long, device=device)
        targets[b] = _value_byte(last_value)

    return demo_tokens, query_tokens, targets
