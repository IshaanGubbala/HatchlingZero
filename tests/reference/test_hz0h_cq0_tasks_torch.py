"""Real correctness tests for reference/hz0h_cq0_tasks_torch.py's
reassignment task generator."""
from __future__ import annotations

import torch

from reference.hz0h_cq0_tasks_torch import (
    ReassignmentTaskConfig,
    demo_length,
    generate_reassignment_batch,
    query_length,
)


def test_shapes_are_fixed_and_correct():
    config = ReassignmentTaskConfig(num_keys=6, num_overwrites=3, num_distractor_keys=2)
    gen = torch.Generator().manual_seed(0)
    demo, query, targets = generate_reassignment_batch(8, config, gen)
    assert demo.shape == (8, demo_length(config)) == (8, 20)
    assert query.shape == (8, query_length()) == (8, 2)
    assert targets.shape == (8,)
    assert demo.dtype == torch.long and query.dtype == torch.long and targets.dtype == torch.long


def _parse_demo(demo_row: torch.Tensor) -> list[tuple[int, int]]:
    """Real parser: decode the byte sequence back into (key_id, value_id)
    events, independent of the generator's own internal logic -- so this
    test can verify the target actually matches what the demo literally
    says, not just trust the generator's own bookkeeping."""
    events = []
    bytes_list = demo_row.tolist()
    for i in range(0, len(bytes_list), 4):
        key_byte, eq, value_byte, semi = bytes_list[i:i + 4]
        assert eq == ord("=") and semi == ord(";"), "malformed event -- generator produced the wrong byte layout"
        key_id = key_byte - ord("A")
        value_id = value_byte - ord("0")
        events.append((key_id, value_id))
    return events


def test_target_matches_the_real_last_assignment_in_the_demo():
    config = ReassignmentTaskConfig(num_keys=8, num_overwrites=4, num_distractor_keys=3)
    gen = torch.Generator().manual_seed(1)
    demo, query, targets = generate_reassignment_batch(16, config, gen)

    for b in range(16):
        events = _parse_demo(demo[b])
        target_key = query[b, 0].item() - ord("A")
        assert query[b, 1].item() == ord("?")

        target_events = [v for k, v in events if k == target_key]
        assert len(target_events) == config.num_overwrites, "wrong number of target-key events in the demo"
        real_last_value = target_events[-1]
        assert targets[b].item() - ord("0") == real_last_value, "target label doesn't match the real last assignment in the demo"


def test_target_key_events_not_always_in_the_same_position():
    """Real check against a positional shortcut: across many generated
    examples, the target key's LAST event should not always land at a
    fixed byte offset -- otherwise a model could learn "always read byte
    N" instead of actually tracking reassignment."""
    config = ReassignmentTaskConfig(num_keys=6, num_overwrites=2, num_distractor_keys=2)
    gen = torch.Generator().manual_seed(2)
    demo, query, _targets = generate_reassignment_batch(64, config, gen)

    last_event_positions = set()
    for b in range(64):
        target_key = query[b, 0].item() - ord("A")
        events = _parse_demo(demo[b])
        last_index = max(i for i, (k, _v) in enumerate(events) if k == target_key)
        last_event_positions.add(last_index)

    assert len(last_event_positions) > 1, "target key's last event always landed at the same position -- real positional-shortcut risk"


def test_no_overlap_between_target_and_distractor_keys():
    config = ReassignmentTaskConfig(num_keys=10, num_overwrites=2, num_distractor_keys=4)
    gen = torch.Generator().manual_seed(3)
    demo, query, _targets = generate_reassignment_batch(32, config, gen)

    for b in range(32):
        target_key = query[b, 0].item() - ord("A")
        events = _parse_demo(demo[b])
        distractor_keys = {k for k, _v in events if k != target_key}
        assert target_key not in distractor_keys
        assert len(distractor_keys) == config.num_distractor_keys


def test_rejects_invalid_configs():
    gen = torch.Generator().manual_seed(4)
    try:
        generate_reassignment_batch(2, ReassignmentTaskConfig(num_keys=2, num_overwrites=1, num_distractor_keys=5), gen)
        assert False, "expected a ValueError for too few keys"
    except ValueError:
        pass
    try:
        generate_reassignment_batch(2, ReassignmentTaskConfig(num_keys=6, num_overwrites=0, num_distractor_keys=1), gen)
        assert False, "expected a ValueError for num_overwrites=0"
    except ValueError:
        pass
