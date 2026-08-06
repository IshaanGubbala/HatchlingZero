import numpy as np
import pytest

from reference.hz0e_e9_dispatch import build_dispatch_plan, scatter_expert_outputs


def test_dispatch_is_stable_bounded_and_preserves_token_order():
    plan = build_dispatch_plan(np.array([1, 0, 1, 1, 0, 2, 1]), 3, 1.0)
    assert plan.capacity == 3
    np.testing.assert_array_equal(plan.rank, [0, 0, 1, 2, 1, 0, 3])
    np.testing.assert_array_equal(plan.overflow, [False, False, False, False, False, False, True])
    np.testing.assert_array_equal(plan.dispatch_slot, [3, 0, 4, 5, 1, 6, -1])
    np.testing.assert_array_equal(plan.grouped_tokens, [[1, 4, -1], [0, 2, 3], [5, -1, -1]])


def test_scatter_replaces_accepted_tokens_and_keeps_fallback_for_overflow():
    plan = build_dispatch_plan(np.array([0, 0, 0]), 1, 2 / 3)
    expert = np.array([[[10.0], [20.0]]])
    fallback = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(scatter_expert_outputs(plan, expert, fallback), [[10.0], [20.0], [3.0]])


def test_capacity_pressure_is_independent_for_each_expert():
    plan = build_dispatch_plan(np.array([0, 0, 0, 1, 1, 1]), 2, 0.5)
    np.testing.assert_array_equal(plan.dispatch_slot, [0, 1, -1, 2, 3, -1])
    expert = np.array([[[10.0], [11.0]], [[20.0], [21.0]]])
    fallback = np.arange(1.0, 7.0).reshape(-1, 1)
    np.testing.assert_allclose(
        scatter_expert_outputs(plan, expert, fallback),
        [[10.0], [11.0], [3.0], [20.0], [21.0], [6.0]],
    )


@pytest.mark.parametrize("bad", [np.array([-1]), np.array([2])])
def test_out_of_range_experts_are_rejected(bad):
    with pytest.raises(ValueError):
        build_dispatch_plan(bad, 2, 1.0)
