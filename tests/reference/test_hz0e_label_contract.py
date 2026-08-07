"""Regression guard for the four-expert supervised-router label contract."""
from __future__ import annotations

import mlx.core as mx
import pytest

from reference.hz0e_e3_routing_objectives import supervised_warm_start
from reference.hz0e_moe_contract import MoeConfig


def test_supervised_router_rejects_labels_outside_expert_range_before_model_use():
    with pytest.raises(ValueError, match="target_expert labels"):
        supervised_warm_start(
            None,
            {"tools": mx.zeros((1, 2), dtype=mx.int32)},
            {"tools": 4},
            MoeConfig(dim=8),
        )
