"""
Tests for oracle memory diagnostics (Phase 7, Step 5).
"""

import pytest
import numpy as np
from src.hz0.metal_gdn2.scratchpad.oracle_memory import (
    test_associative_recall,
    test_overwrite,
    test_protected_unrelated_memory,
    test_recall_vs_distance,
    test_multi_key_interference,
    test_state_reset_isolation,
    run_all_diagnostics,
)


class TestOracleMemory:
    """Oracle routing memory tests."""

    def test_associative_recall_basic(self):
        """Recall accuracy on simple pairs."""
        results = test_associative_recall(num_pairs=4, slot_dim=8)

        assert "mean_reconstruction_error" in results
        assert "perfect_match_rate" in results
        # Oracle should have perfect or near-perfect accuracy
        assert results["perfect_match_rate"] > 0.5

    def test_associative_recall_scaling(self):
        """Accuracy across slot dimensions."""
        for dim in [4, 8, 16, 32]:
            results = test_associative_recall(num_pairs=4, slot_dim=dim)
            assert results["mean_reconstruction_error"] < 1e-4

    def test_overwrite_success(self):
        """Overwrite replaces old value."""
        results = test_overwrite(slot_dim=8)

        assert "overwrite_success" in results
        assert results["overwrite_success"] == 1.0
        # Green should be closer than red
        assert results["green_reconstruction_error"] < results["red_reconstruction_error"]

    def test_protected_unrelated_memory(self):
        """Overwriting A doesn't affect B."""
        results = test_protected_unrelated_memory(slot_dim=8)

        assert "protected_success" in results
        assert results["protected_success"] == 1.0
        # Blue should be preserved
        assert results["blue_reconstruction_error"] < results["green_interference_error"]

    def test_recall_distance_curve(self):
        """Recall as function of distance."""
        results = test_recall_vs_distance(num_keys=8, slot_dim=8)

        assert isinstance(results, dict)
        # Oracle: distance shouldn't matter (all exact)
        for dist, recall in results.items():
            assert recall > 0.5, f"Recall at distance {dist} too low"

    def test_multi_key_interference(self):
        """Distractors don't interfere with target."""
        results = test_multi_key_interference(num_distractors=4, slot_dim=8)

        assert results["interference_recall_accuracy"] == 1.0
        assert results["target_reconstruction_error"] < 1e-4

    def test_state_reset_isolation(self):
        """Reset clears state."""
        results = test_state_reset_isolation(slot_dim=8)

        assert results["state_isolation_success"] == 1.0
        assert results["zero_reconstruction_error"] < results["red_contamination_error"]

    def test_full_diagnostic_suite(self):
        """All diagnostics pass."""
        results = run_all_diagnostics(slot_dim=8)

        # Check structure
        assert "associative_recall" in results
        assert "overwrite" in results
        assert "protected_memory" in results
        assert "recall_vs_distance" in results
        assert "interference" in results
        assert "state_reset" in results

        # Check key metrics exist
        assert results["associative_recall"]["perfect_match_rate"] > 0
        assert results["overwrite"]["overwrite_success"] == 1.0
        assert results["protected_memory"]["protected_success"] == 1.0
        assert results["state_reset"]["state_isolation_success"] == 1.0


class TestMemoryDiagnosticsReport:
    """Format diagnostics for reporting."""

    def test_diagnostic_report_generation(self):
        """Generate formatted diagnostic report."""
        results = run_all_diagnostics(slot_dim=8)

        report = {
            "test_associative_recall": results["associative_recall"]["perfect_match_rate"],
            "test_overwrite": results["overwrite"]["overwrite_success"],
            "test_protected_memory": results["protected_memory"]["protected_success"],
            "test_interference": results["interference"]["interference_recall_accuracy"],
            "test_state_reset": results["state_reset"]["state_isolation_success"],
        }

        # All should pass
        for test_name, score in report.items():
            assert score > 0, f"{test_name} failed"

    def test_compare_multiple_dimensions(self):
        """Compare memory performance across dimensions."""
        dims = [4, 8, 16]
        scores = {}

        for dim in dims:
            results = run_all_diagnostics(slot_dim=dim)
            scores[dim] = results["associative_recall"]["perfect_match_rate"]

        # Should maintain accuracy across dims
        for dim, score in scores.items():
            assert score > 0.5, f"Score at dim {dim} too low"
