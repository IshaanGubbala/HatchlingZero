import numpy as np

from restart.hz0a_pmetal.python.training import PmetalOptimizerPath


def test_pmetal_optimizer_accumulation_and_resume_are_exact(tmp_path):
    gradients = [np.array([1.0, -2.0, 0.5]) * (index + 1) for index in range(12)]
    full = PmetalOptimizerPath(np.array([0.2, -0.1, 0.4]), accumulation_steps=2, total_steps=6)
    resumed = PmetalOptimizerPath(np.array([0.2, -0.1, 0.4]), accumulation_steps=2, total_steps=6)
    for index, gradient in enumerate(gradients[:6]):
        full.add_microbatch(gradient, tokens=8)
        resumed.add_microbatch(gradient, tokens=8)
        if index == 3:
            resumed.checkpoint(tmp_path / "optimizer.json")
            resumed = PmetalOptimizerPath.restore(tmp_path / "optimizer.json")
    for gradient in gradients[6:]:
        full.add_microbatch(gradient, tokens=8)
        resumed.add_microbatch(gradient, tokens=8)
    assert full.state.tokens_seen == resumed.state.tokens_seen == 96
    assert full.fingerprint() == resumed.fingerprint()
    assert full.state.optimizer_step == resumed.state.optimizer_step == 6


def test_pmetal_optimizer_reports_clipping_and_finite_metrics():
    runner = PmetalOptimizerPath(np.zeros(2), accumulation_steps=1, max_grad_norm=1.0, total_steps=2)
    metric = runner.add_microbatch(np.array([3.0, 4.0]), tokens=4)
    assert metric["unclipped_gradient_norm"] == 5.0
    assert metric["clipped_gradient_norm"] == 1.0
    assert metric["update_norm"] > 0
