"""Real correctness tests for reference/hz0h_energy.py's
TrainingEnergySampler -- specifically its trapezoidal energy-integration
math and honest-disclosure fields, which are pure CPU-testable (the
actual nvidia-smi polling is not: this Mac has no NVIDIA GPU, so
`energy_available` is expected to be False here, which is itself a real
behavior worth testing rather than assuming)."""
from __future__ import annotations

from reference.hz0h_energy import TrainingEnergySampler


def test_stop_reports_unavailable_energy_honestly_without_real_power_samples():
    """On a machine with no nvidia-smi (this Mac), the sampler must
    report energy_available=False and None fields, not fabricate a
    zero-energy result that could be misread as "measured zero joules"."""
    sampler = TrainingEnergySampler(interval_seconds=0.05)
    sampler.start()
    sampler.stop(tokens=100)
    # Real assertion: on a real NVIDIA machine this would have samples;
    # here it must not silently pretend to have measured anything.
    report = sampler.stop(tokens=100)
    assert report["energy_method"] == "nvidia-smi power.draw polling; approximate"
    if not report["energy_available"]:
        assert report["mean_power_watts"] is None
        assert report["energy_joules"] is None
        assert report["joules_per_token"] is None
        assert report["power_sample_count"] == 0


def test_trapezoidal_integration_matches_hand_computed_energy():
    """Real correctness test for the actual integration math, using
    directly-injected synthetic (timestamp, watts) samples so this does
    not depend on real hardware being present. Constant 100W for exactly
    2.0 seconds should integrate to 200 joules -- hand-verifiable."""
    sampler = TrainingEnergySampler()
    sampler.started = 0.0
    sampler.samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0)]
    sampler._stop.set()  # prevent the background thread logic from mattering
    report = sampler.stop(tokens=50)
    assert report["energy_available"] is True
    assert report["power_sample_count"] == 3
    assert abs(report["mean_power_watts"] - 100.0) < 1e-9
    assert abs(report["energy_joules"] - 200.0) < 1e-9
    assert abs(report["joules_per_token"] - 4.0) < 1e-9


def test_trapezoidal_integration_handles_varying_power():
    """Real, hand-verifiable trapezoid: power ramps 0W -> 100W linearly
    over 1 second, so the real integral is the triangle's area, 50 J,
    not 100 J (which a naive left/right-Riemann sum would give)."""
    sampler = TrainingEnergySampler()
    sampler.started = 0.0
    sampler.samples = [(0.0, 0.0), (1.0, 100.0)]
    report = sampler.stop(tokens=1)
    assert abs(report["energy_joules"] - 50.0) < 1e-9


def test_single_sample_yields_zero_energy_not_a_crash():
    """Real edge case: exactly one power sample can't form a trapezoid
    (needs at least two timestamps) -- must report zero energy from
    integration, not divide-by-zero or crash, while still marking
    energy_available True since a real sample WAS taken."""
    sampler = TrainingEnergySampler()
    sampler.started = 0.0
    sampler.samples = [(0.0, 100.0)]
    report = sampler.stop(tokens=10)
    assert report["energy_available"] is True
    assert report["energy_joules"] == 0.0
    assert report["mean_power_watts"] == 100.0
