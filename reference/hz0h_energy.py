"""Best-effort, explicitly labelled training energy sampling for Phase F.

NVIDIA's ``nvidia-smi`` power.draw is a polling estimate, not an energy
counter.  The report therefore includes the method and sample count so an
unavailable sampler cannot be mistaken for measured joules.
"""
from __future__ import annotations

import subprocess
import threading
import time


def _read_power_watts() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return sum(values) / len(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class TrainingEnergySampler:
    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0.0
        self.finished = 0.0

    def start(self) -> None:
        self.started = time.perf_counter()
        self._stop.clear()

        def poll() -> None:
            while not self._stop.is_set():
                watts = _read_power_watts()
                if watts is not None:
                    self.samples.append((time.perf_counter(), watts))
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=poll, name="hz0h-power-sampler", daemon=True)
        self._thread.start()

    def stop(self, *, tokens: int) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self.finished = time.perf_counter()
        elapsed = max(self.finished - self.started, 1e-9)
        watts = [value for _, value in self.samples]
        # Trapezoidal integration over the actual polling timestamps.
        joules = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            joules += (t1 - t0) * (w0 + w1) / 2.0
        return {
            "energy_method": "nvidia-smi power.draw polling; approximate",
            "energy_available": bool(watts),
            "power_sample_count": len(watts),
            "mean_power_watts": sum(watts) / len(watts) if watts else None,
            "energy_joules": joules if watts else None,
            "joules_per_token": joules / tokens if watts and tokens else None,
            "energy_elapsed_seconds": elapsed,
        }
