import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]


def test_native_metal_forward_matches_reference(tmp_path: Path):
    if subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode != 0:
        pytest.skip("Apple Metal toolchain unavailable")
    air = tmp_path / "gdn2_forward.metallib"
    subprocess.run([sys.executable, "scripts/hz0a_compile_metal.py", "--output", str(air)], cwd=ROOT, check=True, capture_output=True, text=True)
    executable = tmp_path / "gdn2_runtime_smoke"
    swift = subprocess.run(["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation", "restart/hz0a_pmetal/metal/gdn2_runtime_smoke.swift", "-o", str(executable)], cwd=ROOT, capture_output=True, text=True)
    assert swift.returncode == 0, swift.stderr
    result = json.loads(subprocess.run([str(executable), str(air)], check=True, capture_output=True, text=True).stdout)
    q = np.array([1, 2, 2, 1, 1, -1], dtype=np.float32).reshape(1, 3, 1, 2)
    k = np.array([0.5, -1, 1, 0.25, -0.5, 2], dtype=np.float32).reshape(1, 3, 1, 2)
    v = np.array([1, -2, 0.5, 3, -1, 0.25], dtype=np.float32).reshape(1, 3, 1, 2)
    decay = np.array([0.8, 0.7, 0.6, 0.9, 0.5, 0.4], dtype=np.float32).reshape(1, 3, 1, 2)
    erase = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype=np.float32).reshape(1, 3, 1, 2)
    write = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32).reshape(1, 3, 1, 2)
    state = np.zeros((1, 1, 2, 2), dtype=np.float32)
    expected = np.zeros((1, 3, 1, 2), dtype=np.float32)
    for t in range(3):
        for value in range(2):
            for key in range(2):
                state[0, 0, value, key] = decay[0, t, 0, key] * (1 - erase[0, t, 0, key]) * state[0, 0, value, key] + write[0, t, 0, key] * v[0, t, 0, value] * k[0, t, 0, key]
                expected[0, t, 0, value] += state[0, 0, value, key] * q[0, t, 0, key]
    np.testing.assert_allclose(result["output"], expected.reshape(-1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result["final_state"], state.reshape(-1), rtol=1e-5, atol=1e-6)
