import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]


def test_metal_kernel_matches_locked_recurrent_state_shape(tmp_path: Path):
    if subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode != 0:
        pytest.skip("Apple Metal toolchain unavailable")
    library = tmp_path / "gdn2.metallib"
    subprocess.run([sys.executable, "scripts/hz0a_compile_metal.py", "--output", str(library)], cwd=ROOT, check=True, capture_output=True, text=True)
    executable = tmp_path / "full_shape"
    compiled = subprocess.run(["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation", "restart/hz0a_pmetal/metal/gdn2_full_shape_smoke.swift", "-o", str(executable)], cwd=ROOT, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = json.loads(subprocess.run([str(executable), str(library)], check=True, capture_output=True, text=True).stdout)
    b, steps, heads, values, keys = 1, 2, 12, 64, 64
    def vals(count, scale, offset):
        return np.asarray([((i * 17 + offset) % 101 - 50) / scale for i in range(count)], dtype=np.float32)
    input_count, value_count = b * steps * heads * keys, b * steps * heads * values
    q = vals(input_count, 50, 1).reshape(b, steps, heads, keys)
    k = vals(input_count, 50, 3).reshape(b, steps, heads, keys)
    decay = (vals(input_count, 100, 20) + 0.5).reshape(b, steps, heads, keys)
    erase = (vals(input_count, 100, 40) + 0.5).reshape(b, steps, heads, keys)
    write = (vals(value_count, 100, 60) + 0.5).reshape(b, steps, heads, values)
    vv = vals(value_count, 50, 7).reshape(b, steps, heads, values)
    state = np.zeros((b, heads, values, keys), dtype=np.float32)
    expected = np.zeros((b, steps, heads, values), dtype=np.float32)
    for step in range(steps):
        state = decay[:, step, :, None, :] * (1 - erase[:, step, :, None, :]) * state + write[:, step, :, :, None] * vv[:, step, :, :, None] * k[:, step, :, None, :]
        expected[:, step] = np.einsum("bhvk,bhk->bhv", state, q[:, step])
    np.testing.assert_allclose(result["output"], expected.reshape(-1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result["final_state"], state.reshape(-1), rtol=1e-5, atol=1e-5)
