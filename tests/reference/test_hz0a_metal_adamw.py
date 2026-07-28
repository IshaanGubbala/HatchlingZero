import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from restart.hz0a_pmetal.python.pmetal_reference import adamw_step

ROOT = Path(__file__).parents[2]


def test_metal_adamw_matches_numpy_contract(tmp_path: Path):
    if subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode != 0:
        pytest.skip("Apple Metal toolchain unavailable")
    library = tmp_path / "adamw.metallib"
    subprocess.run([sys.executable, "scripts/hz0a_compile_metal.py", "--source", "restart/hz0a_pmetal/metal/adamw.metal", "--output", str(library)], cwd=ROOT, check=True, capture_output=True, text=True)
    executable = tmp_path / "adamw"
    compiled = subprocess.run(["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation", "restart/hz0a_pmetal/metal/adamw_runtime_smoke.swift", "-o", str(executable)], cwd=ROOT, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = json.loads(subprocess.run([str(executable), str(library)], check=True, capture_output=True, text=True).stdout)
    parameters = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=np.float32)
    gradients = np.array([1, -2, 3, -4, 5, -6, 7, -8], dtype=np.float32)
    expected = adamw_step(parameters, gradients, learning_rate=1e-4, weight_decay=0.01)
    np.testing.assert_allclose(result["parameters"], expected.parameters, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(result["first"], expected.state.first_moment, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(result["second"], expected.state.second_moment, rtol=2e-5, atol=2e-7)
