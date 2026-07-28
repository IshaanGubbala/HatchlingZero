import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]


def test_metal_backward_matches_torch_oracle(tmp_path: Path):
    if subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode != 0:
        pytest.skip("Apple Metal toolchain unavailable")
    library = tmp_path / "backward.metallib"
    subprocess.run([sys.executable, "scripts/hz0a_compile_metal.py", "--source", "restart/hz0a_pmetal/metal/gdn2_backward.metal", "--output", str(library)], cwd=ROOT, check=True, capture_output=True, text=True)
    executable = tmp_path / "backward"
    compiled = subprocess.run(["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation", "restart/hz0a_pmetal/metal/gdn2_backward_runtime_smoke.swift", "-o", str(executable)], cwd=ROOT, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    actual = json.loads(subprocess.run([str(executable), str(library)], check=True, capture_output=True, text=True).stdout)
    arrays = {name: torch.tensor(values, dtype=torch.float32).reshape(shape).clone().detach().requires_grad_() for name, values, shape in [("q", [1, 2, 2, 1, 1, -1], (1, 3, 1, 2)), ("k", [0.5, -1, 1, 0.25, -0.5, 2], (1, 3, 1, 2)), ("v", [1, -2, 0.5, 3, -1, 0.25], (1, 3, 1, 2)), ("decay", [0.8, 0.7, 0.6, 0.9, 0.5, 0.4], (1, 3, 1, 2)), ("erase", [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], (1, 3, 1, 2)), ("write", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], (1, 3, 1, 2))]}
    initial = torch.zeros((1, 1, 2, 2), requires_grad=True)
    state = initial
    outputs = []
    for step in range(3):
        state = arrays["decay"][:, step, :, None, :] * (1 - arrays["erase"][:, step, :, None, :]) * state + arrays["write"][:, step, :, :, None] * arrays["v"][:, step, :, :, None] * arrays["k"][:, step, :, None, :]
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, arrays["q"][:, step]))
    torch.stack(outputs, dim=1).sum().backward()
    for name in ("q", "k", "v", "decay", "erase", "write"):
        np.testing.assert_allclose(actual[name], arrays[name].grad.detach().numpy().reshape(-1), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual["initial"], initial.grad.detach().numpy().reshape(-1), rtol=2e-5, atol=2e-6)
