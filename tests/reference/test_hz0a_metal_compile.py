import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_native_metal_forward_kernel_compiles(tmp_path: Path):
    if subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode != 0:
        pytest.skip("Apple Metal toolchain unavailable")
    output = tmp_path / "gdn2_forward.metallib"
    result = subprocess.run([sys.executable, "scripts/hz0a_compile_metal.py", "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.exists() and output.stat().st_size > 0
