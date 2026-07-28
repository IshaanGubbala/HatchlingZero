from __future__ import annotations

import importlib
import json
import platform
import sys

import torch


def dependency_status(name: str) -> str:
    try:
        importlib.import_module(name)
    except Exception as exc:
        return f"missing: {exc}"
    return "ok"


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "dependencies": {
            "fla": dependency_status("fla"),
            "flash_attn": dependency_status("flash_attn"),
            "triton": dependency_status("triton"),
            "einops": dependency_status("einops"),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
