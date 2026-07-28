from __future__ import annotations

import platform
import sys

import torch

from hz0.model.backends import gdn2_status


def main() -> None:
    status = gdn2_status()
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(
        "mps_available="
        f"{torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}"
    )
    print(f"gdn2_available={status['available']}")
    print(f"gdn2_reason={status['reason']}")


if __name__ == "__main__":
    main()
