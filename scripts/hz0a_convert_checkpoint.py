"""CLI: convert a real HZ-0A checkpoint between MLX and PyTorch formats.

Usage:
  python3 scripts/hz0a_convert_checkpoint.py torch-to-mlx \
    --torch-checkpoint outputs/rtx3060_run/checkpoint.pt \
    --mlx-out outputs/converted_mlx_checkpoint \
    --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2304 \
    --attention-indices 4,9,14,19,24,29 --mixer gdn2_fix

  python3 scripts/hz0a_convert_checkpoint.py mlx-to-torch \
    --mlx-checkpoint outputs/hz0g_g1_gdn2_fix_301m/native_metal_checkpoint_best_full_holdout \
    --torch-out outputs/converted_torch_checkpoint.pt \
    --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2304 \
    --attention-indices 4,9,14,19,24,29 --mixer gdn2_fix

Real, disclosed precision note (see tests/reference/test_hz0a_checkpoint_converter.py):
forward-pass parity after conversion is real but not bit-exact -- max
abs logit diff ~0.005-0.006 on a tiny test model, from GDN2Fix's
softplus/sigmoid math being computed via different (both valid,
non-bit-identical) numerically-stable formulas in MLX vs PyTorch.
Verify with a real forward-pass check on your own converted checkpoint
before trusting it for anything beyond casual inspection -- do not
assume this scales the same way at 301M params without checking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reference.hz0a_checkpoint_converter import (
    HZ0AArchSpec, mlx_checkpoint_to_torch_state_dict, torch_state_dict_to_mlx_arrays, write_mlx_checkpoint,
)


def _spec_from_args(args) -> HZ0AArchSpec:
    attention_indices = tuple(int(x) for x in args.attention_indices.split(",")) if args.attention_indices else ()
    return HZ0AArchSpec(vocab_size=args.vocab_size, dim=args.dim, layers=args.layers, heads=args.heads, d_ff=args.d_ff, attention_indices=attention_indices, mixer=args.mixer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="direction", required=True)

    for direction in ("torch-to-mlx", "mlx-to-torch"):
        p = sub.add_parser(direction)
        p.add_argument("--vocab-size", type=int, required=True)
        p.add_argument("--dim", type=int, required=True)
        p.add_argument("--layers", type=int, required=True)
        p.add_argument("--heads", type=int, required=True)
        p.add_argument("--d-ff", type=int, required=True)
        p.add_argument("--attention-indices", type=str, default="", help="comma-separated, e.g. 4,9,14,19,24,29")
        p.add_argument("--mixer", choices=("gdn2", "gdn2_fix"), default="gdn2_fix")
        if direction == "torch-to-mlx":
            p.add_argument("--torch-checkpoint", type=Path, required=True, help=".pt file with a 'model' key (state_dict), matching scripts/hz0a_torch_stage2_runner.py's save format")
            p.add_argument("--mlx-out", type=Path, required=True)
        else:
            p.add_argument("--mlx-checkpoint", type=Path, required=True, help="MLX checkpoint directory (contains state.json)")
            p.add_argument("--torch-out", type=Path, required=True)

    args = parser.parse_args()
    spec = _spec_from_args(args)

    if args.direction == "torch-to-mlx":
        import torch
        blob = torch.load(str(args.torch_checkpoint), map_location="cpu")
        state_dict = blob["model"] if "model" in blob else blob
        torch_state_np = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
        mlx_arrays = torch_state_dict_to_mlx_arrays(torch_state_np, spec)
        write_mlx_checkpoint(args.mlx_out, mlx_arrays)
        print(f"Wrote MLX checkpoint to {args.mlx_out} ({len(mlx_arrays)} arrays)")
    else:
        import torch
        torch_state_np = mlx_checkpoint_to_torch_state_dict(args.mlx_checkpoint, spec)
        state_dict = {k: torch.from_numpy(v) for k, v in torch_state_np.items()}
        args.torch_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": state_dict}, str(args.torch_out))
        print(f"Wrote Torch checkpoint to {args.torch_out} ({len(state_dict)} tensors)")


if __name__ == "__main__":
    main()
