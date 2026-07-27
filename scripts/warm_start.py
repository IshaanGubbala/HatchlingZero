"""Warm-start adapter for architecture-drift checkpoints.

The HZ-0A baseline was trained with no scratchpad. The HZ-0B model adds
``scratchpad_query/key/value/gate`` parameters once ``scratchpad_slots > 0``.
A direct ``model.load_state_dict(source["model"])`` against a wider HZ-0B
model fails with a missing-key error on those scratchpad parameters.

This script warms an HZ-0A checkpoint into an HZ-0B model by:

1. Building the HZ-0B model from the supplied config.
2. ``load_state_dict(strict=False)`` so the freshly-init scratchpad params
   stay random.
3. Refusing to silently drop or randomise any **non-scratchpad** mismatch.
   If the source checkpoint and the target architecture disagree on anything
   outside the scratchpad block, this script raises rather than continuing.
4. Constructing a fresh AdamW optimiser (matches the new parameter count, so
   ``optimizer.load_state_dict`` cannot blow up later).
5. Saving a ``step_<source_step>.pt`` in the new output dir via the existing
   ``save_checkpoint`` helper, so the standard
   ``python -m hz0.train --resume outputs/.../step_<source_step>.pt`` flow
   continues to work against the new architecture.

Usage::

    python -m hz0.warm_start \
        --source-checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt \
        --output-dir outputs/hz0b-mac-110m-scratchpad-ft \
        --config configs/hz0b-mac-110m-scratchpad-ft.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hz0.checkpoint import save_checkpoint
from hz0.config import Config
from hz0.model import build_model


# Parameter name prefixes that are allowed to be missing in the source.
# Anything outside this allow-list is treated as a real architecture mismatch.
# As of the HZ-0B slot-addressed-routing fix, the scratchpad is an
# ``nn.Module`` and owns its own parameters (``slot_addresses``) on top of
# the four projection ``Linear`` layers it used in HZ-0A v0.
SCRATCHPAD_KEY_ALLOWLIST: tuple[str, ...] = (
    "scratchpad_query.",
    "scratchpad_key.",
    "scratchpad_value.",
    "scratchpad_gate.",
    "scratchpad.slot_addresses",
    # v2 (this iteration): ``scratchpad_norm`` is the LayerNorm on the
    # routing-side scratchpad input that the induction-head fix added to
    # ``HybridLM``. Without this entry the warm-start script would refuse
    # the v2 HZ-0B model back to an HZ-0A source checkpoint.
    "scratchpad_norm.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-step",
        type=int,
        default=None,
        help=(
            "Step number recorded in the new warm-start checkpoint. "
            "Defaults to the step recorded in the source checkpoint."
        ),
    )
    args = parser.parse_args()

    payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    source_state = payload["model"]
    source_step = (
        args.source_step if args.source_step is not None else int(payload.get("step", 0))
    )
    print(f"source_checkpoint={args.source_checkpoint} source_step={source_step}")

    cfg = Config.load(args.config).raw
    model = build_model(cfg["model"])
    missing, unexpected = model.load_state_dict(source_state, strict=False)

    missing_scratchpad = [k for k in missing if k.startswith(SCRATCHPAD_KEY_ALLOWLIST)]
    missing_other = [k for k in missing if not k.startswith(SCRATCHPAD_KEY_ALLOWLIST)]

    print(f"missing_scratchpad_params={missing_scratchpad}")
    print(f"missing_other_params={missing_other}")
    print(f"unexpected_params={unexpected}")

    if missing_other or unexpected:
        raise ValueError(
            "Architecture mismatch on non-scratchpad keys. "
            "Refusing to silently drop or randomise: "
            f"missing_other={missing_other} unexpected={unexpected}"
        )
    if not missing_scratchpad:
        print("note: source checkpoint already had scratchpad params; behaviour is a no-op rename.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    save_checkpoint(args.output_dir, source_step, model, optimizer, cfg)
    out_path = Path(args.output_dir) / f"step_{source_step:07d}.pt"
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
