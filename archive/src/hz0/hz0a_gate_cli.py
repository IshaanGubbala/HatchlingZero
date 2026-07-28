from __future__ import annotations

import argparse
import json
from pathlib import Path

from hz0.hz0a_gate import evaluate_hz0a_gate_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-loss", type=float, required=True)
    parser.add_argument("--required-transformer-step", type=int, default=300)
    parser.add_argument("--min-loss-margin", type=float, default=0.05)
    parser.add_argument("--min-decode-ratio", type=float, default=0.5)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_hz0a_gate_paths(
        scorecard_path=args.scorecard,
        reference_manifest_path=args.reference_manifest,
        reference_loss=args.reference_loss,
        required_transformer_step=args.required_transformer_step,
        min_loss_margin=args.min_loss_margin,
        min_decode_ratio=args.min_decode_ratio,
    )
    text = json.dumps(result, indent=2)
    if args.output_path is not None:
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
