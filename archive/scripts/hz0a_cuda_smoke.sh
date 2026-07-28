#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python -m hz0.backend_check
pytest

python -m hz0.train --config configs/hz0a-tiny.yaml --max-steps 4
python -m hz0.train --config configs/hz0a-tiny.yaml --model-key baseline --max-steps 4

python -m hz0.eval_cli \
  --config configs/hz0a-tiny.yaml \
  --checkpoint outputs/hz0a-tiny/latest.pt

python -m hz0.eval_cli \
  --config configs/hz0a-tiny.yaml \
  --model-key baseline \
  --checkpoint outputs/hz0a-tiny-baseline/latest.pt

python -m hz0.compare_cli \
  --config configs/hz0a-tiny.yaml \
  --hybrid-checkpoint outputs/hz0a-tiny/latest.pt \
  --baseline-checkpoint outputs/hz0a-tiny-baseline/latest.pt
