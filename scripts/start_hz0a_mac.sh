#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

./.venv/bin/python -m hz0.prepare_corpus
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-mps.yaml "$@"
