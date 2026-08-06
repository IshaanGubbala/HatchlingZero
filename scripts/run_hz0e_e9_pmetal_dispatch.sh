#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
METAL_DIR="$ROOT/restart/hz0a_pmetal/metal"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hz0e-e9-dispatch.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

TOKENS="${1:-4096}"
WIDTH="${2:-768}"
ITERATIONS="${3:-20}"

case "$TOKENS" in ''|*[!0-9]*) echo "tokens must be a positive integer" >&2; exit 2;; esac
case "$WIDTH" in ''|*[!0-9]*) echo "width must be a positive integer" >&2; exit 2;; esac
case "$ITERATIONS" in ''|*[!0-9]*) echo "iterations must be a positive integer" >&2; exit 2;; esac
if (( TOKENS == 0 || WIDTH == 0 || ITERATIONS == 0 )); then
    echo "tokens, width, and iterations must be positive" >&2
    exit 2
fi

xcrun -sdk macosx metal -c "$METAL_DIR/moe_dispatch.metal" -o "$TMP_DIR/moe_dispatch.air"
xcrun -sdk macosx metallib "$TMP_DIR/moe_dispatch.air" -o "$TMP_DIR/moe_dispatch.metallib"
xcrun -sdk macosx metal -c "$METAL_DIR/moe_swiglu.metal" -o "$TMP_DIR/moe_swiglu.air"
xcrun -sdk macosx metallib "$TMP_DIR/moe_swiglu.air" -o "$TMP_DIR/moe_swiglu.metallib"
swiftc "$METAL_DIR/moe_dispatch_runtime_smoke.swift" -framework Metal -o "$TMP_DIR/moe_dispatch_smoke"
swiftc "$METAL_DIR/moe_dispatch_benchmark.swift" -framework Metal -o "$TMP_DIR/moe_dispatch_benchmark"
swiftc "$METAL_DIR/moe_swiglu_runtime_smoke.swift" -framework Metal -o "$TMP_DIR/moe_swiglu_smoke"

echo "E9 Metal dispatch smoke:"
"$TMP_DIR/moe_dispatch_smoke" "$TMP_DIR/moe_dispatch.metallib"
echo "E9 Metal SwiGLU expert smoke:"
"$TMP_DIR/moe_swiglu_smoke" "$TMP_DIR/moe_swiglu.metallib"
echo "E9 Metal dispatch benchmark:"
"$TMP_DIR/moe_dispatch_benchmark" "$TMP_DIR/moe_dispatch.metallib" "$TOKENS" "$WIDTH" "$ITERATIONS"
