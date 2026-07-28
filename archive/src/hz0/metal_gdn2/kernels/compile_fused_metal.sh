#!/bin/bash
# ---------------------------------------------------------------------------
# Offline sanity compile: gdn2_fused_sequence.metal → .air → .metallib.
#
# NOTE: Round-6 onwards the .metal is BODY-ONLY. MLX's runtime path wraps it
# with its own `[[kernel]] void gdn2_fused_fwd(...)` entry + thread-position
# attributes. For offline `xcrun metal -c` we provide an analogous wrapper.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/gdn2_fused_sequence.metal"
TMP_METAL="$(mktemp -t gdn2_fused_sequence_XXXXXX.metal)"
AIR="$SCRIPT_DIR/gdn2_fused_sequence.air"
LIB="$SCRIPT_DIR/gdn2_fused_sequence.metallib"

# HZ-0A 110M canonical shape for offline compile.
B=2; T=256; H=12; Dk=64; Dv=64

./.venv/bin/python - "$SRC" "$B" "$T" "$H" "$Dk" "$Dv" > "$TMP_METAL" <<'PY'
import sys
src_path, B, T, H, Dk, Dv = sys.argv[1], *sys.argv[2:7]
body = open(src_path, encoding="utf-8").read()
body = body.format(B=int(B), T=int(T), H=int(H), Dk=int(Dk), Dv=int(Dv))
out = []
out.append('#include <metal_stdlib>')
out.append('using namespace metal;')
out.append('')
# Mirror the MLX runtime wrapper shape: kernel signature with auto-bound
# pointers (named after input/output names) plus thread-position attrs.
out.append('kernel void gdn2_fused_fwd_offline(')
out.append('    device const float *q           [[buffer(0)]],')
out.append('    device const float *k           [[buffer(1)]],')
out.append('    device const float *v           [[buffer(2)]],')
out.append('    device const float *d           [[buffer(3)]],')
out.append('    device const float *e           [[buffer(4)]],')
out.append('    device const float *w           [[buffer(5)]],')
out.append('    device const float *state_in    [[buffer(6)]],')
out.append('    device       float *out         [[buffer(7)]],')
out.append('    device       float *state_out   [[buffer(8)]],')
out.append('    uint3  threadgroup_position_in_grid [[threadgroup_position_in_grid]],')
out.append('    uint3  thread_position_in_threadgroup [[thread_position_in_threadgroup]]')
out.append(') {')
out.append(body)
out.append('}')
sys.stdout.write('\n'.join(out))
PY

echo "[1/3] metal -c (offline wrapper; shapes=$B,$T,$H,$Dk,$Dv)"
xcrun metal -c "$TMP_METAL" -o "$AIR"

echo "[2/3] metallib $AIR -> $LIB"
xcrun metallib "$AIR" -o "$LIB"

echo "[3/3] sizes:"
ls -la "$AIR" "$LIB"
rm -f "$TMP_METAL"
echo "✓ Built $LIB"
