#!/bin/bash
# Compile Metal MSL kernels to .metallib

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SOURCE="${SCRIPT_DIR}/gdn2_backward.metal"
OUTPUT_LIB="${SCRIPT_DIR}/gdn2_backward.metallib"

echo "Compiling Metal kernels..."
echo "Source: ${KERNEL_SOURCE}"
echo "Output: ${OUTPUT_LIB}"

# Check if xcrun is available (requires Xcode)
if ! command -v xcrun &> /dev/null; then
    echo "✗ xcrun not found. Install Xcode Command Line Tools:"
    echo "  xcode-select --install"
    exit 1
fi

# Compile MSL to metallib
# Target: macOS with Metal 3.0+
echo "\n[1/2] Compiling MSL to intermediate..."
xcrun -sdk macosx metal \
    -ffast-math \
    -O3 \
    -c "${KERNEL_SOURCE}" \
    -o "${SCRIPT_DIR}/gdn2_backward.ir"

echo "✓ Intermediate compiled"

echo "\n[2/2] Linking to metallib..."
xcrun -sdk macosx metallib \
    "${SCRIPT_DIR}/gdn2_backward.ir" \
    -o "${OUTPUT_LIB}"

echo "✓ Metallib created: ${OUTPUT_LIB}"
echo "  Size: $(du -h ${OUTPUT_LIB} | cut -f1)"

echo "\nCompilation complete!"
echo "Next: Test with gdn2_backward_wrapper.py"
