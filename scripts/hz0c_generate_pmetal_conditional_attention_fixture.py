"""Generate a fixed-weight fixture for the Rust hz0a-pmetal-kernel
<-> Python `reference/hz0c_surprise_trigger.py::masked_anchor_attention`
cross-language parity test -- the "Python-reference machine-readable
parity report" C8 has named as open, distinct from (and stronger than)
the CPU-vs-GPU-only parity already checked inside the Rust crates
(that only proves the two Rust kernels agree with EACH OTHER, not that
either agrees with the actual Python reference the model uses).

Same fixed-weight-dump pattern as
`scripts/hz0a_generate_rust_parity_fixture.py`: different RNGs across
languages never produce identical values even with "the same seed", so
this dumps the Python side's actual random inputs/weights plus its
forward output and gradients (via `mx.grad`), for the Rust side to load
and compare against directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from reference.hz0c_surprise_trigger import masked_anchor_attention


def main() -> None:
    batch, seq, dim, heads = 2, 3, 4, 2
    mx.random.seed(1234)
    x = mx.random.normal((batch, seq, dim)) * 0.3
    qkv_w = mx.random.normal((3 * dim, dim)) * 0.1
    qkv_b = mx.random.normal((3 * dim,)) * 0.1
    out_w = mx.random.normal((dim, dim)) * 0.1
    out_b = mx.random.normal((dim,)) * 0.1
    # BINARY triggers only (0.0/1.0) -- PMetal's contract (see
    # `conditional_anchor_attention_f32`'s module-level comment in
    # hz0a-pmetal-kernel): real inference always discretizes the trigger
    # (this project's own Hard Constraint: "Inference triggering must be
    # deterministic and reproducible"), which is what makes the sparse
    # skip-non-triggered-keys optimization valid at all. A genuinely soft
    # (fractional) trigger was tried here first (2026-08-04) and found to
    # diverge from Python's `-1e9` additive-masking softmax by ~0.018 --
    # real, but a training-time-only differentiability property of the
    # Python reference that PMetal (an inference-only kernel) is not
    # trying to reproduce; documented, not silently swept aside.
    trigger = mx.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0]).reshape(batch, seq)
    grad_output = mx.random.normal((batch, seq, dim)) * 0.2

    def loss_fn(x, qkv_w, qkv_b, out_w, out_b):
        out = masked_anchor_attention(x, trigger, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)
        return mx.sum(out * grad_output)

    output = masked_anchor_attention(x, trigger, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)
    grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    grad_x, grad_qkv_w, grad_qkv_b, grad_out_w, grad_out_b = grad_fn(x, qkv_w, qkv_b, out_w, out_b)
    mx.eval(output, grad_x, grad_qkv_w, grad_qkv_b, grad_out_w, grad_out_b)

    def flat(array) -> list[float]:
        return [float(v) for v in array.reshape(-1).tolist()]

    fixture = {
        "config": {"batch": batch, "seq": seq, "dim": dim, "heads": heads},
        "x": flat(x),
        "qkv_weight": flat(qkv_w),
        "qkv_bias": flat(qkv_b),
        "out_weight": flat(out_w),
        "out_bias": flat(out_b),
        "trigger": flat(trigger),
        "grad_output": flat(grad_output),
        "output": flat(output),
        "grad_x": flat(grad_x),
        "grad_qkv_weight": flat(grad_qkv_w),
        "grad_qkv_bias": flat(grad_qkv_b),
        "grad_out_weight": flat(grad_out_w),
        "grad_out_bias": flat(grad_out_b),
    }
    out_path = Path(__file__).resolve().parents[1] / "restart/hz0a_pmetal/crates/hz0a-pmetal-kernel/tests/fixtures/conditional_attention_parity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
