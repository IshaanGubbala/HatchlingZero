"""Generate a fixed-weight fixture for the Rust hz0a-pmetal-tensor <-> Python
native_model.py cross-language parity test.

Different RNGs across languages never produce identical initial weights
even with "the same seed", so this dumps the Python reference's actual
initialized parameter values (in its own parameters() order, which the
Rust TinyModel::parameters_mut() mirrors exactly) plus a forward+backward
pass's logits/loss/gradients, for the Rust side to load and compare against
directly -- the strongest form of "PMetal and the simple reference agree".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel
from restart.hz0a_pmetal.python.pmetal_reference import adamw_step


def main() -> None:
    vocab, dim, heads, d_k, d_v, d_ff = 5, 4, 2, 2, 2, 6
    attention_indices = [1]
    layers = 2
    model = NativeTinyHZ0AModel(vocab, dim, layers, heads, d_k, d_v, d_ff, attention_indices, seed=7)

    token_ids = np.array([[0, 1, 2]])
    targets = np.array([[1, 2, 0]])
    loss, _ = model.loss_and_backward(token_ids, targets)

    flat_params = model.flat_parameters().tolist()
    flat_grads = np.concatenate([p.grad.reshape(-1) for p in model.parameters()]).astype(np.float64).tolist()
    param_shapes = [{"name": p.name, "shape": list(p.data.shape)} for p in model.parameters()]

    logits, _ = model.forward(token_ids)

    flat_grads_f64 = np.array(flat_grads, dtype=np.float64)
    flat_params_f64 = np.array(flat_params, dtype=np.float64)
    optimizer_result = adamw_step(flat_params_f64, flat_grads_f64, learning_rate=1e-4, weight_decay=0.0)
    updated_parameters = optimizer_result.parameters.tolist()

    fixture = {
        "config": {"vocab": vocab, "dim": dim, "heads": heads, "d_k": d_k, "d_v": d_v, "d_ff": d_ff, "attention_indices": attention_indices, "layers": layers},
        "token_ids": token_ids[0].tolist(),
        "targets": targets[0].tolist(),
        "flat_parameters": flat_params,
        "param_shapes": param_shapes,
        "loss": loss,
        "logits": logits[0].reshape(-1).tolist(),
        "flat_gradients": flat_grads,
        "updated_parameters_after_one_adamw_step": updated_parameters,
        "adamw_config": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.0},
    }
    output = Path("restart/hz0a_pmetal/crates/hz0a-pmetal-tensor/tests/fixtures/tiny_model_parity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {output}, loss={loss}, {len(flat_params)} parameter values")


if __name__ == "__main__":
    main()
