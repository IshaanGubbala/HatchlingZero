"""Bidirectional MLX <-> PyTorch checkpoint converter for HZ-0A.

Real, verified mapping (checked directly against both
`reference/hz0a_mlx_model.py` and `reference/hz0a_torch_model.py` on a
tiny model, not assumed from the architectures "matching" in the
abstract): weight tensor SHAPES are identical between frameworks
(`nn.Linear` uses `(out_features, in_features)` on both sides -- no
transpose needed, a direct array copy), but a few parameter NAMES
differ:

- MLX puts the FFN directly on the block (`blocks.i.gate/up/down`);
  Torch nests it under `blocks.i.mlp.gate/up/down`.
- For RECURRENT mixer layers (gdn2/gdn2_fix) only, MLX's
  `blocks.i.mixer.out` is Torch's `blocks.i.mixer.out_proj`.
- For ATTENTION layers, `blocks.i.mixer.out`/`blocks.i.mixer.qkv` are
  named identically on both sides, but `qkv`'s ROWS (weight) and
  entries (bias) need a real PERMUTATION, not just a name copy: Torch
  reshapes the combined projection head-major
  (`.view(B,T,heads,3*d_k).chunk(3,dim=-1)` -> column index
  `head*3*d_k + component*d_k + d`), MLX reshapes it component-major
  (`.reshape(B,T,3,heads,head_dim)` -> column index
  `component*heads*head_dim + head*head_dim + d`). Same weight SHAPE,
  different semantic layout -- confirmed empirically (an initial,
  naive 1:1-copy version of this converter passed shape/key-matching
  tests but failed real forward-pass parity by ~1.4 max abs diff;
  `out`/`out_proj`'s own reshape order was checked and found to
  already match between frameworks, so only `qkv` needs this).
  GDN2/GDN2Fix's `in_proj` does NOT need permutation -- both
  frameworks reshape it identically (`(B,T,6,heads,head_dim)`,
  6-way-split outer, heads inner), verified the same way.
- `embedding.weight`, `final_norm.weight`, `blocks.i.norm1/2.weight`,
  `blocks.i.mixer.in_proj.{weight,bias}`, `blocks.i.mixer.decay_a`
  (gdn2_fix only) are named identically on both sides.

Optimizer state is NOT converted (framework-specific, not portable --
resuming training in a different framework starts a fresh optimizer,
same as this project's own existing "no converter, framework switch is
a new run" precedent in docs/rtx3060_windows_setup.md, now narrowed to
just the optimizer state rather than the whole checkpoint).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HZ0AArchSpec:
    vocab_size: int
    dim: int
    layers: int
    heads: int
    d_ff: int
    attention_indices: tuple[int, ...]
    mixer: str  # "gdn2" or "gdn2_fix"


def _qkv_permutation(heads: int, head_dim: int) -> np.ndarray:
    """`perm` such that `mlx_array = torch_array[perm]` converts a qkv
    weight's rows (or a qkv bias's entries) from Torch's head-major
    layout to MLX's component-major layout. Apply `np.argsort(perm)`
    for the reverse (MLX -> Torch) direction."""
    perm = np.zeros(3 * heads * head_dim, dtype=np.int64)
    for head in range(heads):
        for component in range(3):
            for d in range(head_dim):
                torch_idx = head * 3 * head_dim + component * head_dim + d
                mlx_idx = component * heads * head_dim + head * head_dim + d
                perm[mlx_idx] = torch_idx
    return perm


def build_key_mapping(spec: HZ0AArchSpec) -> dict[str, str]:
    """Returns {mlx_key: torch_key} for every real model parameter (not
    optimizer state) in this architecture. Verified 1:1, both directions
    -- every MLX key maps to exactly one Torch key and vice versa, same
    count, same shapes (checked in tests, not assumed here)."""
    mapping: dict[str, str] = {
        "embedding.weight": "embedding.weight",
        "final_norm.weight": "final_norm.weight",
    }
    for i in range(spec.layers):
        is_attention = i in spec.attention_indices
        mapping[f"blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
        mapping[f"blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
        for name in ("gate", "up", "down"):
            for part in ("weight", "bias"):
                mapping[f"blocks.{i}.{name}.{part}"] = f"blocks.{i}.mlp.{name}.{part}"
        if is_attention:
            for part in ("weight", "bias"):
                mapping[f"blocks.{i}.mixer.qkv.{part}"] = f"blocks.{i}.mixer.qkv.{part}"
                mapping[f"blocks.{i}.mixer.out.{part}"] = f"blocks.{i}.mixer.out.{part}"
        else:
            for part in ("weight", "bias"):
                mapping[f"blocks.{i}.mixer.in_proj.{part}"] = f"blocks.{i}.mixer.in_proj.{part}"
                mapping[f"blocks.{i}.mixer.out.{part}"] = f"blocks.{i}.mixer.out_proj.{part}"
            if spec.mixer == "gdn2_fix":
                mapping[f"blocks.{i}.mixer.decay_a"] = f"blocks.{i}.mixer.decay_a"
    return mapping


def load_mlx_checkpoint_arrays(checkpoint_dir: Path) -> dict[str, np.ndarray]:
    """Reads a real MLX checkpoint dir's state.json + per-array .npy
    files (the exact format scripts/hz0a_native_stage_runner.py writes)
    -- returns {mlx_key: numpy_array} for group=="model" arrays only."""
    payload = json.loads((checkpoint_dir / "state.json").read_text())
    arrays = {}
    for item in payload["arrays"]:
        if item["group"] != "model":
            continue
        arrays[item["key"]] = np.load(str(checkpoint_dir / item["file"]))
    return arrays


def mlx_checkpoint_to_torch_state_dict(checkpoint_dir: Path, spec: HZ0AArchSpec) -> dict[str, np.ndarray]:
    """Real conversion: MLX checkpoint dir -> a dict ready for
    `torch.from_numpy(v)` per value, keyed by Torch's own parameter
    names. Caller does the numpy->torch.Tensor step (keeps this module
    torch-independent, matching reference/hz0h_bdh_mlx.py's own
    convention of not importing the other framework at module level)."""
    mlx_arrays = load_mlx_checkpoint_arrays(checkpoint_dir)
    mapping = build_key_mapping(spec)
    head_dim = spec.dim // spec.heads
    inv_perm = np.argsort(_qkv_permutation(spec.heads, head_dim))
    torch_state = {}
    for mlx_key, torch_key in mapping.items():
        if mlx_key not in mlx_arrays:
            raise KeyError(f"expected MLX key {mlx_key!r} not found in checkpoint (arch spec mismatch?)")
        value = mlx_arrays[mlx_key]
        if mlx_key.endswith("mixer.qkv.weight"):
            value = value[inv_perm, :]
        elif mlx_key.endswith("mixer.qkv.bias"):
            value = value[inv_perm]
        torch_state[torch_key] = value
    return torch_state


def torch_state_dict_to_mlx_arrays(torch_state: dict[str, np.ndarray], spec: HZ0AArchSpec) -> dict[str, np.ndarray]:
    """Reverse: a Torch state_dict (already converted to numpy by the
    caller, e.g. `{k: v.numpy() for k, v in model.state_dict().items()}`)
    -> {mlx_key: numpy_array}, ready to `mx.array(v)` and load into an
    MLX model or write out as a real MLX checkpoint."""
    mapping = build_key_mapping(spec)
    torch_to_mlx = {v: k for k, v in mapping.items()}
    head_dim = spec.dim // spec.heads
    perm = _qkv_permutation(spec.heads, head_dim)
    mlx_arrays = {}
    for torch_key, mlx_key in torch_to_mlx.items():
        if torch_key not in torch_state:
            raise KeyError(f"expected Torch key {torch_key!r} not found in state_dict (arch spec mismatch?)")
        value = torch_state[torch_key]
        if mlx_key.endswith("mixer.qkv.weight"):
            value = value[perm, :]
        elif mlx_key.endswith("mixer.qkv.bias"):
            value = value[perm]
        mlx_arrays[mlx_key] = value
    return mlx_arrays


def write_mlx_checkpoint(checkpoint_dir: Path, mlx_arrays: dict[str, np.ndarray], *, step: int = 0, tokens_seen: int = 0) -> None:
    """Writes a real, loadable MLX checkpoint dir (state.json + per-key
    .npy files) from a {mlx_key: numpy_array} dict -- the same format
    scripts/hz0a_native_stage_runner.py's own save_checkpoint produces,
    minus optimizer state (not portable, see module docstring)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arrays_meta = []
    for index, (key, value) in enumerate(sorted(mlx_arrays.items())):
        filename = f"model-{index:04d}.npy"
        np.save(str(checkpoint_dir / filename), value)
        arrays_meta.append({"group": "model", "key": key, "shape": list(value.shape), "file": filename})
    payload = {
        "step": step, "tokens_seen": tokens_seen, "batch_index": 0, "microbatch_count": 0,
        "epoch_or_data_pass": 0, "best_validation_loss": None, "milestones_hit": [],
        "metrics": [], "arrays": arrays_meta,
    }
    (checkpoint_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")
