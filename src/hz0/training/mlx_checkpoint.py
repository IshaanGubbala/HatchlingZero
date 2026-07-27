"""MLX-native atomic checkpointing for Phase 14/15+ training.

Replaces the PyTorch-only `hz0.checkpoint` module for MLX-based training
runs (Phase 14 GDN-2 + Transformer). PyTorch's `torch.save` cannot
serialize MLX Modules — we use `mx.save_safetensors` with atomic writes
(tmp → fsync → rename) and load-back verification.

API mirrors the PyTorch equivalent so callers stay simple:

    save_mlx_checkpoint(dir, step, model, optimizer, cfg, metrics, model_only=...)
    load_mlx_checkpoint(path, model, optimizer)
    prune_mlx_checkpoints(dir, keep_last_full=2, keep_last_model_only=5)
    latest_checkpoint(dir)

Each checkpoint writes three sibling files in `dir`:

    step_NNNNNN.safetensors   weights (+ optimizer state if kind == "full")
    step_NNNNNN.kind.txt      one of {"full", "model_only"}
    step_NNNNNN.metrics.json  optional JSON snapshot of eval metrics

`prune_mlx_checkpoints` walks the directory, groups by the kind sidecar,
keeps the most recent N of each, and unlinks the rest.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_flatten  # top-level mlx.utils, NOT mx.utils


KIND_FULL = "full"
KIND_MODEL_ONLY = "model_only"


def _basename(step: int) -> str:
    return f"step_{step:07d}"


def _flatten_to_dict(prefix: str, mapping) -> dict:
    """Convert an MLX-tree flattening into a {flat_key: array} dict.

    `mlx.utils.tree_flatten(tree)` returns an iterable of
    `(key_path, value)` tuples where `key_path` is either a tuple of
    strings or a single dotted string. We always emit dotted strings
    with the given prefix (e.g. ``opt.`` for optimizer state).
    """
    out: dict[str, "mx.array"] = {}
    for key_path, value in mapping:
        if isinstance(key_path, (list, tuple)):
            key = ".".join(str(k) for k in key_path)
        else:
            key = str(key_path)
        out[prefix + key] = value
    return out


def _model_arrays(model) -> dict:
    if model is None:
        return {}
    return _flatten_to_dict("", tree_flatten(model.parameters()))


def _optimizer_arrays(optimizer) -> dict:
    if optimizer is None:
        return {}
    state = getattr(optimizer, "state", None)
    if not state:
        return {}
    return _flatten_to_dict("opt.", tree_flatten(state))


def _atomic_save_safetensors(arrays: dict, dest: Path, metadata: dict) -> None:
    """Write ``arrays`` to ``dest`` atomically with load-back verification.

    IMPORTANT: mlx 0.32.0's ``mx.save_safetensors`` always APPENDS
    ``.safetensors`` to the supplied path, even if the path already ends in
    that suffix (it then writes ``<path>.safetensors``). So the tmp→rename
    protocol below operates on `<base>.tmp` as input; mlx writes to
    `<base>.tmp.safetensors`; we then rename that artefact to the
    `<dir>/<basename>.safetensors` final location.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    base = str(dest.with_suffix(""))  # strip ".safetensors" from dest
    tmp_input = base + ".tmp"          # mlx will write <tmp_input>.safetensors

    safe_metadata = {str(k): str(v) for k, v in metadata.items()}
    mx.save_safetensors(tmp_input, arrays, metadata=safe_metadata)

    actual_tmp_path = Path(tmp_input + ".safetensors")
    if not actual_tmp_path.exists():
        raise RuntimeError(
            f"MLX safetensors tmp file missing: expected {actual_tmp_path}"
        )

    with open(actual_tmp_path, "rb") as fh:
        os.fsync(fh.fileno())

    try:
        reloaded = dict(mx.load(str(actual_tmp_path)))
    except Exception as e:
        actual_tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MLX safetensors verification failed for {dest}: {e}"
        ) from e
    if set(reloaded.keys()) != set(arrays.keys()):
        actual_tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MLX safetensors key mismatch for {dest}: "
            f"expected {len(arrays)} arrays, got {len(reloaded)}"
        )

    os.replace(str(actual_tmp_path), str(dest))


def _safe_metrics(m: dict) -> dict:
    """Convert mx.array / numpy values to JSON-friendly scalars."""
    out: dict = {}
    for k, v in m.items():
        try:
            if hasattr(v, "tolist") and callable(v.tolist):
                arr = v.tolist()
                if isinstance(arr, list):
                    if len(arr) == 1 and isinstance(arr[0], (int, float)):
                        out[k] = float(arr[0])
                    else:
                        out[k] = [_coerce(x) for x in arr]
                else:
                    out[k] = float(arr)
                continue
        except Exception:
            pass
        try:
            out[k] = float(v)
            continue
        except Exception:
            pass
        out[k] = str(v)
    return out


def _coerce(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def save_mlx_checkpoint(
    directory,
    step: int,
    model,
    optimizer=None,
    cfg: Optional[dict] = None,
    metrics: Optional[dict] = None,
    model_only: bool = False,
    save_optimizer_every: int = 0,
) -> Path:
    """Save an MLX checkpoint atomically.

    Mirrors `hz0.checkpoint.save_checkpoint` semantics. When
    `model_only=True`, optimizer state is dropped unless
    `step % save_optimizer_every == 0` (allowing a periodic full save
    inside an otherwise-cheap model-only cadence).

    Returns the Path to the .safetensors file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if model_only and save_optimizer_every > 0 and (step % save_optimizer_every != 0):
        kind = KIND_MODEL_ONLY
    else:
        kind = KIND_FULL

    arrays = _model_arrays(model)
    if kind == KIND_FULL:
        arrays.update(_optimizer_arrays(optimizer))

    basename = _basename(step)
    safetensors_path = directory / f"{basename}.safetensors"

    n_params = 0
    for k, arr in arrays.items():
        if k.startswith("opt."):
            continue
        try:
            n_params += int(arr.size)
        except Exception:
            pass

    metadata = {
        "step": step,
        "kind": kind,
        "saved_at_unix": int(time.time()),
        "model_param_count": n_params,
        "model_only_requested": model_only,
    }
    if isinstance(cfg, dict):
        for k in ("model_dim", "num_layers", "num_heads", "vocab_size", "lr"):
            if k in cfg:
                metadata[k] = cfg[k]

    _atomic_save_safetensors(arrays, safetensors_path, metadata)

    (directory / f"{basename}.kind.txt").write_text(kind + "\n", encoding="utf-8")
    if metrics is not None:
        metrics_path = directory / f"{basename}.metrics.json"
        tmp_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        tmp_metrics.write_text(json.dumps(_safe_metrics(metrics), indent=2), encoding="utf-8")
        os.replace(tmp_metrics, metrics_path)

    return safetensors_path


def load_mlx_checkpoint(path, model=None, optimizer=None):
    """Load an MLX checkpoint. Returns a dict with step/kind/metrics/path.

    If `path` is a directory, the most recent `step_*.safetensors` is used.
    Optimizer state is loaded if both `optimizer` and a stored state exist.
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("step_*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors in {path}")
        path = candidates[-1]

    arrays = dict(mx.load(str(path)))

    model_weights = {k: v for k, v in arrays.items() if not k.startswith("opt.")}
    opt_arrays = {k[len("opt."):]: v for k, v in arrays.items() if k.startswith("opt.")}

    if model is not None and model_weights:
        model.load_weights(list(model_weights.items()))

    if optimizer is not None and opt_arrays:
        try:
            optimizer.state = opt_arrays
        except Exception:
            pass

    kind = KIND_FULL
    kind_path = path.with_name(path.stem + ".kind.txt")
    if kind_path.exists():
        kind = (kind_path.read_text() or KIND_FULL).strip() or KIND_FULL

    step = None
    try:
        step = int(path.stem.split("_")[1])
    except Exception:
        pass

    metrics = None
    metrics_path = path.with_name(path.stem + ".metrics.json")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None

    return {"step": step, "kind": kind, "metrics": metrics, "path": path}


def prune_mlx_checkpoints(
    directory,
    keep_last_full: int = 2,
    keep_last_model_only: int = 5,
) -> list:
    """Prune old MLX checkpoints, keeping the most recent N of each kind.

    Returns the list of safetensors files that were removed.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    safetensors_files = sorted(
        directory.glob("step_*.safetensors"),
        key=lambda p: p.stat().st_mtime,
    )

    full: list = []
    model_only: list = []
    for f in safetensors_files:
        kind_path = f.with_name(f.stem + ".kind.txt")
        kind = (kind_path.read_text() if kind_path.exists() else KIND_FULL).strip()
        (full if kind == KIND_FULL else model_only).append(f)

    full.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    model_only.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    to_remove = sorted(set(full[keep_last_full:]) | set(model_only[keep_last_model_only:]))

    for safetensors in to_remove:
        stem = safetensors.stem
        for suffix in (".safetensors", ".kind.txt", ".metrics.json"):
            p = directory / f"{stem}{suffix}"
            if p.exists():
                p.unlink()

    return to_remove


def latest_checkpoint(directory):
    """Return Path to the most recent safetensors checkpoint or None."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = sorted(
        directory.glob("step_*.safetensors"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None
