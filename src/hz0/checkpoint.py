from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import torch


_CHECKPOINT_KIND_DIR = {"full", "model_only"}


def _atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
    kind: str,
    write_latest: bool,
    *,
    write_kind_sidecar: bool = True,
) -> None:
    """Write ``payload`` to ``path`` atomically and verify it's reloadable.

    Strategy (per ``docs/hz0b-mem-fix-plan-2026-07-26.md`` Phase 8):

    1. Serialize to a sibling ``.tmp`` file so a partial write never
       overwrites the live checkpoint.
    2. ``fsync`` the file descriptor so the bytes are durable on disk
       before the rename.
    3. ``os.replace`` the temp onto the live path. ``os.replace`` is
       atomic on POSIX (unlink + rename under the hood) and survives
       crashes mid-rename.
    4. Cheap verification: ``zipfile.is_zipfile(path)`` confirms the
       central directory is intact. This was the exact failure mode in
       the v1 training run: ``PytorchStreamReader failed reading zip
       archive: failed finding central directory``. A full
       ``torch.load`` here would be O(file-size) RAM use (peaks at 1.3+
       GB for a state dict); ``zipfile.is_zipfile`` is O(header-size).
    5. Optionally mirror to ``latest.pt`` (only for full saves; model-only
       snapshots don't claim to be a fresh resume point). The recursive
       mirror pass sets ``write_kind_sidecar=False`` so we don't leak
       ``latest.kind.txt`` into the output directory.
    6. Write a ``.kind.txt`` sidecar so ``train.py`` cleanup can prune
       full and model_only checkpoints separately without re-reading
       every ``.pt`` to discover its kind. Skipped for ``latest.pt`` to
       avoid an orphan sidecar.
    """
    if kind not in _CHECKPOINT_KIND_DIR:
        raise ValueError(f"Unknown checkpoint kind: {kind!r}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        # Cheap post-save verification: outer zip integrity.
        if not zipfile.is_zipfile(path):
            raise RuntimeError(
                f"Checkpoint verification failed: {path} is not a valid zip archive"
            )
        if write_kind_sidecar:
            kind_path = path.with_suffix(".kind.txt")
            kind_tmp = kind_path.with_suffix(kind_path.suffix + ".tmp")
            kind_tmp.write_text(kind, encoding="utf-8")
            try:
                with open(kind_tmp, "rb") as fh:
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(kind_tmp, kind_path)
            finally:
                if kind_tmp.exists():
                    try:
                        kind_tmp.unlink()
                    except OSError:
                        pass
        if write_latest:
            latest_path = path.parent / "latest.pt"
            _atomic_torch_save(
                payload,
                latest_path,
                kind=kind,
                write_latest=False,
                write_kind_sidecar=False,
            )
    except BaseException:
        # Cleanup orphan ``.tmp`` files if anything went wrong (including
        # a ``torch.save`` partway failure before fsync/rename could run).
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    *,
    model_only: bool = False,
) -> Path:
    """Save a training checkpoint atomically.

    ``model_only=True`` skips serializing the optimizer state (which is
    typically 2× the model for AdamW and is the dominant storage cost)
    AND skips mirroring to ``latest.pt``. Routinely-saved ``step_*``
    checkpoints should be ``model_only=True``; full resumable checkpoints
    (which include the optimizer) should be saved much less often per
    the plan's ``save_optimizer_every`` knob.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"step_{step:07d}.pt"

    kind = "model_only" if model_only else "full"
    payload: dict[str, Any]
    if model_only:
        payload = {
            "step": int(step),
            "model": model.state_dict(),
            "config": config,
            "metrics": metrics or {},
            "checkpoint_kind": "model_only",
        }
    else:
        payload = {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "metrics": metrics or {},
            "checkpoint_kind": "full",
        }

    # Only ``full`` saves mirror to latest.pt. The ``latest.pt`` symlink
    # always points at a *resumable* checkpoint; model-only snapshots are
    # discardable.
    _atomic_torch_save(payload, checkpoint_path, kind=kind, write_latest=not model_only)

    if metrics is not None:
        metrics_path = out_dir / f"step_{step:07d}.json"
        metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        metrics_tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        try:
            with open(metrics_tmp, "rb") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(metrics_tmp, metrics_path)
        finally:
            if metrics_tmp.exists():
                try:
                    metrics_tmp.unlink()
                except OSError:
                    pass

    return checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def prune_checkpoints(
    output_dir: str | Path,
    keep_last_full: int,
    keep_last_model_only: int,
) -> int:
    """Drop old checkpoints beyond the per-kind retention cap.

    Reads the ``.kind.txt`` sidecar written by ``save_checkpoint`` to
    classify each ``step_*.pt``. Checkpoints without a sidecar are
    treated as ``full`` (legacy default). When a ``step_*.pt`` is
    pruned, the sibling ``step_*.json`` (eval-time metrics) and the
    ``.kind.txt`` sidecar are pruned alongside it so the directory
    doesn't accumulate orphan eval metrics over a long run.

    Returns the number of files pruned (counted across ``.pt``,
    ``.kind.txt``, and ``.json`` siblings).
    """
    if keep_last_full < 0 or keep_last_model_only < 0:
        raise ValueError("keep_last_* must be non-negative")

    out_dir = Path(output_dir)
    full_paths: list[Path] = []
    model_only_paths: list[Path] = []
    # Legacy checkpoints predating the kind-sidecar scheme (i.e. anything
    # before this commit) are treated as ``full`` by default. Only files
    # with an explicit ``model_only`` sidecar take the other branch; that
    # way old artifacts still count toward ``keep_last_full`` and don't get
    # silently re-classified at cleanup time.
    for pt_file in out_dir.glob("step_*.pt"):
        kind = "full"  # legacy default
        kind_file = pt_file.with_suffix(".kind.txt")
        if kind_file.exists():
            try:
                kind = kind_file.read_text(encoding="utf-8").strip() or "full"
            except OSError:
                kind = "full"
        if kind == "model_only":
            model_only_paths.append(pt_file)
        else:
            full_paths.append(pt_file)

    pruned = 0

    def _by_step(p: Path) -> int:
        stem = p.stem  # step_0000425
        return int(stem.split("_")[-1])

    def _drop_siblings(pt_file: Path) -> None:
        nonlocal pruned
        for sibling in (
            pt_file,
            pt_file.with_suffix(".kind.txt"),
            pt_file.with_suffix(".json"),
        ):
            if sibling.exists():
                try:
                    sibling.unlink()
                except OSError:
                    pass
                pruned += 1
        # ``.kind.tmp`` / ``.json.tmp`` orphans from interrupted saves
        for tmp in (
            pt_file.with_suffix(".kind.txt.tmp"),
            pt_file.with_suffix(".json.tmp"),
        ):
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _prune(paths: list[Path], keep_last: int) -> None:
        if keep_last == 0:
            for old in paths:
                _drop_siblings(old)
            return
        paths_sorted = sorted(paths, key=_by_step)
        for old in paths_sorted[:-keep_last]:
            _drop_siblings(old)

    _prune(full_paths, keep_last_full)
    _prune(model_only_paths, keep_last_model_only)
    return pruned
