"""Live-training instrumentation for the artifact dashboard requested
2026-09-17: "live visualization of training... showing the actual model
and its weights... or the gradient descent loss function and the
traversal." Two real, distinct pieces, both computed here, cheaply,
alongside the existing eval step (no extra forward/backward passes on
the training path itself):

1. Weight heatmaps -- a fixed set of real weight matrices
   (token_embed.weight, lm_head.weight, local_mixer.qkv.weight; verified
   present via named_parameters() at chat_only=True), downsampled via
   adaptive average pooling to a small heatmap_dim x heatmap_dim grid so
   the JSONL stays small even over thousands of eval points.

2. Loss-landscape + trajectory, via the FIXED RANDOM DIRECTIONS method
   (Li et al., "Visualizing the Loss Landscape of Neural Nets", the
   documented alternative to PCA-of-checkpoints when you don't want to
   store every checkpoint): two random unit vectors in the full
   flattened-parameter space, fixed once at logger construction. Every
   log() call projects (current_params - init_params) onto those two
   directions for a 2D trajectory point (cheap, always computed).
   Periodically (every landscape_every_evals calls), also evaluates loss
   on a small fixed eval batch across a small alpha/beta grid spanning
   those same two directions around the CURRENT weights -- by literally
   perturbing the live model's parameters via
   torch.nn.utils.vector_to_parameters, evaluating, then restoring the
   exact original values. Real, disclosed approximation: no per-layer
   filter normalization (the paper's refinement for prettier landscapes)
   -- plain global directions, scaled relative to the current parameter
   vector's own norm.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

HEATMAP_PARAM_NAMES = ["token_embed.weight", "lm_head.weight", "local_mixer.qkv.weight"]


def _downsample_2d(weight: torch.Tensor, dim: int) -> list[list[float]]:
    x = weight.detach().float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (dim, dim)).squeeze(0).squeeze(0)
    return pooled.cpu().tolist()


class TrainingMetricsLogger:
    def __init__(self, model, out_path: Path, heatmap_dim: int = 24,
                landscape_grid: int = 7, landscape_every_evals: int = 10, seed: int = 1234):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("")  # truncate/start fresh each run
        self.heatmap_dim = heatmap_dim
        self.landscape_grid = landscape_grid
        self.landscape_every_evals = landscape_every_evals
        self._eval_count = 0

        with torch.no_grad():
            flat = parameters_to_vector(model.parameters()).detach().clone()
        self.init_flat = flat
        gen = torch.Generator().manual_seed(seed)  # CPU generator -- mps doesn't support Generator(device="mps")
        d1 = torch.randn(flat.shape, generator=gen).to(flat.device)
        d2 = torch.randn(flat.shape, generator=gen).to(flat.device)
        self.dir1 = d1 / d1.norm()
        self.dir2 = d2 / d2.norm()

    def _weight_norms(self, model) -> dict:
        return {name: float(p.detach().norm().item()) for name, p in model.named_parameters()}

    def _heatmaps(self, model) -> dict:
        named = dict(model.named_parameters())
        out = {}
        for name in HEATMAP_PARAM_NAMES:
            if name in named and named[name].ndim == 2:
                out[name] = {"shape": list(named[name].shape),
                            "grid": _downsample_2d(named[name], self.heatmap_dim)}
        return out

    @torch.no_grad()
    def _landscape(self, model, eval_batch_fn) -> dict:
        flat = parameters_to_vector(model.parameters()).detach().clone()
        span = 0.15 * flat.norm().item()
        alphas = torch.linspace(-1.0, 1.0, self.landscape_grid).tolist()
        betas = torch.linspace(-1.0, 1.0, self.landscape_grid).tolist()
        losses = []
        for a in alphas:
            row = []
            for b in betas:
                perturbed = flat + (a * span) * self.dir1 + (b * span) * self.dir2
                vector_to_parameters(perturbed, model.parameters())
                row.append(eval_batch_fn())
                del perturbed
            losses.append(row)
        vector_to_parameters(flat, model.parameters())
        if torch.cuda.is_available():
            # Real, disclosed mitigation attempt (not a confirmed root-cause
            # fix) for the recurring ~23GB OOM seen on long runs: this loop
            # allocates 49 full-parameter-sized transient tensors (~1.1GB
            # each at 286M params) every landscape_every_evals evals. The
            # CUDA caching allocator should reuse freed blocks of the same
            # size without needing this, but empty_cache() after the burst
            # is cheap insurance against fragmentation actually being (part
            # of) the cause -- profiling to confirm is out of scope here.
            torch.cuda.empty_cache()
        cur_x = float((flat - self.init_flat).dot(self.dir1).item())
        cur_y = float((flat - self.init_flat).dot(self.dir2).item())
        return {"alphas": alphas, "betas": betas, "losses": losses, "span": span,
               "current_xy": [cur_x, cur_y]}

    def log(self, *, step, train_loss, train_acc, val_loss, val_acc, lr, elapsed, model,
           eval_batch_fn=None):
        with torch.no_grad():
            flat = parameters_to_vector(model.parameters()).detach()
            traj_x = float((flat - self.init_flat).dot(self.dir1).item())
            traj_y = float((flat - self.init_flat).dot(self.dir2).item())

        record = {
            "step": step, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr, "elapsed_seconds": elapsed,
            "wall_time": time.time(),
            "trajectory_xy": [traj_x, traj_y],
            "weight_norms": self._weight_norms(model),
            "heatmaps": self._heatmaps(model),
        }
        self._eval_count += 1
        if eval_batch_fn is not None and self._eval_count % self.landscape_every_evals == 1:
            record["landscape"] = self._landscape(model, eval_batch_fn)

        with self.out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
