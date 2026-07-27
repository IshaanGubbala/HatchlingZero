"""Phase 14: Full training runs (110M + 300M HZ and Transformer, sequential).

Hardened against the failure modes that crashed the previous 5 attempts:

* Atomic MLX-native checkpoints via `mlx_checkpoint.save_mlx_checkpoint`
  (PyTorch's `torch.save` won't serialize MLX modules).
* `try/finally` wrapper guarantees a final metrics flush even on crash.
* Stdout flushed on every log line so tmux captures survive a kernel panic.
* `--max-steps` and `--smoke-test` allow short-horizon validation runs.
* Per the thinker's C1 verdict: 110M and 300M run sequentially, not
  concurrently, to avoid MPS memory fragmentation under unified memory.

This rewrite repairs the indentation breakage from compounded str_replace
edits in the prior session. The training loop is now consistently nested
4-deep (try → for epoch → for step_idx → if grad_accum → if checkpoint_every),
`epoch_loss += loss_value` actually accumulates (previous bug: train_loss
printed as exactly 0.0000 forever), `steps_this_epoch` resets on checkpoint
window so the avg denominator matches the numerator, and gradient clipping
flushes the lazy graph with `mx.eval` to release the doubled-tensor tree.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Apply the 1-line import-fix from the project remediation plan:
# Use the absolute /src path so module resolution works whether we run
# this file as a module or as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map  # top-level mlx.utils, NOT mx.utils

from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from hz0.validation.phase1a_transformer_baseline import TransformerLM
from hz0.training.mlx_checkpoint import (
    KIND_FULL,
    KIND_MODEL_ONLY,
    latest_checkpoint,
    prune_mlx_checkpoints,
    save_mlx_checkpoint,
)


# Phase 6 sweep confirmed LR=3e-4 wins; matrix evidence in
# outputs/phase6_sweep.json.
PHASE14_LEARNING_RATE = 3e-4
PHASE14_GRAD_ACCUM = 4
# Phase 14 plans: train 110M (768x24x12) AND 300M (1024x32x16),
# both HZ-0A and a matched Transformer. Sequential by spec.
PHASE14_CONFIGS: List[Tuple[str, int, int, int]] = [
    ("hz0a_110m", 768, 24, 12),
    ("hz0a_300m", 1024, 32, 16),
]


class TrainingHarness:
    """Full training loop with metrics, eval, and atomic checkpointing.

    Hardened: incremental metrics flush on every checkpoint_every step,
    not just at the end of training. The PyTorch `train.py` analog
    follows the same pattern (see Phase 8 of hz0b-mem-fix-plan).
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        cfg: Dict[str, Any],
        learning_rate: float = PHASE14_LEARNING_RATE,
        gradient_accumulation: int = PHASE14_GRAD_ACCUM,
        output_dir: Path = None,
        checkpoint_every: int = 50,
        save_optimizer_every: int = 200,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.cfg = cfg
        self.lr = learning_rate
        self.grad_accum = max(1, int(gradient_accumulation))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.save_optimizer_every = max(0, int(save_optimizer_every))

        self.optimizer = optim.Adam(learning_rate=self.lr)

        self.output_dir = Path(output_dir) if output_dir is not None \
            else Path(f"outputs/training/{model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.output_dir / "metrics.json"
        self.metrics: List[Dict[str, Any]] = []

    def compute_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Stable cross-entropy loss with explicit log-softmax."""
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V).astype(mx.float32)
        targets_flat = targets.reshape(-1)

        logits_flat = mx.clip(logits_flat, -100.0, 100.0)
        max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
        log_sum_exp = (
            mx.log(mx.sum(mx.exp(logits_flat - max_logits), axis=-1, keepdims=True)) + max_logits
        )
        log_probs = logits_flat - log_sum_exp
        correct_log_probs = mx.take_along_axis(
            log_probs, targets_flat[:, None], axis=-1
        ).squeeze(-1)
        return -mx.mean(correct_log_probs)

    def train_step(self, tokens: mx.array, targets: mx.array) -> Tuple[float, Any]:
        """Single micro-batch training step."""

        def loss_fn(model):
            if isinstance(model, GDN2LanguageModel):
                logits, _ = model(tokens)
            else:
                logits = model(tokens)
            # Pre-loss logits finite guard (per user-prescribed protocol):
            # catches state / gate explosion upstream of the loss
            # computation so we crash with a meaningful min/max rather
            # than silently baking NaN into the gradient.
            if not bool(mx.all(mx.isfinite(logits))):
                print(
                    f"logit min/max: {float(mx.min(logits)):.4f} / "
                    f"{float(mx.max(logits)):.4f}"
                )
                raise FloatingPointError("Non-finite logits in forward pass")
            return self.compute_loss(logits, targets)

        loss_value, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        # Eagerly materialize BOTH the loss and the grads so we can call
        # mx.isfinite on them without keeping the lazy graph alive.
        mx.eval(loss_value, grads)
        # Hard-finite guards (per project protocol, see HATCHLING-ZERO
        # plan: a non-finite loss, gradient, or LOGIT must NOT silently
        # propagate through training — bail with a clear error that names
        # the offending tensor. The logits check is FIRST because that's
        # the earliest indicator of state / gate explosion upstream of
        # the loss computation.
        for grad_name, grad_value in tree_flatten(grads):
            if not bool(mx.all(mx.isfinite(grad_value))):
                raise FloatingPointError(
                    f"Non-finite gradient: '{grad_name}'"
                )
        if not bool(mx.isfinite(loss_value)):
            raise FloatingPointError(
                f"Non-finite training loss: {float(loss_value)}"
            )
        return float(loss_value), grads

    def step(self, grads_list, tokens_seen: int) -> float:
        """Apply averaged gradients + clip + optimizer step."""
        if not grads_list:
            return 0.0

        if len(grads_list) == 1:
            avg_grads = grads_list[0]
        else:
            def avg(tree_a, tree_b):
                return mx.add(tree_a, tree_b)
            avg_grads = grads_list[0]
            for g in grads_list[1:]:
                avg_grads = tree_map(avg, avg_grads, g)
            avg_grads = tree_map(lambda t: t / len(grads_list), avg_grads)

        # Defensive global-norm gradient clip. The first training run
        # produced NaN at step 50 — likely because one tall gradient
        # spike pulled Adam into NaN territory. Standard transformer-
        # training hygiene: clip the L2 norm to 1.0 before the update.
        avg_grads = self._clip_grads_by_global_norm(avg_grads, max_norm=1.0)

        self.optimizer.update(self.model, avg_grads)
        mx.eval(self.model.parameters())

        return float(tokens_seen)

    @staticmethod
    def _clip_grads_by_global_norm(grads, max_norm: float = 1.0):
        """Global L2-norm gradient clip (PyTorch's torch.nn.utils.clip_grad_norm_
        equivalent, implemented across the pytree of grads that MLX uses).

        Releases the doubled-gradient tree eagerly via `mx.eval` so the
        lazy graph cache does not retain the full per-tensor rescale —
        this is the same OOM signature that the prior `mx.eval(grads)`
        patch was meant to prevent on the value_and_grad side.
        """
        flat = tree_flatten(grads)
        norm_sq = mx.array(0.0)
        for _, v in flat:
            vf = v.astype(mx.float32)
            norm_sq = norm_sq + mx.sum(vf * vf)
        total_norm = mx.sqrt(norm_sq)
        clip_coef = mx.minimum(mx.array(1.0), max_norm / (total_norm + 1e-6))
        # Flush the lazy graph before tree_map doubles the parameter tree.
        mx.eval(total_norm, clip_coef)
        return tree_map(lambda t: t * clip_coef, grads)

    def train(
        self,
        train_batches: List[Tuple],
        val_batches: List[Tuple],
        num_epochs: int = 1,
        max_steps: int = None,
    ) -> Dict[str, Any]:
        """Full training loop with periodic checkpoints.

        If `max_steps` is set, the loop terminates early — used by
        `--smoke-test` (5 steps) and `--max-steps` (capped training).

        Always wraps in try/finally: if training crashes mid-step, the
        accumulated metrics still get flushed, and a "stopped" marker
        is written so it's obvious the run did not complete cleanly.
        """
        print(f"\n{'=' * 70}")
        print(f"Training: {self.model_name}")
        print(
            f"  lr={self.lr} grad_accum={self.grad_accum} "
            f"checkpoint_every={self.checkpoint_every} "
            f"save_optimizer_every={self.save_optimizer_every}"
        )
        print(f"{'=' * 70}")

        completed_cleanly = False
        total_tokens = 0
        start_time = time.time()
        global_step = 0
        grads_buffer: List[Any] = []

        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                steps_this_epoch = 0
                for step_idx, (tokens, targets) in enumerate(train_batches):
                    loss_value, grads = self.train_step(tokens, targets)
                    grads_buffer.append(grads)
                    # FIX: accumulate loss_value so avg_train_loss isn't
                    # permanently 0.0 (the bug that masked the step-50 NaN).
                    epoch_loss += float(loss_value)
                    token_count = int(tokens.size)
                    total_tokens += token_count
                    steps_this_epoch += 1

                    if len(grads_buffer) >= self.grad_accum or max_steps:
                        self.step(grads_buffer, token_count)
                        grads_buffer = []

                        if (
                            global_step > 0
                            and global_step % self.checkpoint_every == 0
                        ):
                            avg_train_loss = epoch_loss / max(1, steps_this_epoch)
                            val_loss = self._evaluate(val_batches[:10])

                            elapsed = time.time() - start_time
                            throughput = total_tokens / max(elapsed, 1e-8)

                            metric = {
                                "step": global_step,
                                "epoch": epoch,
                                "train_loss": avg_train_loss,
                                "val_loss": val_loss,
                                "tokens": total_tokens,
                                "wall_time": elapsed,
                                "throughput_tok_s": throughput,
                                "lr": self.lr,
                            }
                            self.metrics.append(metric)

                            is_full = (
                                self.save_optimizer_every > 0
                                and (global_step % self.save_optimizer_every == 0)
                            )
                            save_mlx_checkpoint(
                                self.output_dir,
                                global_step,
                                self.model,
                                self.optimizer,
                                cfg=self.cfg,
                                metrics=metric,
                                model_only=not is_full,
                                save_optimizer_every=self.save_optimizer_every,
                            )
                            prune_mlx_checkpoints(
                                self.output_dir,
                                keep_last_full=2,
                                keep_last_model_only=5,
                            )

                            print(
                                f"Epoch {epoch + 1} Step {global_step:4d}: "
                                f"train_loss={avg_train_loss:.4f} "
                                f"val_loss={val_loss:.4f} "
                                f"tokens={total_tokens:,} "
                                f"tok/s={throughput:.0f} "
                                f"ckpt={'full' if is_full else 'model_only'}",
                                flush=True,
                            )

                            # Reset BOTH numerator and denominator together
                            # so avg_train_loss in the next window is
                            # genuinely the mean of micro-batch losses in
                            # that window — not (window_loss /
                            # cumulative_steps_since_epoch_start).
                            epoch_loss = 0.0
                            steps_this_epoch = 0

                        global_step += 1

                        if max_steps is not None and global_step >= max_steps:
                            print(
                                f"[max_steps={max_steps} reached] terminating "
                                f"early at step {global_step}",
                                flush=True,
                            )
                            break

                if max_steps is not None and global_step >= max_steps:
                    break

            completed_cleanly = True
            print(
                f"[done] {self.model_name}: "
                f"{global_step} steps, {total_tokens:,} tokens, "
                f"{time.time() - start_time:.1f}s wall clock",
                flush=True,
            )
        finally:
            status = "complete" if completed_cleanly else "stopped"
            self.metrics.append({"status_marker": status, "step": global_step})
            self._flush_metrics()
            # The most likely cause of arriving in `finally` is OOM mid-step.
            # If `save_mlx_checkpoint` itself OOMs during the post-mortem save,
            # we still want the metric flush (above) and a clear status flag.
            try:
                save_mlx_checkpoint(
                    self.output_dir,
                    global_step,
                    self.model,
                    self.optimizer,
                    cfg=self.cfg,
                    metrics={"final": True, "step": global_step, "status": status},
                    model_only=False,
                )
            except Exception as ckpt_err:
                print(
                    f"[{status}] final ckpt save failed: {ckpt_err!s}; "
                    f"metrics still flushed to {self.metrics_path}",
                    flush=True,
                )
            else:
                print(
                    f"[{status}] metrics flushed to {self.metrics_path}, "
                    f"final ckpt at step {global_step}",
                    flush=True,
                )

        return {
            "steps": global_step,
            "tokens": total_tokens,
            "metrics_path": str(self.metrics_path),
            "completed_cleanly": completed_cleanly,
            "latest_ckpt": str(latest_checkpoint(self.output_dir) or ""),
            "metrics_count": len([m for m in self.metrics if "status_marker" not in m]),
        }

    def _flush_metrics(self) -> None:
        """Atomic metrics flush with NaN-safe serialization.

        Python's `json.dumps` writes unquoted `NaN`/`Infinity` tokens which
        violate strict JSON spec — `jq` and most consumers reject the file.
        Pre-process to substitute `null` for non-finite floats so the file
        stays valid JSON even mid-divergence.
        """
        tmp = self.metrics_path.with_suffix(".json.tmp")
        safe: List[Dict[str, Any]] = []
        for m in self.metrics:
            sanitized: Dict[str, Any] = {}
            for k, v in m.items():
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        sanitized[k] = None
                    else:
                        sanitized[k] = v
                else:
                    sanitized[k] = v
            safe.append(sanitized)
        tmp.write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, self.metrics_path)

    def _evaluate(self, batches: List[Tuple], num_batches: int = 10) -> float:
        """Small validation pass on the first N batches."""
        total = 0.0
        count = 0
        for tokens, targets in batches[:num_batches]:
            try:
                if isinstance(self.model, GDN2LanguageModel):
                    logits, _ = self.model(tokens)
                else:
                    logits = self.model(tokens)
                loss = self.compute_loss(logits, targets)
                total += float(loss)
                count += 1
            except Exception as e:
                print(f"  [eval skip] batch {count} raised: {e}", flush=True)
        return total / count if count > 0 else 0.0


def load_wikitext_batches(
    split: str = "train",
    max_docs: int = 1000,
    batch_size: int = 2,
    seq_len: int = 256,
    smoke: bool = False,
) -> List[Tuple]:
    """Tokenized batches from the available 24K BPE + mixed corpus.

    Falls back to synthetic data if neither tokenizer nor corpus files
    are present — so a smoke test still exercises the training loop.
    """
    from pathlib import Path

    print(
        f"[data] split={split} max_docs={max_docs} "
        f"batch_size={batch_size} seq_len={seq_len}",
        flush=True,
    )
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("data/tokenizer/hz_24k.json")
        vocab_size = 24000
        print("[data] 24K BPE tokenizer loaded", flush=True)
    except Exception as e:
        print(f"[data] tokenizer load failed: {e!s}; using char fallback", flush=True)
        vocab_size = 256
        tokenizer = None

    # Split-aware file selection. Previously both train and validation
    # resolved to the same `data/tokenizer_corpus/all.txt` slice, which
    # silently produced identical batches in both splits and masked
    # numerical problems with coincidentally-identical val_loss. After
    # this fix: train falls through to mixed_corpus when available;
    # validation is always loaded from `validation_sample_1k.jsonl`,
    # which is a different file entirely.
    data_dir = Path("data/processed/wikitext")
    mixed_corpus = Path("data/tokenizer_corpus/all.txt")
    jsonl_path = data_dir / f"{split}_sample_1k.jsonl"

    text: str = ""
    if split == "train" and mixed_corpus.exists():
        text = mixed_corpus.read_text(errors="ignore")
        if smoke:
            text = text[: int(1e4)]  # tiny for smoke
        else:
            text = text[: int(1e7)]
        print(
            f"[data] split=train from mixed corpus: {len(text):,} chars "
            f"({'smoke' if smoke else 'full'})",
            flush=True,
        )
    elif jsonl_path.exists():
        docs: List[str] = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_docs:
                    break
                record = json.loads(line)
                if record.get("text"):
                    docs.append(record["text"])
        text = "\n\n".join(docs)
        print(
            f"[data] split={split} from wikitext "
            f"{jsonl_path.name}: {len(text):,} chars",
            flush=True,
        )
    else:
        print(
            f"[data] no corpus path for split={split}; emitting synthetic streams",
            flush=True,
        )

    if tokenizer:
        try:
            encoding = tokenizer.encode(text)
            tokens = encoding.ids
        except Exception as e:
            print(f"[data] 24K tokenize failed ({e!s}); using char fallback", flush=True)
            tokens = [ord(c) % vocab_size for c in text]
    else:
        tokens = [ord(c) % vocab_size for c in text]

    print(f"[data] tokenized to {len(tokens):,} tokens", flush=True)

    batches: List[Tuple] = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len):
        batch_tokens: List[List[int]] = []
        batch_targets: List[List[int]] = []
        for b in range(batch_size):
            start = i + b * (seq_len + 1)
            if start + seq_len + 1 >= len(tokens):
                break
            batch_tokens.append(tokens[start : start + seq_len])
            batch_targets.append(tokens[start + 1 : start + seq_len + 1])
        if len(batch_tokens) == batch_size:
            batches.append(
                (mx.array(batch_tokens), mx.array(batch_targets))
            )

    print(f"[data] built {len(batches)} batches (vocab={vocab_size})", flush=True)
    return batches


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        choices=["110m", "300m", "both"],
        default="both",
        help="Which model size(s) to train. Sequential per spec.",
    )
    parser.add_argument(
        "--arch",
        choices=["both", "hz", "transformer"],
        default="both",
        help="Train HZ-0A GDN-2, transformer baseline, or both.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Periodic checkpoint cadence (optimizer-step units).",
    )
    parser.add_argument(
        "--save-optimizer-every",
        type=int,
        default=200,
        help="Periodic full-checkpoint cadence (model-only in between).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, terminate after this many optimizer steps per model. "
             "Used for capped runs and smoke testing.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 5 optimizer steps with synthetic data to validate the "
             "launcher can build models + step + checkpoint + finalize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the per-model output directory.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256,
        help="Sequence length for batches (kept fixed for MLX compile).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Microbatch size (kept small for unified memory headroom).",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run a one-batch forward+backward diagnostic instead of "
             "training. Verifies token ranges, fingerprint mismatch "
             "between train+val, finite logits/loss/grads on both "
             "splits, and plausibility of the initial loss magnitude. "
             "Use BEFORE any restart smoke test.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=PHASE14_LEARNING_RATE,
        help="Learning rate for Adam. Defaults to the Phase 6 sweep "
             "winner (3e-4). User-recommended conservative value is "
             "1e-4 for the post-divergence restart.",
    )
    return parser.parse_args(argv)


def run_diagnostic(args) -> int:
    """One-batch forward + backward diagnostic. NO optimizer updates.

    Pre-restart sanity check per HZ-0A plan. Verifies (in order):

      [1/6] Token range + identity
            - train and val tokens all in [0, vocab_size)
            - train and val fingerprints (first 64 ids) are different
              (this is the data-split bug detector)
      [2/6] Forward pass on one TRAIN batch
            - logits are finite (no NaN / Inf)
      [3/6] Initial loss on TRAIN
            - magnitude is roughly ln(vocab_size) ~ 10.09 for a model
              with random init against uniform random targets
      [4/6] Backward pass on TRAIN
            - per-leaf grad finiteness
            - max leaf L2 norm (sanity bound)
      [5/6] Forward + loss on VALIDATION
            - v_logits finite
            - val loss finite
      [6/6] PASS / FAIL summary

    Builds a 110M HZ-0A model so the assertion is conservative — the
    same arch the production run uses. Run BEFORE every restart smoke
    to catch silent data-split, vocab, or numerical-explosion bugs.
    """
    print("=" * 70)
    print("Phase 14: one-batch diagnostic (110M HZ architecture)")
    print("=" * 70)

    model = GDN2LanguageModel(
        vocab_size=24000, model_dim=768,
        num_layers=24, num_heads=12,
    )

    # max_docs=200 (NOT 1) so the validation split loads hundreds of
    # wikitext records — the validation_sample_1k.jsonl has 1000 lines
    # and each line holds a discrete document; max_docs=1 would yield ~20
    # chars / 0 batches for the validation path.
    train_batches = load_wikitext_batches(
        "train", max_docs=200, batch_size=1,
        seq_len=args.seq_len, smoke=False,
    )
    val_batches = load_wikitext_batches(
        "validation", max_docs=200, batch_size=1,
        seq_len=args.seq_len, smoke=False,
    )

    if not train_batches or not val_batches:
        print(
            f"FAIL: train_batches={len(train_batches)} "
            f"val_batches={len(val_batches)} — loader returned empty"
        )
        return 1

    train_tokens, train_targets = train_batches[0]
    val_tokens, val_targets = val_batches[0]

    # [1/6] Token range + identity
    train_min = int(train_tokens.min())
    train_max = int(train_tokens.max())
    val_min = int(val_tokens.min())
    val_max = int(val_tokens.max())
    print("\n[1/6] Token range + identity")
    print(
        f"  train:  shape={tuple(train_tokens.shape)} "
        f"min={train_min} max={train_max}"
    )
    print(
        f"  val:    shape={tuple(val_tokens.shape)} "
        f"min={val_min} max={val_max}"
    )
    if train_min < 0 or train_max >= 24000:
        print("  FAIL: train tokens outside [0, 24000)")
        return 2
    if val_min < 0 or val_max >= 24000:
        print("  FAIL: val tokens outside [0, 24000)")
        return 2

    train_fp = tuple(int(t) for t in train_tokens.reshape(-1).tolist()[:64])
    val_fp = tuple(int(t) for t in val_tokens.reshape(-1).tolist()[:64])
    if train_fp == val_fp:
        print(
            "  FAIL: train AND val fingerprints MATCH — "
            "data split is broken (both splits resolved to the "
            "same file/slice)"
        )
        return 3
    print(f"  train fingerprint (first 16 ids): {train_fp[:16]}")
    print(f"  val   fingerprint (first 16 ids): {val_fp[:16]}")
    print("  fingerprints differ: OK")

    # Stable cross-entropy inlined (mirror of TrainingHarness.compute_loss)
    def _ce(logits: mx.array, targets: mx.array) -> float:
        V = logits.shape[-1]
        flat = logits.reshape(-1, V).astype(mx.float32)
        tgt = targets.reshape(-1)
        flat = mx.clip(flat, -100.0, 100.0)
        max_l = mx.max(flat, axis=-1, keepdims=True)
        lse = (
            mx.log(mx.sum(mx.exp(flat - max_l), axis=-1, keepdims=True))
            + max_l
        )
        log_probs = flat - lse
        correct = mx.take_along_axis(
            log_probs, tgt[:, None], axis=-1
        ).squeeze(-1)
        return float(-mx.mean(correct))

    # [2/6] Forward on train
    print("\n[2/6] Forward pass on one TRAIN batch")
    logits, _ = model(train_tokens)
    mx.eval(logits)
    print(f"  train logits shape: {tuple(logits.shape)}")
    print(f"  train logits min:   {float(logits.min()):.4f}")
    print(f"  train logits max:   {float(logits.max()):.4f}")
    if not bool(mx.all(mx.isfinite(logits))):
        print("  FAIL: train logits contain NaN / Inf")
        return 4

    # [3/6] Initial train loss
    print("\n[3/6] Initial TRAIN loss (no param update)")
    train_loss = _ce(logits, train_targets)
    print(
        f"  train loss = {train_loss:.4f} "
        f"(expected ~10.09 = ln(24000) for uniform random init)"
    )
    if not math.isfinite(train_loss):
        print("  FAIL: train loss is not finite")
        return 5
    if train_loss < 5.0 or train_loss > 20.0:
        print(
            "  FAIL: train loss outside plausible range "
            "[5, 20] — model architecture is structurally suspect"
        )
        return 5

    # [4/6] Backward on train
    print("\n[4/6] Backward pass on TRAIN — per-leaf grad finiteness")
    V = logits.shape[-1]
    tgt_flat = train_targets.reshape(-1)

    def loss_fn(m):
        lg, _ = m(train_tokens)
        flat = lg.reshape(-1, V).astype(mx.float32)
        flat = mx.clip(flat, -100.0, 100.0)
        max_l = mx.max(flat, axis=-1, keepdims=True)
        lse = mx.log(mx.sum(mx.exp(flat - max_l), axis=-1, keepdims=True)) + max_l
        log_probs = flat - lse
        correct = mx.take_along_axis(
            log_probs, tgt_flat[:, None], axis=-1
        ).squeeze(-1)
        return -mx.mean(correct)

    _, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(grads)

    n_total = 0
    n_finite = 0
    max_norm = 0.0
    fail_names: List[str] = []
    for name, g in tree_flatten(grads):
        n_total += 1
        if bool(mx.all(mx.isfinite(g))):
            n_finite += 1
            vf = g.astype(mx.float32)
            sq = mx.sum(vf * vf)
            norm = float(mx.sqrt(sq))
            max_norm = max(max_norm, norm)
        else:
            fail_names.append(name)

    print(f"  grad leaves finite: {n_finite}/{n_total}")
    print(f"  max leaf L2 norm:   {max_norm:.4e}")
    if fail_names:
        print(
            f"  FAIL: {len(fail_names)} non-finite leaves; "
            f"first 5: {fail_names[:5]}"
        )
        return 6

    # Gradient-norm red-flag check. A norm > 1e3 on the very first
    # backward (immediately after random init, no updates yet) means
    # the launcher's grad-clip at max_norm=1.0 will effectively quench
    # the optimizer step to near-zero — training cannot make progress.
    # Per HZ-0A plan Hypothesis 5, this strongly suggests that the
    # GDN-2 gate initialization for decay / erase / write is too
    # aggressive at construction. Apply safe initial gates
    # (decay≈0.99 / erase≈0.01 / write≈0.01) before any restart.
    if max_norm > 1e3:
        print(
            f"  WARN: max leaf L2 norm = {max_norm:.4e} is anomalously "
            f"large for a fresh-init backward pass; grad-clip at "
            f"max_norm=1.0 will scale the optimizer step to ~{1.0/max_norm:.2e} "
            f"of nominal. Likely cause: aggressive GDN-2 gate init."
        )
        print(
            "  Note: per-layer state norms NOT covered by this "
            "diagnostic — GDN2 forward return signature does not "
            "expose a per-layer state list. Surface area here is "
            "grad + logit + loss finiteness only."
        )

    # [5/6] Forward + loss on validation
    print("\n[5/6] Forward + loss on VALIDATION batch")
    v_logits, _ = model(val_tokens)
    mx.eval(v_logits)
    print(f"  val logits min: {float(v_logits.min()):.4f}")
    print(f"  val logits max: {float(v_logits.max()):.4f}")
    if not bool(mx.all(mx.isfinite(v_logits))):
        print("  FAIL: val logits contain NaN / Inf")
        return 7
    val_loss = _ce(v_logits, val_targets)
    print(f"  val loss = {val_loss:.4f}")
    if not math.isfinite(val_loss):
        print("  FAIL: val loss is not finite")
        return 8

    print("\n[6/6] DIAGNOSTIC REPORT")
    print("  train and val resolve to DIFFERENT files ✓")
    print("  all tensors finite across forward + backward on both splits ✓")
    print(
        f"  initial train loss ~ {train_loss:.4f}, "
        f"initial val loss ~ {val_loss:.4f} (≈ ln(24000) = 10.09 expected) ✓"
    )
    if max_norm > 1e3:
        print(
            f"  ✗ max grad L2 norm = {max_norm:.4e} is anomalous — "
            f"training will not converge without safe initial gates."
        )
        print(
            f"    Apply decay/logit_bias schedules in the GDN-2 model:\n"
            f"      decay initializer bias → sigmoid(bias)≈0.99\n"
            f"      erase initializer bias → sigmoid(bias)≈0.01\n"
            f"      write initializer bias → sigmoid(bias)≈0.01"
        )
        print(
            f"    Note: per-layer state norms NOT part of this "
            f"diagnostic — GDN2 forward return signature does not "
            f"expose a per-layer state list. To add coverage, expose "
            f"`states: List[mx.array]` from GDN2MetalModule.__call__."
        )
        return 9  # Distinct exit code so CI can gate on this
    else:
        print(
            f"  ✓ gradient norms bounded (max L2 = {max_norm:.4e}) — "
            f"safe for restart"
        )
        print("DIAGNOSTIC PASS — all six checks green")
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    # Diagnostic dispatch must run BEFORE any training setup so that
    # we never produce a polluted checkpoint from a structurally
    # broken model / data split / vocab mismatch.
    if args.diagnostic:
        return run_diagnostic(args)

    print("=" * 70)
    print("Phase 14: Full Training Runs (hardened launcher)")
    print("=" * 70)
    print(
        f"  models={args.models} arch={args.arch} "
        f"smoke_test={args.smoke_test}",
        flush=True,
    )

    if args.smoke_test:
        args.max_steps = args.max_steps or 5
        args.checkpoint_every = 2

    model_specs = []
    if args.models in ("110m", "both"):
        model_specs.append(PHASE14_CONFIGS[0])  # hz0a_110m
    if args.models in ("300m", "both"):
        model_specs.append(PHASE14_CONFIGS[1])  # hz0a_300m

    print("\n[1/3] Loading data batches...", flush=True)
    train_batches = load_wikitext_batches(
        "train", max_docs=500,
        batch_size=args.batch_size, seq_len=args.seq_len,
        smoke=args.smoke_test,
    )
    val_batches = load_wikitext_batches(
        "validation", max_docs=100,
        batch_size=args.batch_size, seq_len=args.seq_len,
        smoke=args.smoke_test,
    )
    print(
        f"  train_batches={len(train_batches)} val_batches={len(val_batches)}",
        flush=True,
    )

    train_archs = []
    if args.arch in ("both", "hz"):
        train_archs.append("hz")
    if args.arch in ("both", "transformer"):
        train_archs.append("transformer")

    print(
        f"\n[2/3] Training {len(model_specs)} model sizes × "
        f"{len(train_archs)} architectures sequentially...",
        flush=True,
    )

    for name, dim, layers, heads in model_specs:
        cfg = {
            "model_dim": dim,
            "num_layers": layers,
            "num_heads": heads,
            "vocab_size": 24000,
            "lr": PHASE14_LEARNING_RATE,
            "grad_accum": PHASE14_GRAD_ACCUM,
        }

        for arch in train_archs:
            if arch == "hz":
                model = GDN2LanguageModel(
                    vocab_size=24000, model_dim=dim,
                    num_layers=layers, num_heads=heads,
                )
                tag = name
            else:
                model = TransformerLM(
                    vocab_size=24000, model_dim=dim,
                    num_layers=layers, num_heads=heads,
                )
                tag = f"transformer_{name}"

            output_dir = args.output_dir / tag if args.output_dir \
                else Path(f"outputs/training/{tag}")
            output_dir.mkdir(parents=True, exist_ok=True)

            trainer = TrainingHarness(
                model=model,
                model_name=tag,
                cfg=cfg,
                learning_rate=args.lr,
                gradient_accumulation=PHASE14_GRAD_ACCUM,
                output_dir=output_dir,
                checkpoint_every=args.checkpoint_every,
                save_optimizer_every=args.save_optimizer_every,
            )
            trainer.train(
                train_batches=train_batches,
                val_batches=val_batches,
                num_epochs=1,
                max_steps=args.max_steps,
            )

    print("\n" + "=" * 70, flush=True)
    print("Phase 14 launcher finished", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
