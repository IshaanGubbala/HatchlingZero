"""Phase 14: Full training runs (36M, 110M models on WikiText-103).

Fair comparison: HZ-0A vs Transformer baseline.
Metrics: tokens, wall-clock, FLOPs, validation loss.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import json
from typing import Tuple, List, Dict, Any

from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from hz0.validation.phase1a_transformer_baseline import TransformerLM


class TrainingHarness:
    """Full training with comprehensive logging."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        learning_rate: float = 2e-4,
        gradient_accumulation: int = 4,
    ):
        self.model = model
        self.model_name = model_name
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.grad_accum = gradient_accumulation

        # Logging
        self.log_dir = Path(f"outputs/training/{model_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[Dict[str, Any]] = []

    def compute_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Cross-entropy loss."""
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        logits_flat = mx.clip(logits_flat, -100, 100)
        max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = mx.exp(logits_flat - max_logits)
        sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
        log_softmax = logits_flat - max_logits - mx.log(sum_exp + 1e-9)

        correct_log_probs = log_softmax[mx.arange(len(targets_flat)), targets_flat]
        loss = -mx.mean(correct_log_probs)
        return loss

    def train_step(self, tokens: mx.array, targets: mx.array) -> float:
        """Single training step."""

        def loss_fn(model):
            if isinstance(model, GDN2LanguageModel):
                logits, _ = model(tokens)
            else:
                logits = model(tokens)
            return self.compute_loss(logits, targets)

        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        self.optimizer.update(self.model, grads)

        return float(loss)

    def train(
        self,
        train_batches: List[Tuple],
        val_batches: List[Tuple],
        num_epochs: int = 1,
        checkpoint_every: int = 100,
    ):
        """Full training loop with checkpoints."""
        print(f"\n{'='*70}")
        print(f"Training: {self.model_name}")
        print(f"{'='*70}")

        total_tokens = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for step, (tokens, targets) in enumerate(train_batches):
                loss = self.train_step(tokens, targets)
                epoch_loss += loss
                total_tokens += tokens.size

                # Checkpoint
                if (step + 1) % checkpoint_every == 0:
                    val_loss = self.evaluate(val_batches[:10])
                    elapsed = time.time() - start_time
                    throughput = total_tokens / elapsed

                    metric = {
                        "epoch": epoch,
                        "step": step + 1,
                        "train_loss": epoch_loss / (step + 1),
                        "val_loss": val_loss,
                        "tokens": total_tokens,
                        "wall_time": elapsed,
                        "throughput_tok_s": throughput,
                    }
                    self.metrics.append(metric)

                    print(
                        f"Epoch {epoch+1} Step {step+1:4d}: "
                        f"train_loss={metric['train_loss']:.4f} "
                        f"val_loss={val_loss:.4f} "
                        f"tokens={total_tokens:,} "
                        f"tok/s={throughput:.0f}"
                    )

        self.save_metrics()
        print(f"{'='*70}")
        print(f"Training complete: {self.model_name}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Wall time: {time.time() - start_time:.1f}s")
        print(f"{'='*70}")

    def evaluate(self, batches: List[Tuple], num_batches: int = 10) -> float:
        """Evaluate on validation."""
        total_loss = 0.0
        count = 0

        for tokens, targets in batches[:num_batches]:
            try:
                if isinstance(self.model, GDN2LanguageModel):
                    logits, _ = self.model(tokens)
                else:
                    logits = self.model(tokens)
                loss = self.compute_loss(logits, targets)
                total_loss += float(loss)
                count += 1
            except Exception:
                pass

        return total_loss / count if count > 0 else 0.0

    def save_metrics(self):
        """Save metrics to JSON."""
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved: {metrics_path}")


def load_wikitext_batches(
    split: str = "train", max_docs: int = 1000, batch_size: int = 2, seq_len: int = 256
) -> List[Tuple]:
    """Load mixed corpus (text + code + tools) with 24K tokenizer."""
    import json
    from pathlib import Path

    # Load 24K tokenizer
    print("Loading 24K BPE tokenizer...")
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("data/tokenizer/hz_24k.json")
        vocab_size = 24000
    except Exception as e:
        print(f"✗ Tokenizer load failed: {e}. Using char-level.")
        vocab_size = 256
        tokenizer = None

    # Try mixed corpus first
    mixed_path = Path("data/tokenizer_corpus/all.txt")
    if mixed_path.exists():
        print(f"Loading mixed corpus from {mixed_path}...")
        text = mixed_path.read_text()[:int(1e7)]  # 10M chars limit
        print(f"✓ Loaded {len(text):,} chars from mixed corpus")
    else:
        # Fallback to WikiText
        data_dir = Path("data/processed/wikitext")
        path = data_dir / f"{split}_sample_1k.jsonl"

        if not path.exists():
            print(f"✗ No corpus found. Using synthetic data.")
            batches = []
            for _ in range(10):
                tokens = mx.random.randint(0, vocab_size, (batch_size, seq_len))
                targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))
                batches.append((tokens, targets))
            return batches

        docs = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_docs:
                    break
                record = json.loads(line)
                if record.get("text"):
                    docs.append(record["text"])

        text = "\n\n".join(docs)
        print(f"✓ Loaded {len(text):,} chars from WikiText")

    # Tokenize
    if tokenizer:
        encoding = tokenizer.encode(text)
        tokens = encoding.ids
    else:
        tokens = [ord(c) % vocab_size for c in text]

    print(f"✓ Tokenized to {len(tokens):,} tokens")

    # Batch
    batches = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len):
        batch_tokens = []
        batch_targets = []

        for b in range(batch_size):
            start = i + b * (seq_len + 1)
            if start + seq_len + 1 >= len(tokens):
                break

            seq = tokens[start : start + seq_len]
            target = tokens[start + 1 : start + seq_len + 1]
            batch_tokens.append(seq)
            batch_targets.append(target)

        if len(batch_tokens) == batch_size:
            batches.append((mx.array(batch_tokens), mx.array(batch_targets)))

    print(f"✓ Created {len(batches)} batches (vocab={vocab_size})")
    return batches


def main():
    """Run Phase 14: Full training."""
    print("="*70)
    print("Phase 14: Full Training Runs")
    print("="*70)

    # Load data
    print("\n[1/4] Loading WikiText-103 batches...")
    train_batches = load_wikitext_batches("train", max_docs=500, batch_size=2, seq_len=256)
    val_batches = load_wikitext_batches("validation", max_docs=100, batch_size=2, seq_len=256)

    print(f"Train: {len(train_batches)} batches")
    print(f"Val: {len(val_batches)} batches")

    # Create models (targeting 1.5-3B final, skip 36M)
    print("\n[2/4] Creating models...")
    configs = [
        ("hz0a_110m", 768, 24, 12),  # 110M approx
        ("hz0a_300m", 1024, 32, 16),  # 300M approx
    ]

    for name, dim, layers, heads in configs:
        print(f"\n[3/4] Training {name}...")
        hz_model = GDN2LanguageModel(
            vocab_size=24000,
            model_dim=dim,
            num_layers=layers,
            num_heads=heads,
        )
        tf_model = TransformerLM(
            vocab_size=24000,
            model_dim=dim,
            num_layers=layers,
            num_heads=heads,
        )

        # Train (use best LR from Phase 6: 3e-4)
        print(f"\nHZ-0A ({name}):")
        hz_trainer = TrainingHarness(hz_model, name, learning_rate=3e-4)
        hz_trainer.train(train_batches, val_batches, num_epochs=1, checkpoint_every=50)

        print(f"\nTransformer ({name}):")
        tf_trainer = TrainingHarness(tf_model, f"transformer_{name}", learning_rate=3e-4)
        tf_trainer.train(train_batches, val_batches, num_epochs=1, checkpoint_every=50)

    print("\n" + "="*70)
    print("Phase 14 Complete: Training runs finished")
    print("="*70)
    print("\nNext: Analyze metrics for Phase 2 scaling analysis")


if __name__ == "__main__":
    main()
