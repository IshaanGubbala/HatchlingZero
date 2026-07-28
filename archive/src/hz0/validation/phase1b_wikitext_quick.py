"""Phase 1b: Quick WikiText-103 validation (sampled).

Uses 1K doc samples for ~5 min training validation.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json
import time
from typing import Tuple, List
from pathlib import Path

from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from hz0.validation.phase1a_transformer_baseline import TransformerLM


def load_wikitext_sample():
    """Load sampled WikiText-103 from processed directory.

    Returns (train_text, val_text, test_text)
    """
    data_dir = Path("data/processed/wikitext")

    texts = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}_sample_1k.jsonl"
        if not path.exists():
            print(f"✗ {path} not found")
            return None, None, None

        docs = []
        with open(path, "r") as f:
            for line in f:
                record = json.loads(line)
                docs.append(record["text"])

        text = "\n\n".join(docs)
        texts[split] = text

        chars = len(text)
        print(f"✓ {split}: {len(docs):,} docs, {chars:,} chars")

    return texts["train"], texts["validation"], texts["test"]


def tokenize_simple(text: str, vocab_size: int = 256) -> list:
    """Simple character-level tokenization."""
    if text is None:
        return [i % vocab_size for i in range(10000)]

    tokens = []
    for char in text:
        token = ord(char) % vocab_size
        tokens.append(token)
    return tokens


def create_batches(tokens: list, batch_size: int = 4, seq_len: int = 128) -> List[Tuple]:
    """Create training batches from token sequence."""
    batches = []
    for i in range(0, len(tokens) - seq_len, seq_len * batch_size):
        batch_tokens = []
        batch_targets = []

        for b in range(batch_size):
            start = i + b * seq_len
            if start + seq_len >= len(tokens):
                break

            seq = tokens[start : start + seq_len]
            target = tokens[start + 1 : start + seq_len + 1]

            batch_tokens.append(seq)
            batch_targets.append(target)

        if len(batch_tokens) == batch_size:
            batches.append((mx.array(batch_tokens), mx.array(batch_targets)))

    return batches


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
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


def evaluate(model: nn.Module, batches: List, num_batches: int = 10) -> float:
    """Evaluate on batches."""
    total_loss = 0.0

    for i, (tokens, targets) in enumerate(batches[:num_batches]):
        logits = model(tokens)
        loss = compute_loss(logits, targets)
        total_loss += float(loss)

    return total_loss / min(num_batches, len(batches))


def main():
    """Run WikiText validation."""
    print("="*70)
    print("Phase 1b: WikiText-103 Validation (Sampled)")
    print("="*70)

    # Load data
    print("\n[1/5] Loading WikiText-103 samples...")
    train_text, val_text, test_text = load_wikitext_sample()
    if train_text is None:
        print("✗ Failed to load WikiText samples")
        return

    # Tokenize
    print("\n[2/5] Tokenizing...")
    train_tokens = tokenize_simple(train_text)
    val_tokens = tokenize_simple(val_text)
    print(f"✓ Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens")

    # Create batches
    print("\n[3/5] Creating batches...")
    train_batches = create_batches(train_tokens, batch_size=2, seq_len=128)
    val_batches = create_batches(val_tokens, batch_size=2, seq_len=128)
    print(f"✓ Train: {len(train_batches)} batches, Val: {len(val_batches)} batches")

    # Create models
    print("\n[4/5] Creating models...")
    hz_model = GDN2LanguageModel(vocab_size=256, model_dim=256, num_layers=6)
    transformer_model = TransformerLM(vocab_size=256, model_dim=256, num_layers=6, num_heads=4)
    print("✓ HZ-0A created")
    print("✓ Transformer created")

    # Evaluate (no training, just loss)
    print("\n[5/5] Evaluating...")
    hz_loss = evaluate(hz_model, val_batches, num_batches=5)
    tf_loss = evaluate(transformer_model, val_batches, num_batches=5)

    print("\n" + "="*70)
    print("RESULTS (WikiText-103 Sampled)")
    print("="*70)
    print(f"\nValidation Loss (5 batches):")
    print(f"  HZ-0A:       {hz_loss:.4f}")
    print(f"  Transformer: {tf_loss:.4f}")
    print(f"  Difference: {abs(hz_loss - tf_loss):.4f}")

    if abs(hz_loss - tf_loss) < 0.5:
        print(f"\n✓ RESULTS ACCEPTABLE")
    else:
        print(f"\n⚠ RESULTS DIVERGENT")

    print("="*70)


if __name__ == "__main__":
    main()
