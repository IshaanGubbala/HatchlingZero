from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hz0.checkpoint import load_checkpoint, save_checkpoint
from hz0.data import build_dataset
from hz0.eval import (
    benchmark_decode_latency,
    evaluate_associative_recall,
    evaluate_copy_retrieval,
    evaluate_multi_anchor_retrieval,
    evaluate_overwrite_retrieval,
    evaluate_protected_memory_retrieval,
    evaluate_recall_by_distance,
)
from hz0.generation import greedy_generate
from hz0.model import HybridLM
from hz0.tokenizer import ByteTokenizer


def build_model() -> HybridLM:
    return HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="fallback",
        attention_every=2,
        max_seq_len=64,
    )


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        output_dir=tmp_path,
        step=3,
        model=model,
        optimizer=optimizer,
        config={"model": {"name": "test"}},
        metrics={"loss": 1.23},
    )
    payload = load_checkpoint(path, torch.device("cpu"))
    assert payload["step"] == 3
    assert "model" in payload
    assert "optimizer" in payload
    assert payload["metrics"]["loss"] == 1.23


def test_checkpoint_roundtrip_preserves_logits(tmp_path: Path) -> None:
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, 256, (2, 8))
    expected = model(inputs)
    path = save_checkpoint(
        output_dir=tmp_path,
        step=1,
        model=model,
        optimizer=optimizer,
        config={"model": {"name": "test"}},
    )

    restored = build_model()
    payload = load_checkpoint(path, torch.device("cpu"))
    restored.load_state_dict(payload["model"])
    actual = restored(inputs)
    torch.testing.assert_close(actual, expected)


def test_generation_and_benchmarks() -> None:
    model = build_model()
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("HZ").unsqueeze(0)
    generated = greedy_generate(model, prompt, max_new_tokens=4, max_seq_len=64)
    assert generated.shape == (1, 6)

    retrieval = evaluate_copy_retrieval(
        model=model,
        device=torch.device("cpu"),
        seq_len=16,
        vocab_size=256,
        num_samples=4,
    )
    speed = benchmark_decode_latency(
        model=model,
        device=torch.device("cpu"),
        prompt_len=16,
        steps=4,
        vocab_size=256,
    )
    multi = evaluate_multi_anchor_retrieval(
        model=model,
        device=torch.device("cpu"),
        seq_len=16,
        vocab_size=256,
        num_samples=4,
    )
    associative = evaluate_associative_recall(
        model=model,
        device=torch.device("cpu"),
        seq_len=16,
        vocab_size=256,
        num_samples=4,
    )
    overwrite = evaluate_overwrite_retrieval(
        model=model,
        device=torch.device("cpu"),
        seq_len=16,
        vocab_size=256,
        num_samples=4,
    )
    protected = evaluate_protected_memory_retrieval(
        model=model,
        device=torch.device("cpu"),
        seq_len=16,
        vocab_size=256,
        num_samples=4,
    )
    by_distance = evaluate_recall_by_distance(
        model=model,
        device=torch.device("cpu"),
        seq_len=16,
        vocab_size=256,
        num_samples=4,
        distances=[4, 8, 16],
    )
    assert 0.0 <= retrieval["copy_retrieval_accuracy"] <= 1.0
    assert 0.0 <= multi["multi_anchor_retrieval_accuracy"] <= 1.0
    assert 0.0 <= multi["multi_anchor_anchor_set_accuracy"] <= 1.0
    assert 0.0 <= associative["associative_recall_accuracy"] <= 1.0
    assert 0.0 <= overwrite["overwrite_retrieval_accuracy"] <= 1.0
    assert 0.0 <= protected["protected_memory_accuracy"] <= 1.0
    assert all(0.0 <= by_distance[f"recall_distance_{distance}_accuracy"] <= 1.0 for distance in [4, 8, 16])
    assert speed["tokens_per_second"] > 0.0


def test_retrieval_augmented_dataset(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz" * 8, encoding="utf-8")
    dataset = build_dataset(
        path=path,
        seq_len=16,
        vocab_size=256,
        random_length=16,
        packed=True,
        retrieval_mix_probability=1.0,
        retrieval_num_anchors=3,
    )
    sample = dataset[0]
    assert sample.shape == (17,)
    assert sample.dtype == torch.long
