"""HZ-0H T1/T2: ternary (1-bit-class, BitNet b1.58) training for the
matched-Transformer control (`reference/hz0a_matched_transformer.py`).

Per docs/restart/hz0h_ternary_training_design.md's T0 contract, this
architecture was always in scope ("Same _make_linear-style swap as the
hybrid model") but had no actual implementation or evidence until now --
unlike HZ-0A hybrid (pre-existing `--bitnet`) and BDH-GPU (this session's
earlier T1/T2 work in test_hz0h_bdh_ternary.py), the matched Transformer's
`qkv`/`attn_out`/`gate`/`up`/`down` were plain `nn.Linear` with no ternary
path at all. Wired via `_make_linear` (reused directly from
reference/hz0a_torch_model.py, not duplicated) behind
`MatchedTransformerConfig(use_bitlinear=True)`.

Mirrors test_hz0h_bdh_ternary.py's structure: T1 stability bar (a short
real run must learn, no NaN/Inf), then T2 (matched FP32-vs-ternary
convergence-gap check). The quantization primitive itself
(`_ste_round_clip`/`BitLinear`) is already unit-tested where it's defined
(reference/hz0a_torch_model.py) -- not re-tested here, only its
integration into this specific architecture.
"""
from __future__ import annotations

import math

import torch

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


def _config(use_bitlinear: bool, vocab_size: int = 48, d_model: int = 48, num_layers: int = 3, num_heads: int = 4, d_ff: int = 96) -> MatchedTransformerConfig:
    return MatchedTransformerConfig({
        "vocab_size": vocab_size, "d_model": d_model, "num_layers": num_layers,
        "num_heads": num_heads, "head_dim": d_model // num_heads, "d_ff": d_ff,
        "use_bitlinear": use_bitlinear,
    })


def test_ternary_matched_transformer_uses_bitlinear_and_default_does_not():
    fp_model = MatchedTransformerLM(_config(use_bitlinear=False))
    ternary_model = MatchedTransformerLM(_config(use_bitlinear=True))
    from reference.hz0a_torch_model import BitLinear
    assert not isinstance(fp_model.blocks[0].qkv, BitLinear)
    assert isinstance(ternary_model.blocks[0].qkv, BitLinear)
    assert isinstance(ternary_model.blocks[0].attn_out, BitLinear)
    assert isinstance(ternary_model.blocks[0].gate, BitLinear)
    assert isinstance(ternary_model.blocks[0].up, BitLinear)
    assert isinstance(ternary_model.blocks[0].down, BitLinear)
    # embedding/norms must stay full precision either way, per the T0 contract
    assert isinstance(ternary_model.embedding, torch.nn.Embedding)
    assert isinstance(ternary_model.final_norm.weight, torch.nn.Parameter)


def test_ternary_matched_transformer_parameter_count_unchanged():
    """BitLinear has the same weight/bias shapes as nn.Linear (only the
    forward computation differs) -- ternary must not change the model's
    parameter-matched-ness, the entire point of this architecture's
    existence in the HZ-0A comparison protocol."""
    fp_model = MatchedTransformerLM(_config(use_bitlinear=False))
    ternary_model = MatchedTransformerLM(_config(use_bitlinear=True))
    fp_count = sum(p.numel() for p in fp_model.parameters())
    ternary_count = sum(p.numel() for p in ternary_model.parameters())
    assert fp_count == ternary_count


def test_ternary_forward_finite_and_differs_from_full_precision():
    torch.manual_seed(1)
    fp_model = MatchedTransformerLM(_config(use_bitlinear=False))
    ternary_model = MatchedTransformerLM(_config(use_bitlinear=True))
    ternary_model.load_state_dict(fp_model.state_dict())

    idx = torch.randint(0, 48, (2, 16))
    fp_model.eval()
    ternary_model.eval()
    with torch.no_grad():
        logits_fp = fp_model(idx)
        logits_ternary = ternary_model(idx)
    assert torch.isfinite(logits_ternary).all()
    assert not torch.allclose(logits_fp, logits_ternary, atol=1e-4)


def test_t1_stability_ternary_matched_transformer_learns_below_random_floor():
    """T1's bar for this architecture: a short real training run with
    use_bitlinear=True must learn (loss well below ln(vocab_size)), no
    NaN/Inf at any step."""
    torch.manual_seed(2)
    vocab_size = 40
    model = MatchedTransformerLM(_config(use_bitlinear=True, vocab_size=vocab_size, d_model=40, num_layers=2, num_heads=4, d_ff=80))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    cycle = torch.randint(0, vocab_size, (8,))
    batch, seq_len = 4, 32
    base = cycle.repeat(seq_len // len(cycle) + 1)[:seq_len]
    tokens = base.unsqueeze(0).repeat(batch, 1)
    idx, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    random_floor = math.log(vocab_size)
    losses = []
    for _ in range(150):
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        assert torch.isfinite(loss), f"non-finite loss during ternary training: {loss}"
        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite gradient during ternary training"
        opt.step()
        losses.append(float(loss))

    final_loss = sum(losses[-10:]) / 10
    assert final_loss < random_floor * 0.5, f"ternary matched Transformer did not learn: final loss {final_loss:.3f} vs random floor {random_floor:.3f}"
    assert losses[-1] < losses[0]


def test_t2_matched_fp32_vs_ternary_convergence_gap_is_small():
    """T2 for this architecture: same seed/init/data/budget, only
    use_bitlinear differs. Gap should stay small, matching the pattern
    already found for BDH-GPU (test_hz0h_bdh_ternary.py) and the HZ-0A
    hybrid model's pre-existing --bitnet evidence."""
    vocab_size = 32
    torch.manual_seed(11)
    fp_model = MatchedTransformerLM(_config(use_bitlinear=False, vocab_size=vocab_size, d_model=32, num_layers=2, num_heads=4, d_ff=64))
    ternary_model = MatchedTransformerLM(_config(use_bitlinear=True, vocab_size=vocab_size, d_model=32, num_layers=2, num_heads=4, d_ff=64))
    ternary_model.load_state_dict(fp_model.state_dict())

    gen = torch.Generator().manual_seed(13)
    transition = torch.randn(vocab_size, vocab_size, generator=gen)
    batch, seq_len = 4, 24
    tokens = torch.randint(0, vocab_size, (batch, 1), generator=gen)
    for _ in range(seq_len):
        probs = torch.softmax(transition[tokens[:, -1]], dim=-1)
        next_tok = torch.multinomial(probs, 1, generator=gen)
        tokens = torch.cat([tokens, next_tok], dim=1)
    idx, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    final_losses = {}
    for name, model in (("fp32", fp_model), ("ternary", ternary_model)):
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(120):
            logits = model(idx)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        final_losses[name] = sum(losses[-10:]) / 10

    random_floor = math.log(vocab_size)
    assert final_losses["fp32"] < random_floor * 0.5, "fp32 control did not learn -- broken test setup"
    assert final_losses["ternary"] < random_floor * 0.5, "ternary did not learn"
    gap = abs(final_losses["ternary"] - final_losses["fp32"])
    assert gap < 0.5, f"convergence gap too large: fp32={final_losses['fp32']:.4f} ternary={final_losses['ternary']:.4f} gap={gap:.4f}"
