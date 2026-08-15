"""Non-negotiable integrity gates for the active upstream BDH oracle."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class _NormalizeKnownBDHExtensions(ast.NodeTransformer):
    """Remove only the two documented model deltas before AST comparison."""

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == "ternary":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_w"
            and len(node.args) == 1
            and not node.keywords
        ):
            return node.args[0]
        return node


class _NormalizeKnownTrainExtensions(ast.NodeTransformer):
    def visit_Import(self, node):
        # The local requests import is deliberately moved into fetch_data so
        # importing the oracle does not require requests at module import time.
        if any(alias.name == "requests" for alias in node.names):
            return None
        return node


def _definitions(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _normalized_dump(node, normalizer):
    node = normalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(node)
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def test_bdh_oracle_matches_pinned_upstream_definitions():
    upstream = ast.parse((ROOT / "specs/upstream/pathway_bdh.py").read_text())
    current = ast.parse((ROOT / "reference/hz0h_bdh_torch.py").read_text())
    up_defs, cur_defs = _definitions(upstream), _definitions(current)
    for name in ("BDHConfig", "get_freqs", "Attention", "BDH"):
        assert name in up_defs and name in cur_defs
        assert _normalized_dump(up_defs[name], _NormalizeKnownBDHExtensions) == _normalized_dump(
            cur_defs[name], _NormalizeKnownBDHExtensions
        ), f"upstream BDH definition drifted: {name}"


def test_training_oracle_matches_pinned_upstream_core():
    upstream = ast.parse((ROOT / "specs/upstream/pathway_train.py").read_text())
    current = ast.parse((ROOT / "reference/hz0h_bdh_train_torch.py").read_text())
    up_defs, cur_defs = _definitions(upstream), _definitions(current)
    for name in ("fetch_data", "get_batch", "eval"):
        assert name in up_defs and name in cur_defs
        assert _normalized_dump(up_defs[name], _NormalizeKnownTrainExtensions) == _normalized_dump(
            cur_defs[name], _NormalizeKnownTrainExtensions
        ), f"upstream training definition drifted: {name}"


def test_bdh_has_shared_iterative_weights_and_no_per_layer_copies():
    from reference.hz0h_bdh_torch import BDH, BDHConfig

    model = BDH(BDHConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=64, dropout=0.0))
    names = dict(model.named_parameters())
    assert {"encoder", "encoder_v", "decoder", "lm_head", "embed.weight"}.issubset(names)
    assert not any(name.startswith("layers.") or ".layers." in name for name in names)
    assert len([name for name in names if name == "encoder"]) == 1
    assert model.encoder.ndim == 3 and model.encoder_v.ndim == 3 and model.decoder.ndim == 2

    source = (ROOT / "reference/hz0h_bdh_torch.py").read_text()
    assert "for level in range(C.n_layer):" in source
    assert "x_sparse = F.relu(x_latent)" in source
    assert "y_sparse = F.relu(y_latent)" in source
    assert ".tril(diagonal=-1)" in source
    assert "return scores @ V" in source


def test_active_runner_uses_the_oracle_not_archived_models():
    source = (ROOT / "scripts/hz0h_stage2_runner_bdh.py").read_text()
    assert "from reference.hz0h_bdh_torch import BDH" in source
    assert "from reference.hz0h_bdh_train_torch import" in source
    assert "hz0i_" not in source
    assert "hz0a_gdn" not in source
    assert "hz0a_torch_model" not in source


def test_integrity_contract_requires_real_next_token_targets():
    from reference.hz0h_bdh_train_torch import shifted_target_batch

    data = torch.arange(12).view(2, 6)
    x, y = shifted_target_batch(data)
    assert torch.equal(x, data[:, :-1])
    assert torch.equal(y, data[:, 1:])
    assert not torch.equal(x, y)


def test_runtime_forward_and_gradient_match_pinned_upstream_snapshot():
    """Execute the pinned fetched source, not merely a textual comparison."""
    upstream_path = ROOT / "specs/upstream/pathway_bdh.py"
    module_spec = importlib.util.spec_from_file_location("pathway_bdh_snapshot", upstream_path)
    assert module_spec and module_spec.loader
    upstream_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream_module)

    from reference.hz0h_bdh_torch import BDH, BDHConfig

    cfg = BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=32, dropout=0.0)
    upstream_cfg = upstream_module.BDHConfig(
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        mlp_internal_dim_multiplier=cfg.mlp_internal_dim_multiplier,
        vocab_size=cfg.vocab_size, dropout=cfg.dropout,
    )
    torch.manual_seed(17)
    current = BDH(cfg).eval()
    upstream = upstream_module.BDH(upstream_cfg).eval()
    upstream.load_state_dict(current.state_dict(), strict=True)
    tokens = torch.randint(0, cfg.vocab_size, (2, 7))

    current_logits, current_loss = current(tokens, targets=tokens)
    upstream_logits, upstream_loss = upstream(tokens, targets=tokens)
    assert torch.allclose(current_logits, upstream_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(current_loss, upstream_loss, atol=1e-6, rtol=1e-6)

    current.zero_grad(set_to_none=True)
    upstream.zero_grad(set_to_none=True)
    current_loss.backward()
    upstream_loss.backward()
    for name in ("encoder", "encoder_v", "decoder", "embed.weight", "lm_head"):
        assert torch.allclose(dict(current.named_parameters())[name].grad, dict(upstream.named_parameters())[name].grad, atol=1e-6, rtol=1e-6), name


def test_pinned_upstream_snapshot_hashes_are_unchanged():
    manifest = __import__("json").loads((ROOT / "specs/upstream/manifest.json").read_text())
    import hashlib
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((ROOT / "specs/upstream" / name).read_bytes()).hexdigest()
        assert actual == expected, f"pinned upstream snapshot changed: {name}"


def test_inference_benchmark_has_fair_transformer_decode_path():
    source = (ROOT / "scripts/hz0h_inference_benchmark.py").read_text()
    assert "from reference.hz0h_bdh_torch import BDH" in source
    assert "bdh_stream_chunk" in source
    assert "measure_transformer_decode_kv_cache" in source
    assert "new_kv_cache" in source
    assert '"use_rope": True' in source


def test_active_plan_names_current_oracle_and_targets():
    plan = (ROOT / "plans/HatchlingZero_Reality_Plan.md").read_text()
    contract = (ROOT / "specs/hz_bdh_integrity_contract.md").read_text()
    assert "reference/hz0h_bdh_torch.py" in plan
    assert "3.0x" in plan and "30%" in plan
    assert "hand-built approximation" in contract
    assert "matched parameter count" in contract


def test_successor_plan_inherits_integrity_and_claim_gates():
    plan = (ROOT / "plans/HatchlingZero_Next_Phase_Plan.md").read_text()
    assert "specs/hz_bdh_integrity_contract.md" in plan
    assert "reference/hz0h_bdh_torch.py" in plan
    assert "reference/hz0h_bdh_train_torch.py" in plan
    assert "3.0x" in plan
    assert "30%" in plan
    assert "real KV cache" in plan
    assert "must not be mislabeled as exact BDH" in plan


def test_deep_research_plan_keeps_ram_and_speed_gates():
    plan = (ROOT / "plans/Deep Reserach Plan.md").read_text()
    assert "at most **70%" in plan
    assert "1.30x end-to-end inference" in plan
    assert "peak inference RAM <= 0.70" in plan
    assert "decode throughput >= 1.30" in plan
    assert "real KV cache" in plan
    assert "compile-only speedup" in plan
