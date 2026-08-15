import torch
from reference.hz0h_bdh_block_gated_torch import BDHBlockGated, BDHBlockGatedConfig
from scripts.hz0h_phase_f_domain_ce import canonical_record_hashes, evaluate


def test_domain_evaluator_supports_learned_gate_direct_split_v_derivative():
    model = BDHBlockGated(BDHBlockGatedConfig(n_layer=1, n_embd=32, n_head=4, mlp_internal_dim_multiplier=2, vocab_size=32, dropout=0.0, block_size=4)).eval()
    result = evaluate(model, [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], batch_size=1, device=torch.device("cpu"), prefill_chunk_length=8, block_gated_active_fraction=0.5)
    assert result["finite"]
    assert result["bdh_prefill_path"] == "learned_gate_direct_split_v"
    assert result["block_gated_active_fraction"] == 0.5


def test_canonical_record_hashes_are_format_independent(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("[1, 2, 3]\n[4,5,6]\n")
    assert len(canonical_record_hashes(path)) == 2
