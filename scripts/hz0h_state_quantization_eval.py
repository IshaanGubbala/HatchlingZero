"""HZ Phase 3 (plans/HatchlingZero_Reality_Plan.md, section 6.4,
"Quantized State"): real quality check for INT8 synaptic-state
quantization (reference/hz0h_bdh_torch.py's
bdh_stream_chunk_int8_state/quantize_state_int8), reusing H5's own
real, already-established passkey-retrieval methodology
(reference/hz0h_bdh_h5_memory_tasks.py) rather than a synthetic/random-
weight check -- a trained model's real retrieval accuracy is what
matters for the Phase 3 exit gate ("<2-3% task/quality degradation").
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence, train_bdh_passkey_model
from reference.hz0h_bdh_torch import (
    bdh_stream_chunk,
    bdh_stream_chunk_int8_state,
    init_bdh_states,
    init_bdh_states_int8,
)


@torch.no_grad()
def evaluate_passkey_fp32_vs_int8_state(model, *, vocab_size: int, prefix_len: int, filler_len: int, passkey_range: int, num_examples: int = 200, seed: int = 3000, token_by_token: bool = True) -> dict:
    """Real accuracy comparison, fp32 streaming state vs INT8-quantized
    streaming state, on the SAME trained model and SAME examples.
    `token_by_token=True` streams one token at a time (the realistic
    decode scenario, where INT8's compounding quantization error can
    actually accumulate across many steps) rather than one big chunk
    (which -- see the module's own dev-time check -- shows near-zero
    error because there's no repeated quantize/dequantize round-trip
    within a single chunk call)."""
    rng = np.random.default_rng(seed)
    fp32_correct = 0
    int8_correct = 0
    agree_with_each_other = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, passkey_range=passkey_range)
        idx = torch.tensor([seq], dtype=torch.long)

        fp32_states = init_bdh_states(model, batch_size=1)
        int8_states = init_bdh_states_int8(model, batch_size=1)
        if token_by_token:
            for t in range(idx.shape[1] - 1):
                tok = idx[:, t:t + 1]
                fp32_states, _ = bdh_stream_chunk(model, fp32_states, tok, start_position=t)
                int8_states, _ = bdh_stream_chunk_int8_state(model, int8_states, tok, start_position=t)
            query_position = idx.shape[1] - 1
            query_idx = idx[:, -1:]
        else:
            prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]
            fp32_states, _ = bdh_stream_chunk(model, fp32_states, prefix_idx, start_position=0)
            int8_states, _ = bdh_stream_chunk_int8_state(model, int8_states, prefix_idx, start_position=0)
            query_position = prefix_idx.shape[1]

        _fp32_states, fp32_logits = bdh_stream_chunk(model, fp32_states, query_idx, start_position=query_position)
        _int8_states, int8_logits = bdh_stream_chunk_int8_state(model, int8_states, query_idx, start_position=query_position)

        fp32_pred = int(fp32_logits[0, -1].argmax())
        int8_pred = int(int8_logits[0, -1].argmax())
        fp32_correct += int(fp32_pred == answer)
        int8_correct += int(int8_pred == answer)
        agree_with_each_other += int(fp32_pred == int8_pred)

    return {
        "fp32_state_accuracy": fp32_correct / num_examples,
        "int8_state_accuracy": int8_correct / num_examples,
        "accuracy_degradation": (fp32_correct - int8_correct) / num_examples,
        "fp32_int8_prediction_agreement_rate": agree_with_each_other / num_examples,
        "num_examples": num_examples,
        "token_by_token": token_by_token,
    }


def main() -> None:
    print("Training a real passkey-retrieval model (same task/config family as H5's own evidence)...")
    model = train_bdh_passkey_model(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, prefix_len=4, filler_len=16, passkey_range=8, steps=400, batch_size=16, seed=0)

    result = evaluate_passkey_fp32_vs_int8_state(model, vocab_size=32, prefix_len=4, filler_len=16, passkey_range=8, num_examples=200, token_by_token=True)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
