"""HZ Next-Phase Plan Phase C: does INT8 recurrent state preserve real
quality relative to the BF16/FP32 state, on the actual trained VB
checkpoint (not a freshly-initialized model)? Streams real held-out
validation text through `reference/hz0h_bdh_vb_torch.py`'s
`bdh_vb_stream_chunk` (FP32/BF16 state) and `bdh_vb_stream_chunk_int8_state`
(INT8 state) with the SAME checkpoint weights, and compares cross-entropy
loss -- the same `final_full_depth_validation_loss`-style metric used
throughout the VB sweep/curriculum/Phase B2 investigation, so the
INT8 drift number is directly comparable to those.

Real methodological note: `bdh_vb_stream_chunk`'s per-layer loop runs
`config.n_layer` iterations (weight-shared, matching BDH's own
shared-weight recurrence), which equals the curriculum's own final
depth (8) for every checkpoint this plan has trained so far -- no
separate variable-depth handling is needed here, this evaluates at the
same full depth the checkpoint was actually trained to converge at.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
)


def load_bdh_vb(checkpoint_pt: Path, *, n_embd: int, n_layer: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int, d_state: int) -> BDHVB:
    config = BDHVBConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0, d_state=d_state)
    model = BDHVB(config)
    blob = torch.load(str(checkpoint_pt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def load_validation_sequences(path: Path, num_sequences: int, sequence_length: int) -> list[list[int]]:
    sequences = []
    with path.open() as handle:
        for line in handle:
            if len(sequences) >= num_sequences:
                break
            tokens = json.loads(line)
            if len(tokens) < sequence_length:
                continue
            sequences.append(tokens[:sequence_length])
    return sequences


@torch.no_grad()
def evaluate_validation_loss(model: BDHVB, sequences: list[list[int]], *, step_fn, init_fn, sub_batch: int, stream_chunk_length: int) -> float:
    """Streams each sequence in multiple `stream_chunk_length`-token
    pieces, carrying `states` forward between calls -- calling step_fn
    once with the whole sequence (the original, buggy version of this
    function) leaves the incoming prefix_state at its all-zero init for
    every token, so the "cross" (state-read) term is exactly zero in
    BOTH the FP32 and INT8 code paths regardless of quantization,
    making any INT8 drift measured that way spurious by construction
    -- the state-carrying mechanism this eval exists to test is only
    actually exercised once state accumulated from an earlier chunk is
    read back by a later one."""
    total, count = 0.0, 0
    for start in range(0, len(sequences), sub_batch):
        chunk = torch.tensor(sequences[start:start + sub_batch], dtype=torch.long)
        x, y = shifted_target_batch(chunk)
        states = init_fn(model, x.shape[0])
        T = x.shape[1]
        logits_pieces = []
        for chunk_start in range(0, T, stream_chunk_length):
            x_piece = x[:, chunk_start:chunk_start + stream_chunk_length]
            states, logits_piece = step_fn(model, states, x_piece, start_position=chunk_start)
            logits_pieces.append(logits_piece)
        logits = torch.cat(logits_pieces, dim=1)  # (B, L, vocab) -- L (sequence) is dim 1, not 2
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += float(loss)
        count += 1
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--sub-batch", type=int, default=8)
    parser.add_argument("--stream-chunk-length", type=int, default=32, help="sequence is streamed in pieces this long, carrying (and, for the INT8 arm, re-quantizing) state between pieces -- must be < --sequence-length or the state-read/quantization path is never actually exercised (a real bug caught in this script's first version: a single whole-sequence call leaves the incoming state at its all-zero init the entire time, making any INT8 drift measured that way spurious by construction)")
    args = parser.parse_args()

    if args.stream_chunk_length >= args.sequence_length:
        raise ValueError("--stream-chunk-length must be < --sequence-length, otherwise the recurrent state is never read back mid-sequence and INT8 quantization is never actually exercised")

    model = load_bdh_vb(
        args.checkpoint, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, d_state=args.d_state,
    )
    sequences = load_validation_sequences(args.validation_data, args.num_sequences, args.sequence_length)

    fp32_loss = evaluate_validation_loss(model, sequences, step_fn=bdh_vb_stream_chunk, init_fn=lambda m, b: init_bdh_vb_states(m, b), sub_batch=args.sub_batch, stream_chunk_length=args.stream_chunk_length)
    int8_loss = evaluate_validation_loss(model, sequences, step_fn=bdh_vb_stream_chunk_int8_state, init_fn=lambda m, b: init_bdh_vb_states_int8(m, b), sub_batch=args.sub_batch, stream_chunk_length=args.stream_chunk_length)

    result = {
        "checkpoint": str(args.checkpoint),
        "d_state": args.d_state,
        "num_sequences": len(sequences),
        "stream_chunk_length": args.stream_chunk_length,
        "fp32_state_validation_loss": fp32_loss,
        "int8_state_validation_loss": int8_loss,
        "int8_drift": int8_loss - fp32_loss,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
