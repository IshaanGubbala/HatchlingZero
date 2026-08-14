"""HZ-Core-1 real next step: passkey/reassignment quality checks against
the ACTUAL trained 25M-param checkpoints from
`docs/restart/hz0h_core1_quality_25m_results.md` (exact BDH and BDH+VB,
trained on real text via `scripts/hz0h_stage2_runner_bdh.py`/
`scripts/hz0h_stage2_runner_bdh_vb.py`), not a freshly-initialized small
model trained directly for the task the way every other H5 result this
session was produced.

Real, important framing difference from every prior H5 result: this
model was never trained on these synthetic in-context tasks at all --
it was trained on real byte-level text. This tests the pretrained
model's OUT-OF-THE-BOX in-context recall capability (the realistic way
such evals are used for real language models), not a model
purpose-trained for the task.

Real methodological finding made while building this, disclosed rather
than silently worked around: `reference/hz0h_bdh_h5_memory_tasks.py`'s
`PASSKEY_VALUE_BASE = 12` (values live in bytes 12-19, obscure ASCII
control characters like form-feed/shift-out) is a reasonable choice for
a model trained FROM SCRATCH directly on this synthetic task (every
other H5 result this session), but is a real confound for a model
pretrained on general real text: control bytes 12-19 essentially never
appear in real text, so a real-text-pretrained model assigns them
~1e-6 probability regardless of context -- checked directly against the
real exact-BDH checkpoint (bytes 12-19 summed to ~0.001 total
probability mass at a sample real-text context, vs. ~0.151 for letter
bytes 97-104 at the same context, a ~150x difference). Argmax accuracy
under the original byte range is trivially near-zero for ANY
real-text-pretrained model regardless of true retrieval capability --
not a fair test. Fixed here by using a real-text-plausible value
alphabet (lowercase letters, byte 97 base) instead, via local generator
functions mirroring `make_passkey_sequence`/`make_reassignment_sequence`'s
exact structure with a configurable value base (the upstream functions
hardcode the module-level byte constants, not parameterized) --
deliberately NOT changing `reference/hz0h_bdh_h5_memory_tasks.py` itself,
since its existing constants are correct and appropriate for every
purpose-trained-from-scratch use elsewhere in this session.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states
from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
)
from reference.hz0h_bdh_vb_selective_torch import (
    BDHVBSelective,
    BDHVBSelectiveConfig,
    bdh_vb_selective_stream_chunk,
    init_bdh_vb_selective_states,
)
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM

MARKER = 10  # '\n' -- real, common in text; used only as an input signal to attend to, not an output target
QUERY_MARKER = 11
VALUE_BASE = 97  # 'a' -- real-text-plausible value alphabet, see module docstring


def make_real_text_passkey_sequence(rng: np.random.Generator, *, prefix_len: int, filler_len: int, value_range: int) -> tuple[list[int], int]:
    prefix = [int(rng.integers(0, 10)) for _ in range(prefix_len)]
    value = VALUE_BASE + int(rng.integers(0, value_range))
    filler = [int(rng.integers(0, 10)) for _ in range(filler_len)]
    seq = prefix + [MARKER, value] + filler + [QUERY_MARKER]
    return seq, value


def make_real_text_reassignment_sequence(rng: np.random.Generator, *, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int = 3) -> tuple[list[int], int]:
    prefix = [int(rng.integers(0, 10)) for _ in range(prefix_len)]
    seq = list(prefix)
    last_value = None
    for _ in range(num_reassignments):
        value = VALUE_BASE + int(rng.integers(0, value_range))
        filler = [int(rng.integers(0, 10)) for _ in range(filler_len)]
        seq += [MARKER, value] + filler
        last_value = value
    seq += [QUERY_MARKER]
    return seq, last_value


def load_bdh(checkpoint_pt: Path, *, n_embd: int, n_layer: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int) -> BDH:
    config = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0)
    model = BDH(config)
    blob = torch.load(str(checkpoint_pt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def load_bdh_vb(checkpoint_pt: Path, *, n_embd: int, n_layer: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int, d_state: int) -> BDHVB:
    config = BDHVBConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0, d_state=d_state)
    model = BDHVB(config)
    blob = torch.load(str(checkpoint_pt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def load_bdh_vb_selective(checkpoint_pt: Path, *, n_embd: int, n_layer: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int, d_state: int) -> BDHVBSelective:
    config = BDHVBSelectiveConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0, d_state=d_state)
    model = BDHVBSelective(config)
    blob = torch.load(str(checkpoint_pt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_bdh_passkey(model: BDH, *, prefix_len: int, filler_len: int, value_range: int, num_examples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct = 0, 0
    for _ in range(num_examples):
        seq, answer = make_real_text_passkey_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range)
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_bdh_states(model, batch_size=1)
        states, _ = bdh_stream_chunk(model, states, prefix_idx, start_position=0)
        _states, real_logits = bdh_stream_chunk(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_bdh_states(model, batch_size=1)
        _zstates, zeroed_logits = bdh_stream_chunk(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
    return {"real_state_accuracy": real_correct / num_examples, "zeroed_state_accuracy": zeroed_correct / num_examples, "num_examples": num_examples}


@torch.no_grad()
def evaluate_bdh_reassignment(model: BDH, *, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int, num_examples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct, real_predicts_first_value = 0, 0, 0
    for _ in range(num_examples):
        seq, answer = make_real_text_reassignment_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range, num_reassignments=num_reassignments)
        first_value = seq[prefix_len + 1]
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_bdh_states(model, batch_size=1)
        states, _ = bdh_stream_chunk(model, states, prefix_idx, start_position=0)
        _states, real_logits = bdh_stream_chunk(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_bdh_states(model, batch_size=1)
        _zstates, zeroed_logits = bdh_stream_chunk(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
        real_predicts_first_value += int(real_pred == first_value and answer != first_value)
    return {
        "real_state_accuracy": real_correct / num_examples,
        "zeroed_state_accuracy": zeroed_correct / num_examples,
        "real_state_predicts_stale_first_value_rate": real_predicts_first_value / num_examples,
        "num_examples": num_examples,
    }


@torch.no_grad()
def evaluate_vb_passkey(model: BDHVB, *, prefix_len: int, filler_len: int, value_range: int, num_examples: int, seed: int, int8: bool) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct = 0, 0
    init_fn = init_bdh_vb_states_int8 if int8 else init_bdh_vb_states
    step_fn = bdh_vb_stream_chunk_int8_state if int8 else bdh_vb_stream_chunk
    for _ in range(num_examples):
        seq, answer = make_real_text_passkey_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range)
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_fn(model, 1)
        states, _ = step_fn(model, states, prefix_idx, start_position=0)
        _states, real_logits = step_fn(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_fn(model, 1)
        _zstates, zeroed_logits = step_fn(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
    return {"real_state_accuracy": real_correct / num_examples, "zeroed_state_accuracy": zeroed_correct / num_examples, "num_examples": num_examples}


@torch.no_grad()
def evaluate_vb_reassignment(model: BDHVB, *, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int, num_examples: int, seed: int, int8: bool) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct, real_predicts_first_value = 0, 0, 0
    init_fn = init_bdh_vb_states_int8 if int8 else init_bdh_vb_states
    step_fn = bdh_vb_stream_chunk_int8_state if int8 else bdh_vb_stream_chunk
    for _ in range(num_examples):
        seq, answer = make_real_text_reassignment_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range, num_reassignments=num_reassignments)
        first_value = seq[prefix_len + 1]
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_fn(model, 1)
        states, _ = step_fn(model, states, prefix_idx, start_position=0)
        _states, real_logits = step_fn(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_fn(model, 1)
        _zstates, zeroed_logits = step_fn(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
        real_predicts_first_value += int(real_pred == first_value and answer != first_value)
    return {
        "real_state_accuracy": real_correct / num_examples,
        "zeroed_state_accuracy": zeroed_correct / num_examples,
        "real_state_predicts_stale_first_value_rate": real_predicts_first_value / num_examples,
        "num_examples": num_examples,
    }


@torch.no_grad()
def evaluate_vb_selective_passkey(model: BDHVBSelective, *, prefix_len: int, filler_len: int, value_range: int, num_examples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct = 0, 0
    for _ in range(num_examples):
        seq, answer = make_real_text_passkey_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range)
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_bdh_vb_selective_states(model, 1)
        states, _ = bdh_vb_selective_stream_chunk(model, states, prefix_idx, start_position=0)
        _states, real_logits = bdh_vb_selective_stream_chunk(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_bdh_vb_selective_states(model, 1)
        _zstates, zeroed_logits = bdh_vb_selective_stream_chunk(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
    return {"real_state_accuracy": real_correct / num_examples, "zeroed_state_accuracy": zeroed_correct / num_examples, "num_examples": num_examples}


@torch.no_grad()
def evaluate_vb_selective_reassignment(model: BDHVBSelective, *, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int, num_examples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct, real_predicts_first_value = 0, 0, 0
    for _ in range(num_examples):
        seq, answer = make_real_text_reassignment_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range, num_reassignments=num_reassignments)
        first_value = seq[prefix_len + 1]
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_bdh_vb_selective_states(model, 1)
        states, _ = bdh_vb_selective_stream_chunk(model, states, prefix_idx, start_position=0)
        _states, real_logits = bdh_vb_selective_stream_chunk(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_bdh_vb_selective_states(model, 1)
        _zstates, zeroed_logits = bdh_vb_selective_stream_chunk(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
        real_predicts_first_value += int(real_pred == first_value and answer != first_value)
    return {
        "real_state_accuracy": real_correct / num_examples,
        "zeroed_state_accuracy": zeroed_correct / num_examples,
        "real_state_predicts_stale_first_value_rate": real_predicts_first_value / num_examples,
        "num_examples": num_examples,
    }


def load_matched_transformer(checkpoint_pt: Path, *, n_embd: int, n_layer: int, n_head: int, d_ff: int, vocab_size: int) -> MatchedTransformerLM:
    config = MatchedTransformerConfig({"vocab_size": vocab_size, "d_model": n_embd, "num_layers": n_layer, "num_heads": n_head, "head_dim": n_embd // n_head, "d_ff": d_ff, "use_rope": True})
    model = MatchedTransformerLM(config)
    blob = torch.load(str(checkpoint_pt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def _content_free_prefix(rng: np.random.Generator, length: int) -> list[int]:
    """Same alphabet as the passkey/reassignment prefix filler (bytes
    0-9), same length, but no MARKER/value/QUERY_MARKER structure at
    all -- carries no information about any passkey value. Used as the
    Transformer's "zeroed" ablation prefix instead of a literally empty
    KV-cache: `MatchedTransformerBlock.forward` derives RoPE position
    purely from `kv_cache["k"].shape[2]` (real, checked directly against
    the code before trusting this), so an empty cache would put the
    query token at position 0 instead of its true absolute position --
    a real, uncontrolled confound (wrong position, not just "no
    content") that a literal empty-cache ablation would introduce. This
    keeps the query at the correct real position (same prefix length)
    while still removing the actual informative content, the real
    intent behind BDH-family's zeroed-state ablation."""
    return [int(rng.integers(0, 10)) for _ in range(length)]


@torch.no_grad()
def evaluate_transformer_passkey(model: MatchedTransformerLM, *, prefix_len: int, filler_len: int, value_range: int, num_examples: int, seed: int) -> dict:
    """No persistent recurrent state exists for a Transformer -- its
    real KV-cache (populated by streaming the prefix through
    model.forward(kv_cache=...)) plays the same real, functional role
    the BDH-family harness's `states` does: it's what the model
    actually carries forward from the prefix into the query. See
    `_content_free_prefix`'s docstring for why the "zeroed" ablation
    here is a same-length content-free prefix, not a literally empty
    cache."""
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct = 0, 0
    for _ in range(num_examples):
        seq, answer = make_real_text_passkey_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range)
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        cache = model.new_kv_cache()
        model(prefix_idx, kv_cache=cache)
        real_logits = model(query_idx, kv_cache=cache)
        real_pred = int(real_logits[0, -1].argmax())

        zeroed_prefix = torch.tensor([_content_free_prefix(rng, prefix_idx.shape[1])], dtype=torch.long)
        zeroed_cache = model.new_kv_cache()
        model(zeroed_prefix, kv_cache=zeroed_cache)
        zeroed_logits = model(query_idx, kv_cache=zeroed_cache)
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
    return {"real_state_accuracy": real_correct / num_examples, "zeroed_state_accuracy": zeroed_correct / num_examples, "num_examples": num_examples}


@torch.no_grad()
def evaluate_transformer_reassignment(model: MatchedTransformerLM, *, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int, num_examples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real_correct, zeroed_correct, real_predicts_first_value = 0, 0, 0
    for _ in range(num_examples):
        seq, answer = make_real_text_reassignment_sequence(rng, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range, num_reassignments=num_reassignments)
        first_value = seq[prefix_len + 1]
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        cache = model.new_kv_cache()
        model(prefix_idx, kv_cache=cache)
        real_logits = model(query_idx, kv_cache=cache)
        real_pred = int(real_logits[0, -1].argmax())

        zeroed_prefix = torch.tensor([_content_free_prefix(rng, prefix_idx.shape[1])], dtype=torch.long)
        zeroed_cache = model.new_kv_cache()
        model(zeroed_prefix, kv_cache=zeroed_cache)
        zeroed_logits = model(query_idx, kv_cache=zeroed_cache)
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
        real_predicts_first_value += int(real_pred == first_value and answer != first_value)
    return {
        "real_state_accuracy": real_correct / num_examples,
        "zeroed_state_accuracy": zeroed_correct / num_examples,
        "real_state_predicts_stale_first_value_rate": real_predicts_first_value / num_examples,
        "num_examples": num_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a *_checkpoint_best.pt or *_checkpoint.pt file")
    parser.add_argument("--architecture", choices=("bdh", "bdh_vb", "bdh_vb_selective", "transformer"), required=True)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--d-state", type=int, default=128, help="only used for --architecture bdh_vb. Default 128 matches HZ-Core-1's original D/4 checkpoint (n_embd=512); Phase B's D/2 (256) and D/3 (round(512/3)=171) checkpoints need this set explicitly to match how they were trained (see scripts/hz0h_stage2_runner_bdh_vb_depth_curriculum.py's own --d-state-divisor).")
    parser.add_argument("--d-ff", type=int, default=2048, help="only used for --architecture transformer. Default 2048 matches Phase F's matched-Transformer config, configs/hz0h_transformer_matched_25m.json.")
    args = parser.parse_args()

    if args.architecture == "bdh":
        model = load_bdh(args.checkpoint, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size)
        passkey = evaluate_bdh_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_examples, seed=1000)
        reassignment = evaluate_bdh_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_examples, seed=2000)
        result = {"architecture": "bdh", "passkey": passkey, "reassignment": reassignment}
    elif args.architecture == "bdh_vb":
        model = load_bdh_vb(args.checkpoint, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, d_state=args.d_state)
        passkey_fp32 = evaluate_vb_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_examples, seed=1000, int8=False)
        passkey_int8 = evaluate_vb_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_examples, seed=1000, int8=True)
        reassignment_fp32 = evaluate_vb_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_examples, seed=2000, int8=False)
        reassignment_int8 = evaluate_vb_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_examples, seed=2000, int8=True)
        result = {
            "architecture": "bdh_vb", "d_state": model.config.d_state,
            "passkey_fp32": passkey_fp32, "passkey_int8": passkey_int8,
            "reassignment_fp32": reassignment_fp32, "reassignment_int8": reassignment_int8,
        }
    elif args.architecture == "bdh_vb_selective":
        model = load_bdh_vb_selective(args.checkpoint, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, d_state=args.d_state)
        passkey = evaluate_vb_selective_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_examples, seed=1000)
        reassignment = evaluate_vb_selective_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_examples, seed=2000)
        result = {"architecture": "bdh_vb_selective", "d_state": model.config.d_state, "passkey": passkey, "reassignment": reassignment}
    else:
        model = load_matched_transformer(args.checkpoint, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, d_ff=args.d_ff, vocab_size=args.vocab_size)
        passkey = evaluate_transformer_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_examples, seed=1000)
        reassignment = evaluate_transformer_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_examples, seed=2000)
        result = {"architecture": "transformer", "passkey": passkey, "reassignment": reassignment}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
