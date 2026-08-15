"""Phase F domain cross-entropy comparison.

Evaluates the already-trained matched checkpoints on the same byte-packed
code and math/reasoning validation streams.  Checkpoints are explicit CLI
inputs so an absent or mismatched artifact cannot silently become a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_prefill_chunked
from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig, bdh_vb_stream_prefill_chunked
from reference.hz0h_bdh_block_gated_torch import BDHBlockGated, BDHBlockGatedConfig, bdh_block_gated_annealed_direct_split_v_forward
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


DOMAIN_FILES = {
    "code": Path("data/packed/external/code_validation.jsonl"),
    "math_reasoning": Path("data/packed/external/mathematical_and_structured_validation.jsonl"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_hashes(path: Path) -> set[str]:
    hashes = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            hashes.add(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest())
    return hashes


def read_sequences(path: Path, *, max_sequences: int | None = None) -> list[list[int]]:
    sequences = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, list) or len(value) < 2:
                raise ValueError(f"{path}: expected token-list JSONL records")
            sequences.append([int(token) for token in value])
            if max_sequences is not None and len(sequences) >= max_sequences:
                break
    if not sequences:
        raise ValueError(f"{path}: no sequences")
    return sequences


def _load_blob(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_arm(name: str, checkpoint: Path, device: torch.device, dtype: torch.dtype):
    blob = _load_blob(checkpoint)
    if name == "bdh":
        model = BDH(BDHConfig(n_layer=8, n_embd=512, n_head=8, mlp_internal_dim_multiplier=32, vocab_size=256, dropout=0.0))
    elif name == "bdh_vb":
        model = BDHVB(BDHVBConfig(n_layer=8, n_embd=512, n_head=8, mlp_internal_dim_multiplier=32, vocab_size=256, dropout=0.0, d_state=128))
    elif name == "block_gated_direct_split_v":
        model = BDHBlockGated(BDHBlockGatedConfig(n_layer=4, n_embd=512, n_head=8, mlp_internal_dim_multiplier=32, vocab_size=256, dropout=0.0, block_size=16))
    elif name == "transformer":
        config = MatchedTransformerConfig({"vocab_size": 256, "d_model": 512, "num_layers": 6, "num_heads": 4, "head_dim": 128, "d_ff": 2048, "use_rope": True})
        model = MatchedTransformerLM(config)
    else:
        raise ValueError(name)
    model.load_state_dict(blob["model"])
    model = model.to(device=device, dtype=dtype).eval()
    if hasattr(model, "attn"):
        model.attn.freqs = model.attn.freqs.to(torch.float32)
    return model


@torch.inference_mode()
def evaluate(model, sequences: list[list[int]], *, batch_size: int, device: torch.device, prefill_chunk_length: int, block_gated_active_fraction: float) -> dict:
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    peak_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    for offset in range(0, len(sequences), batch_size):
        batch = sequences[offset:offset + batch_size]
        width = min(len(row) for row in batch)
        tokens = torch.tensor([row[:width] for row in batch], dtype=torch.long, device=device)
        if isinstance(model, BDH):
            _, logits = bdh_stream_prefill_chunked(model, tokens, chunk_length=prefill_chunk_length)
        elif isinstance(model, BDHVB):
            _, logits = bdh_vb_stream_prefill_chunked(model, tokens, chunk_length=prefill_chunk_length)
        elif isinstance(model, BDHBlockGated):
            logits, _ = bdh_block_gated_annealed_direct_split_v_forward(model, tokens, block_gated_active_fraction)
        else:
            output = model(tokens)
            logits = output[0] if isinstance(output, tuple) else output
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="sum")
        total_nll += float(loss)
        total_tokens += tokens.shape[0] * (width - 1)
    elapsed = max(time.perf_counter() - started, 1e-9)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else peak_before
    ce = total_nll / total_tokens
    return {"tokens": total_tokens, "cross_entropy": ce, "perplexity": math.exp(min(ce, 80.0)), "seconds": elapsed, "tokens_per_second": total_tokens / elapsed, "peak_memory_bytes": peak, "finite": math.isfinite(ce), "bdh_prefill_path": "chunked_streaming" if isinstance(model, (BDH, BDHVB)) else "learned_gate_direct_split_v" if isinstance(model, BDHBlockGated) else "plain_forward", "block_gated_active_fraction": block_gated_active_fraction if isinstance(model, BDHBlockGated) else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdh-checkpoint", type=Path)
    parser.add_argument("--bdh-vb-checkpoint", type=Path)
    parser.add_argument("--transformer-checkpoint", type=Path)
    parser.add_argument("--block-gated-direct-split-v-checkpoint", type=Path, help="Experimental derivative; evaluated only at its trained sparse fraction.")
    parser.add_argument("--block-gated-active-fraction", type=float, default=0.5)
    parser.add_argument("--code-data", type=Path, default=DOMAIN_FILES["code"])
    parser.add_argument("--math-data", type=Path, default=DOMAIN_FILES["math_reasoning"])
    parser.add_argument("--training-data", type=Path, help="Optional packed training JSONL for exact-record contamination check.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="Chunk size for bounded BDH/VB streaming CE evaluation; Transformer CE remains plain forward.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if not 0.0 < args.block_gated_active_fraction <= 1.0:
        raise ValueError("--block-gated-active-fraction must be in (0, 1]")
    checkpoints = {"bdh": args.bdh_checkpoint, "bdh_vb": args.bdh_vb_checkpoint, "transformer": args.transformer_checkpoint, "block_gated_direct_split_v": args.block_gated_direct_split_v_checkpoint}
    checkpoints = {name: path for name, path in checkpoints.items() if path is not None}
    if not checkpoints:
        raise ValueError("provide at least one checkpoint")
    domains = {"code": read_sequences(args.code_data, max_sequences=args.max_sequences), "math_reasoning": read_sequences(args.math_data, max_sequences=args.max_sequences)}
    report = {"phase": "F", "evaluation": "domain_cross_entropy", "device": str(device), "dtype": args.dtype, "domains": {}, "arms": {}, "block_gated_direct_split_v_is_derivative": True, "claim_eligible": False}
    domain_paths = {"code": args.code_data, "math_reasoning": args.math_data}
    for domain, sequences in domains.items():
        domain_path = domain_paths[domain]
        report["domains"][domain] = {"path": str(domain_path), "sha256": file_sha256(domain_path), "sequences": len(sequences), "sequence_length": len(sequences[0])}
    if args.training_data is None:
        report["contamination_check"] = {"checked": False, "reason": "--training-data not supplied"}
    else:
        training_hashes = canonical_record_hashes(args.training_data)
        report["contamination_check"] = {"checked": True, "method": "exact canonical JSON token-record SHA256 intersection (does not detect substrings)", "training_data": str(args.training_data), "training_data_sha256": file_sha256(args.training_data), "training_record_count": len(training_hashes), "overlap_records": {domain: len(training_hashes.intersection(canonical_record_hashes(path))) for domain, path in domain_paths.items()}}
    for name, checkpoint in checkpoints.items():
        model = load_arm(name, checkpoint, device, dtype)
        report["arms"][name] = {"checkpoint": str(checkpoint), "domains": {domain: evaluate(model, sequences, batch_size=args.batch_size, device=device, prefill_chunk_length=args.prefill_chunk_length, block_gated_active_fraction=args.block_gated_active_fraction) for domain, sequences in domains.items()}}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
