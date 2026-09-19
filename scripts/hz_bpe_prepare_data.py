"""BPE-token corpus preparation, replacing the byte-level corpus
(hz2_patch_prepare_data.py) for the vocab-size switch requested after the
byte-level HZ chat model (val_acc=0.35, 50k steps) generated only garbage
under both greedy and sampled decoding -- confirmed a real exposure-bias /
compounding-error problem at byte granularity, not a pipeline bug (see
results/local/hz_300m_committed_run2.log). Byte-level BPE (same alphabet
trick as GPT-2: every byte value is representable, so there is no <unk>
and no lossy re-encoding) raises per-step information content, which is
the intended fix.

Reuses stream_documents/collect_domain from hz2_patch_prepare_data.py
UNCHANGED -- same sources, same weights, same per-domain chronological
sort, same domain-timeout safety. The only real change here is what gets
packed: BPE token ids (uint16, since vocab_size <= 65535) instead of raw
bytes (uint8).

Vocab-size choice (real constraint, not arbitrary): the live model
(reference/hz_language_model_torch.py) has SEPARATE token_embed
(Embedding(vocab, d_model)) and lm_head (Linear(d_model, vocab, bias=False))
-- not tied. At d_model=3072, going from vocab=256 to vocab=V costs
(V-256)*3072*2 extra params. The prior chat_only run was 237,751,506 params
total; keeping the stated "sub 300M" budget leaves ~62.25M of headroom,
which caps V at ~10,387. vocab_size=8192 (default below) lands at
~286.5M total, a real, checked margin under 300M.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz2_patch_prepare_data import SOURCES, collect_domain, sha256_file  # noqa: E402


def train_tokenizer(sample_train_chunks: list[bytes], vocab_size: int, out_dir: Path) -> ByteLevelBPETokenizer:
    """Trains a byte-level BPE tokenizer directly from the already-collected
    sample documents (decoded utf-8, errors ignored per-document -- avoids
    the cross-document byte-boundary corruption that decoding a
    concatenated blob would risk). No special tokens: byte-level BPE's
    alphabet already covers all 256 byte values, so every string is
    encodable without an <unk>, matching the existing ByteTokenizer's
    guarantee."""
    texts = [chunk.decode("utf-8", errors="ignore") for chunk in sample_train_chunks]
    tmp_path = out_dir / "_tokenizer_train_sample.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("\n".join(texts))
    tok = ByteLevelBPETokenizer()
    tok.train(files=[str(tmp_path)], vocab_size=vocab_size, min_frequency=2, special_tokens=[])
    tmp_path.unlink()
    return tok


def pack_token_sequences(byte_chunks: list[bytes], tok: ByteLevelBPETokenizer, sequence_length: int,
                         out_path: Path) -> int:
    """Tokenizes each document chunk independently (no cross-document
    token bleed) and concatenates the resulting token-id streams, then
    packs into fixed-length uint16 sequences -- same truncation-of-
    remainder convention as hz2_patch_prepare_data.pack_sequences."""
    import numpy as np
    all_ids: list[int] = []
    for chunk in byte_chunks:
        text = chunk.decode("utf-8", errors="ignore")
        if text:
            all_ids.extend(tok.encode(text).ids)
    n_sequences = len(all_ids) // sequence_length
    arr = np.array(all_ids[:n_sequences * sequence_length], dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return n_sequences


class BinaryTokenCorpusPool:
    """uint16-token analogue of hz2_patch_prepare_data.BinaryCorpusPool."""

    def __init__(self, path: Path, sequence_length: int, max_sequences: int | None = None):
        import numpy as np
        self.sequence_length = sequence_length
        total_bytes = Path(path).stat().st_size
        n_sequences = total_bytes // (sequence_length * 2)
        if n_sequences == 0:
            raise ValueError(f"{path} is smaller than one sequence_length={sequence_length} uint16 tokens")
        self.n_sequences = n_sequences if max_sequences is None else min(n_sequences, max_sequences)
        self._mmap = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_sequences, sequence_length))

    def sample(self, rng):
        i = rng.randrange(self.n_sequences)
        return self._mmap[i].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bytes", type=int, default=4_500_000_000,
                        help="Total raw source bytes (pre-tokenization) across all domains -- "
                             "same scale target as the byte-corpus run for a fair comparison.")
    parser.add_argument("--val-bytes-per-domain", type=int, default=2_000_000)
    parser.add_argument("--tokenizer-sample-bytes-per-domain", type=int, default=60_000_000,
                        help="Raw bytes per domain used ONLY to train the BPE tokenizer itself, "
                             "before the full collection pass.")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=256, help="In TOKENS, not bytes.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("results/local/hz_bpe_corpus_v1"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("results/local/hz_bpe_tokenizer"))
    parser.add_argument("--domain-timeout-seconds", type=int, default=900)
    parser.add_argument("--reuse-tokenizer", action="store_true",
                        help="Load an already-trained tokenizer from --tokenizer-dir instead of retraining.")
    args = parser.parse_args()

    manifest = {"seed": args.seed, "sequence_length_tokens": args.sequence_length,
               "train_byte_target": args.train_bytes, "vocab_size_requested": args.vocab_size,
               "format": "binary_uint16_bpe_fixed_length", "domains": []}
    start_time = time.time()

    vocab_file = args.tokenizer_dir / "vocab.json"
    merges_file = args.tokenizer_dir / "merges.txt"
    if args.reuse_tokenizer and vocab_file.exists():
        print(f"loading existing tokenizer from {args.tokenizer_dir}", flush=True)
        tok = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
    else:
        print(f"collecting tokenizer-training sample "
             f"({args.tokenizer_sample_bytes_per_domain:,} bytes/domain)...", flush=True)
        sample_chunks: list[bytes] = []
        for name, config, weight, text_key, date_key in SOURCES:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(collect_domain, name, config, text_key, date_key,
                                         args.tokenizer_sample_bytes_per_domain, 0, args.seed)
                train_chunks, _val, train_bytes, _vb, _nu = future.result(timeout=args.domain_timeout_seconds)
            print(f"  {name}: {train_bytes:,} sample bytes for tokenizer training", flush=True)
            sample_chunks.extend(train_chunks)
        print(f"training ByteLevelBPE tokenizer, vocab_size={args.vocab_size}...", flush=True)
        tok = train_tokenizer(sample_chunks, args.vocab_size, args.tokenizer_dir)
        tok.save_model(str(args.tokenizer_dir))
    actual_vocab_size = tok.get_vocab_size()
    print(f"tokenizer ready, actual vocab_size={actual_vocab_size}", flush=True)
    manifest["vocab_size_actual"] = actual_vocab_size
    manifest["tokenizer_dir"] = str(args.tokenizer_dir)

    all_train_chunks, per_domain_val = [], {}
    domain_timings = []
    for name, config, weight, text_key, date_key in SOURCES:
        domain_train_budget = int(args.train_bytes * weight)
        print(f"streaming {name} ({config}), target train={domain_train_budget:,} bytes...", flush=True)
        domain_t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(collect_domain, name, config, text_key, date_key,
                                     domain_train_budget, args.val_bytes_per_domain, args.seed)
            try:
                train_chunks, val_chunks, train_bytes, val_bytes, n_undated = future.result(
                    timeout=args.domain_timeout_seconds)
            except concurrent.futures.TimeoutError:
                print(f"  TIMEOUT: {name} exceeded {args.domain_timeout_seconds}s -- treating as 0 bytes.",
                     flush=True)
                domain_timings.append({"source": name, "timed_out": True})
                per_domain_val[name.split("/")[-1]] = []
                continue
        domain_elapsed = time.time() - domain_t0
        domain_timings.append({"source": name, "elapsed_seconds": domain_elapsed})
        print(f"  {name}: {domain_elapsed:.1f}s, {(train_bytes+val_bytes)/domain_elapsed:,.0f} bytes/sec",
             flush=True)
        all_train_chunks.extend(train_chunks)
        domain_key = name.split("/")[-1]
        per_domain_val[domain_key] = val_chunks
        manifest["domains"].append({
            "source": name, "config": config, "weight": weight,
            "train_bytes_actual": train_bytes, "val_bytes_actual": val_bytes,
            "n_train_docs": len(train_chunks), "n_val_docs": len(val_chunks),
        })

    pack_t0 = time.time()
    train_path = args.out_dir / "train.bin"
    n_train_seq = pack_token_sequences(all_train_chunks, tok, args.sequence_length, train_path)
    manifest["train_file"] = str(train_path)
    manifest["train_sequences"] = n_train_seq
    manifest["train_file_sha256"] = sha256_file(train_path)
    manifest["train_file_bytes"] = train_path.stat().st_size
    manifest["train_tokens_total"] = n_train_seq * args.sequence_length

    manifest["val_files"] = {}
    for domain_key, val_chunks in per_domain_val.items():
        val_path = args.out_dir / f"val_{domain_key}.bin"
        n_val_seq = pack_token_sequences(val_chunks, tok, args.sequence_length, val_path)
        manifest["val_files"][domain_key] = {
            "path": str(val_path), "sequences": n_val_seq, "sha256": sha256_file(val_path),
            "bytes": val_path.stat().st_size,
        }
    packing_elapsed = time.time() - pack_t0
    total_elapsed = time.time() - start_time
    manifest["timing"] = {"domains": domain_timings, "packing_seconds": packing_elapsed,
                          "total_seconds": total_elapsed}

    manifest_path = args.out_dir / "manifest.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote manifest to {manifest_path}")
    print(f"total elapsed: {total_elapsed:.1f}s, {n_train_seq:,} train sequences of "
         f"{args.sequence_length} tokens = {n_train_seq*args.sequence_length:,} tokens")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
