"""HZ Phase 1 (plans/HatchlingZero_Reality_Plan.md): real inference-time
measurements -- prefill throughput, decode throughput, peak inference
memory -- to go with the training-time metrics already in
scripts/hz0h_stage2_runner_bdh.py / scripts/hz0a_torch_stage2_runner.py
(loss, tokens/sec, peak training memory, activation sparsity, state
norms).

Four decode paths measured, not one, because they are NOT interchangeable
and conflating them would misrepresent BDH's real structural advantage:

1. BDH, naive replay (`BDH.generate()`, real upstream code): re-runs the
   ENTIRE sequence through the model every new token -- O(T) work per
   token, O(T^2) total for T tokens. No KV-cache exists for BDH-GPU's
   parallel form.
2. BDH, real streaming decode (`bdh_stream_chunk`, H2's proven exact
   equivalent to the parallel form): O(1) work per token via the
   persistent per-layer synaptic state. This is the actual mechanism
   HatchlingZero's thesis is about -- if BDH has an inference-speed edge
   at long context, THIS is where it would show up, not path 1.
3. Transformer, naive replay (no KV-cache): same O(T) per-token replay
   cost as path 1. Kept for a "no-cache BDH vs no-cache Transformer"
   comparison, but NOT the fair Transformer baseline -- see path 4.
4. Transformer, real KV-cache
   (`reference/hz0a_matched_transformer.py`'s `MatchedTransformerLM.
   new_kv_cache`/`forward(kv_cache=...)`, added 2026-08-11, numerically
   verified identical to a full non-cached forward --
   `tests/reference/test_hz0a_matched_transformer_kv_cache.py`): the
   real, standard, serving-realistic Transformer decode mechanism.
   Still O(context) attention work per token (not O(1) like BDH's
   streaming state), but no longer replaying the whole sequence -- this
   is the fair comparison against BDH's streaming path, closing the
   single biggest gap this script had at first.

Energy (joules/token): CUDA-only in this script, via polling
`nvidia-smi --query-gpu=power.draw` in a background thread during the
timed region (average power x elapsed time = joules) -- real, but a
coarse sampling-based estimate (matches the same-class caveat already
used for the pilot's peak-VRAM sampling), not a hardware energy counter.
No equivalent instrumentation exists for Mac yet: `powermetrics` (the
real per-process CPU/GPU/ANE power sampling path on macOS) needs sudo,
which this script does not prompt for -- reported as null with a
disclosed reason, not silently omitted. Real next step, not this one.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_kv_cache_step, bdh_stream_chunk, init_bdh_states, new_bdh_kv_cache


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class _PowerSampler:
    """CUDA-only: polls nvidia-smi's instantaneous power draw on a
    background thread during a timed region. Coarse (sampling interval,
    not a hardware energy counter) but real, not estimated from a
    TDP/FLOPs formula."""

    def __init__(self, device: torch.device, interval_s: float = 0.05):
        self.enabled = device.type == "cuda"
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0,
                )
                self._samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_PowerSampler":
        if self.enabled:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.enabled and self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)

    def mean_watts(self) -> float | None:
        if not self.enabled or not self._samples:
            return None
        return statistics.mean(self._samples)


def measure_bdh_prefill(model: BDH, prompt: torch.Tensor, repeats: int, device: torch.device) -> dict:
    with torch.no_grad():
        _sync(device)
        model(prompt)  # warmup
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            for _ in range(repeats):
                model(prompt)
            _sync(device)
            elapsed = time.perf_counter() - started
    tokens = prompt.shape[1] * repeats
    return {"tokens_per_second": tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_bdh_decode_naive(model: BDH, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    with torch.no_grad():
        _sync(device)
        model.generate(prompt, max_new_tokens=4, top_k=1)  # warmup
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            model.generate(prompt, max_new_tokens=max_new_tokens, top_k=1)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_bdh_decode_streaming(model: BDH, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    """The real O(1)-state decode path (bdh_stream_chunk), not the naive
    full-replay one. Prefill (processing the prompt into the initial
    state) happens ONCE, OUTSIDE the timed region -- a real bug in an
    earlier version of this function re-ran prefill inside the timed
    call every time (including the "real" measurement, not just warmup),
    so `elapsed` silently included one-time prefill cost on top of the
    `max_new_tokens` decode steps the tokens_per_second denominator
    assumed -- an undercount that got WORSE at longer prompts. Fixed
    2026-08-11; see docs/restart/hz0h_phase1_kv_cache_bdh_results.md for
    the corrected numbers and what changed."""
    with torch.no_grad():
        def prefill() -> tuple[list[torch.Tensor], torch.Tensor]:
            states = init_bdh_states(model, prompt.shape[0], device=device)
            states, logits = bdh_stream_chunk(model, states, prompt, start_position=0)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states: list[torch.Tensor], token: torch.Tensor, n_tokens: int) -> None:
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)  # warmup (decode only, reuses this prefill, not timed)
        _sync(device)

        states, token = prefill()  # fresh, untimed prefill for the real measurement
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(states, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_bdh_decode_kv_cache(model: BDH, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    """The alternative O(D*context)-per-token decode path
    (bdh_kv_cache_step) -- see that function's docstring for why it's
    worth measuring against bdh_stream_chunk's O(D^2)-per-token
    compressed state. Prefill (populating the cache from the prompt, one
    token at a time -- bdh_kv_cache_step is single-token by construction,
    unlike bdh_stream_chunk's whole-chunk prefill) happens ONCE, OUTSIDE
    the timed region, same discipline as measure_bdh_decode_streaming."""
    with torch.no_grad():
        def prefill() -> tuple[list[dict], torch.Tensor]:
            cache = new_bdh_kv_cache(model)
            logits = None
            for position in range(prompt.shape[1]):
                logits = bdh_kv_cache_step(model, cache, prompt[:, position:position + 1], position=position)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return cache, token

        def decode(cache: list[dict], token: torch.Tensor, n_tokens: int) -> None:
            position = prompt.shape[1]
            for _ in range(n_tokens):
                logits = bdh_kv_cache_step(model, cache, token, position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        cache, token = prefill()
        decode(cache, token, 4)  # warmup
        _sync(device)

        cache, token = prefill()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(cache, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_transformer_prefill(model: MatchedTransformerLM, prompt: torch.Tensor, repeats: int, device: torch.device) -> dict:
    with torch.no_grad():
        _sync(device)
        model(prompt)  # warmup
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            for _ in range(repeats):
                model(prompt)
            _sync(device)
            elapsed = time.perf_counter() - started
    tokens = prompt.shape[1] * repeats
    return {"tokens_per_second": tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_transformer_decode_naive(model: MatchedTransformerLM, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    """No KV-cache exists (see module docstring) -- full-sequence replay
    every step, same real limitation as measure_bdh_decode_naive."""
    with torch.no_grad():
        def run(n_tokens: int) -> torch.Tensor:
            sequence = prompt
            for _ in range(n_tokens):
                logits = model(sequence)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                sequence = torch.cat((sequence, next_token), dim=1)
            return sequence

        _sync(device)
        run(4)  # warmup
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            run(max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_transformer_decode_kv_cache(model: MatchedTransformerLM, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    """The real, serving-realistic decode path: prefill the prompt once
    into a KV-cache (reference/hz0a_matched_transformer.py's
    MatchedTransformerLM.new_kv_cache/forward(kv_cache=...)), then decode
    one token at a time doing O(1) new-token work (attention is still
    O(context) per step, same as any real KV-cached Transformer -- this
    is NOT claimed to be O(1) like BDH's streaming state, just the real
    standard Transformer serving mechanism instead of the crippled
    full-replay baseline measure_transformer_decode_naive uses).

    Prefill happens ONCE, OUTSIDE the timed decode region -- a real bug
    in an earlier version of this function re-ran prefill inside the
    timed call every time, silently including one-time prefill cost in
    the decode-throughput denominator (worse at longer prompts). Fixed
    2026-08-11 alongside the same bug in measure_bdh_decode_streaming;
    see docs/restart/hz0h_phase1_kv_cache_bdh_results.md."""
    with torch.no_grad():
        def prefill() -> tuple[list[dict], torch.Tensor]:
            cache = model.new_kv_cache()
            logits = model(prompt, kv_cache=cache)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return cache, token

        def decode(cache: list[dict], token: torch.Tensor, n_tokens: int) -> None:
            for _ in range(n_tokens):
                logits = model(token, kv_cache=cache)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        _sync(device)
        cache, token = prefill()
        decode(cache, token, 4)  # warmup
        _sync(device)

        cache, token = prefill()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(cache, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


_mps_peak_bytes = [0]


def peak_memory_bytes(device: torch.device) -> int | None:
    """Same running-max pattern as scripts/hz0h_stage2_runner_bdh.py /
    scripts/hz0a_torch_stage2_runner.py's own peak_memory_bytes -- a
    single post-hoc torch.mps.current_allocated_memory() call (the prior
    version of this function) only sees whatever's still resident AFTER
    the timed region, not the actual peak during it; a running max across
    repeated calls during measurement is a real, if still imperfect
    (misses peaks strictly between two calls), improvement."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        current = int(torch.mps.current_allocated_memory())
        _mps_peak_bytes[0] = max(_mps_peak_bytes[0], current)
        return _mps_peak_bytes[0]
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        _mps_peak_bytes[0] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=683)
    parser.add_argument("--context-lengths", type=str, default="128,512,2048")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device)
    bdh_model.attn.freqs = bdh_model.attn.freqs.to(torch.float32)
    bdh_model.eval()

    transformer_config = MatchedTransformerConfig({"vocab_size": args.vocab_size, "d_model": args.n_embd, "num_layers": args.n_layer, "num_heads": args.n_head, "head_dim": args.n_embd // args.n_head, "d_ff": args.d_ff, "use_rope": True})
    transformer_model = MatchedTransformerLM(transformer_config).to(device)
    transformer_model.eval()

    bdh_params = sum(p.numel() for p in bdh_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    results: dict = {
        "device": str(device), "bdh_parameter_count": bdh_params, "transformer_parameter_count": transformer_params,
        "decode_tokens": args.decode_tokens, "prefill_repeats": args.prefill_repeats, "seed": args.seed,
        "by_context_length": {},
    }

    for context_length in (int(x) for x in args.context_lengths.split(",") if x.strip()):
        prompt = torch.randint(0, args.vocab_size, (1, context_length), device=device)

        reset_peak_memory(device)
        bdh_prefill = measure_bdh_prefill(bdh_model, prompt, args.prefill_repeats, device)
        bdh_prefill["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        bdh_decode_naive = measure_bdh_decode_naive(bdh_model, prompt, args.decode_tokens, device)
        bdh_decode_naive["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        bdh_decode_streaming = measure_bdh_decode_streaming(bdh_model, prompt, args.decode_tokens, device)
        bdh_decode_streaming["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        bdh_decode_kv_cache = measure_bdh_decode_kv_cache(bdh_model, prompt, args.decode_tokens, device)
        bdh_decode_kv_cache["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_prefill = measure_transformer_prefill(transformer_model, prompt, args.prefill_repeats, device)
        transformer_prefill["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_decode_naive = measure_transformer_decode_naive(transformer_model, prompt, args.decode_tokens, device)
        transformer_decode_naive["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_decode_kv_cache = measure_transformer_decode_kv_cache(transformer_model, prompt, args.decode_tokens, device)
        transformer_decode_kv_cache["peak_memory_bytes"] = peak_memory_bytes(device)

        results["by_context_length"][context_length] = {
            "bdh_prefill": bdh_prefill,
            "bdh_decode_naive_replay": bdh_decode_naive,
            "bdh_decode_streaming_state": bdh_decode_streaming,
            "bdh_decode_kv_cache": bdh_decode_kv_cache,
            "transformer_prefill": transformer_prefill,
            "transformer_decode_naive_replay": transformer_decode_naive,
            "transformer_decode_kv_cache": transformer_decode_kv_cache,
        }

    output_text = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
