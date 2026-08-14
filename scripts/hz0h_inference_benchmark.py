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
from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_base_delta_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8_base_delta,
)


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


def measure_vb_prefill(model: BDHVB, prompt: torch.Tensor, repeats: int, device: torch.device) -> dict:
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


def measure_vb_decode_streaming(model: BDHVB, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    """HZ-Core-2's HZ-Speed mode: plain BF16/FP32 recurrent state
    (bdh_vb_stream_chunk), same real O(1)-per-token streaming mechanism
    as measure_bdh_decode_streaming, just with the value-bottleneck-
    compressed state. Same prefill-outside-timed-region discipline as
    every other decode measurement in this file."""
    with torch.no_grad():
        def prefill() -> tuple[list[torch.Tensor], torch.Tensor]:
            states = init_bdh_vb_states(model, prompt.shape[0], device=device)
            states, logits = bdh_vb_stream_chunk(model, states, prompt, start_position=0)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states: list[torch.Tensor], token: torch.Tensor, n_tokens: int) -> None:
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_vb_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)  # warmup
        _sync(device)

        states, token = prefill()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(states, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_vb_decode_int8_base_delta(model: BDHVB, prompt: torch.Tensor, max_new_tokens: int, device: torch.device, merge_every_k: int) -> dict:
    """HZ-Core-2's HZ-Memory mode: two-level base+delta INT8 recurrent
    state (bdh_vb_stream_chunk_int8_base_delta_state), locked in
    docs/restart/hz0h_phase_d_base_delta_int8_results.md at
    merge_every_k>=32. Same real O(1)-per-token streaming shape as the
    plain-state path, with the state itself compressed 4x (INT8 base)
    at a real, previously-measured decode-throughput cost relative to
    the plain-state path (~21% slower at K=64 on the RTX3060) -- this
    function measures that same tradeoff at whatever context length/
    scale this benchmark run uses, not a re-derivation of that number."""
    with torch.no_grad():
        def prefill() -> tuple[list[dict], torch.Tensor]:
            states = init_bdh_vb_states_int8_base_delta(model, prompt.shape[0], device=device)
            states, logits = bdh_vb_stream_chunk_int8_base_delta_state(model, states, prompt, start_position=0, merge_every_k=merge_every_k)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states: list[dict], token: torch.Tensor, n_tokens: int) -> None:
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_vb_stream_chunk_int8_base_delta_state(model, states, token, start_position=position, merge_every_k=merge_every_k)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)  # warmup
        _sync(device)

        states, token = prefill()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(states, token, max_new_tokens)
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


def compute_state_bytes(*, batch_size: int, n_layer: int, n_head: int, N: int, n_embd: int, context_length: int, d_state: int | None, head_dim: int, transformer_n_layer: int | None = None, transformer_n_head: int | None = None, transformer_head_dim: int | None = None, kv_dtype_bytes: int = 2, state_dtype_bytes: int = 4) -> dict:
    """Real, analytic (not measured) byte counts for each arch's
    persistent per-token state, at a given context length -- the
    clearest direct demonstration of this plan's core architectural
    claim: BDH/VB state is O(1) in context length (shape doesn't
    depend on context_length at all), a Transformer's KV cache is
    O(context) (grows linearly with every token generated). Real
    numbers, not a peak-allocator sample that also includes transient
    compute buffers -- this isolates just the persistent state tensor
    itself.

    bdh_state_bytes: shape (B, n_head, N, n_embd) per
    reference/hz0h_bdh_torch.py's init_bdh_states -- the state's last
    dimension is the FULL n_embd, not head_dim (real, easy mistake to
    make: this is exactly what the Value Bottleneck project exists to
    shrink -- caught before trusting an earlier draft of this function
    that used head_dim here). Independent of context_length by
    construction.
    vb_state_bytes: shape (B, n_head, N, d_state) per
    reference/hz0h_bdh_vb_torch.py's init_bdh_vb_states -- same shape,
    last dim shrunk to d_state (< n_embd). Also independent of
    context_length.
    vb_int8_base_delta_state_bytes: base (1 byte/element, INT8) + delta
    (state_dtype_bytes/element, full precision) -- real total for the
    two-level design, not just the INT8 base alone.
    transformer_kv_cache_bytes: 2 (K and V) x transformer_n_layer x B x
    transformer_n_head x context_length x transformer_head_dim x
    kv_dtype_bytes -- deliberately SEPARATE transformer_n_layer/
    transformer_n_head/transformer_head_dim parameters (defaulting to
    n_layer/n_head/head_dim if not given): a parameter-matched
    Transformer often has a genuinely different layer/head count than
    the BDH/VB arms (e.g. this session's real Phase F comparison used
    8 layers/8 heads for BDH/VB but 6 layers/4 heads for the
    Transformer to hit the same ~25.5M parameter target) -- reusing
    BDH's own n_layer/n_head here would silently compute the WRONG
    model's KV-cache size whenever the two architectures' matched
    configs differ, which they usually do. This term genuinely scales
    with context_length, unlike the other three."""
    bdh_state_elements = n_layer * batch_size * n_head * N * n_embd
    bdh_state_bytes = bdh_state_elements * state_dtype_bytes

    vb_state_bytes = None
    vb_int8_base_delta_state_bytes = None
    if d_state is not None:
        vb_state_elements = n_layer * batch_size * n_head * N * d_state
        vb_state_bytes = vb_state_elements * state_dtype_bytes
        vb_int8_base_delta_state_bytes = vb_state_elements * 1 + vb_state_elements * state_dtype_bytes  # base (int8) + delta (full precision)

    t_layer = transformer_n_layer if transformer_n_layer is not None else n_layer
    t_head = transformer_n_head if transformer_n_head is not None else n_head
    t_head_dim = transformer_head_dim if transformer_head_dim is not None else head_dim
    transformer_kv_cache_bytes = 2 * t_layer * batch_size * t_head * context_length * t_head_dim * kv_dtype_bytes

    return {
        "context_length": context_length,
        "bdh_state_bytes": bdh_state_bytes,
        "vb_state_bytes": vb_state_bytes,
        "vb_int8_base_delta_state_bytes": vb_int8_base_delta_state_bytes,
        "transformer_kv_cache_bytes": transformer_kv_cache_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=683)
    parser.add_argument("--transformer-layers", type=int, default=None, help="Override the Transformer's own layer count independent of --n-layer (BDH/VB). Matched-parameter-count configs often need different layer/head counts per architecture -- e.g. this session's real Phase F comparison used 8 layers/8 heads for BDH/VB but 6 layers/4 heads for the Transformer to hit the same ~25.5M parameter target. Defaults to --n-layer for backward compatibility.")
    parser.add_argument("--transformer-heads", type=int, default=None, help="Override the Transformer's own head count independent of --n-head (BDH/VB). Defaults to --n-head.")
    parser.add_argument("--transformer-head-dim", type=int, default=None, help="Override the Transformer's head_dim independent of n_embd // n_head. Defaults to n_embd // transformer_heads.")
    parser.add_argument("--d-state-divisor", type=int, default=4, help="HZ-Core-2's VB d_state = n_embd // this. Default 4 matches the locked Pareto choice, docs/restart/hz0h_phase_b_vb_sweep_results.md.")
    parser.add_argument("--merge-every-k", type=int, default=32, help="HZ-Memory mode's base+delta INT8 state merge interval. Default 32 matches the locked recommendation, docs/restart/hz0h_phase_d_base_delta_int8_results.md.")
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

    d_state = max(1, args.n_embd // args.d_state_divisor)
    vb_config = BDHVBConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0, d_state=d_state)
    vb_model = BDHVB(vb_config).to(device)
    vb_model.attn.freqs = vb_model.attn.freqs.to(torch.float32)
    vb_model.eval()

    transformer_layers = args.transformer_layers if args.transformer_layers is not None else args.n_layer
    transformer_heads = args.transformer_heads if args.transformer_heads is not None else args.n_head
    transformer_head_dim = args.transformer_head_dim if args.transformer_head_dim is not None else args.n_embd // transformer_heads
    transformer_config = MatchedTransformerConfig({"vocab_size": args.vocab_size, "d_model": args.n_embd, "num_layers": transformer_layers, "num_heads": transformer_heads, "head_dim": transformer_head_dim, "d_ff": args.d_ff, "use_rope": True})
    transformer_model = MatchedTransformerLM(transformer_config).to(device)
    transformer_model.eval()

    bdh_params = sum(p.numel() for p in bdh_model.parameters())
    vb_params = sum(p.numel() for p in vb_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    N = args.n_embd * args.mlp_internal_dim_multiplier // args.n_head

    results: dict = {
        "device": str(device), "bdh_parameter_count": bdh_params, "vb_parameter_count": vb_params, "transformer_parameter_count": transformer_params,
        "vb_d_state": d_state, "vb_merge_every_k": args.merge_every_k,
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
        vb_prefill = measure_vb_prefill(vb_model, prompt, args.prefill_repeats, device)
        vb_prefill["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        vb_decode_streaming = measure_vb_decode_streaming(vb_model, prompt, args.decode_tokens, device)
        vb_decode_streaming["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        vb_decode_int8_base_delta = measure_vb_decode_int8_base_delta(vb_model, prompt, args.decode_tokens, device, args.merge_every_k)
        vb_decode_int8_base_delta["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_prefill = measure_transformer_prefill(transformer_model, prompt, args.prefill_repeats, device)
        transformer_prefill["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_decode_naive = measure_transformer_decode_naive(transformer_model, prompt, args.decode_tokens, device)
        transformer_decode_naive["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_decode_kv_cache = measure_transformer_decode_kv_cache(transformer_model, prompt, args.decode_tokens, device)
        transformer_decode_kv_cache["peak_memory_bytes"] = peak_memory_bytes(device)

        state_bytes = compute_state_bytes(
            batch_size=1, n_layer=args.n_layer, n_head=args.n_head, N=N, n_embd=args.n_embd,
            context_length=context_length, d_state=d_state, head_dim=args.n_embd // args.n_head,
            transformer_n_layer=transformer_layers, transformer_n_head=transformer_heads, transformer_head_dim=transformer_head_dim,
        )

        results["by_context_length"][context_length] = {
            "bdh_prefill": bdh_prefill,
            "bdh_decode_naive_replay": bdh_decode_naive,
            "bdh_decode_streaming_state": bdh_decode_streaming,
            "bdh_decode_kv_cache": bdh_decode_kv_cache,
            "vb_prefill": vb_prefill,
            "vb_decode_streaming_state_speed_mode": vb_decode_streaming,
            "vb_decode_int8_base_delta_state_memory_mode": vb_decode_int8_base_delta,
            "transformer_prefill": transformer_prefill,
            "transformer_decode_naive_replay": transformer_decode_naive,
            "transformer_decode_kv_cache": transformer_decode_kv_cache,
            "state_bytes": state_bytes,
        }

    output_text = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
