# Phase F RAM/speed target-gate results

Date: 2026-08-14

Source report:
`outputs/hz0h_phase_f_chunked_long_context/hz0h_phase_f_chunked_long_context.json`

Hardware: RTX 3060, CUDA, BF16, batch 1, seed 7, matched ~25M-parameter
configurations. This is an **execution-only** result: the benchmark uses
untrained weights, so it is not a quality-matched architecture claim.

## Automated gate

The report was evaluated with:

```text
scripts/hz0h_phase_f_target_gate.py
```

Thresholds: total parameter ratio ≤1.01, peak decode RAM ratio ≤0.70, and
decode throughput ratio ≥1.30 relative to the Transformer KV-cache path.

| Context | Candidate | Params ratio | Candidate peak RAM | Transformer peak RAM | RAM ratio | Candidate tok/s | Transformer tok/s | Speed ratio | RAM gate | Speed gate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 8,192 | VB BF16 speed | 1.0085 | 712.6 MB | 695.0 MB | 1.025 | 167.0 | 157.1 | 1.063 | FAIL | FAIL |
| 16,384 | VB BF16 speed | 1.0085 | 716.8 MB | 1,630.4 MB | 0.440 | 168.7 | 102.6 | 1.645 | PASS | PASS |
| 32,768 | VB BF16 speed | 1.0085 | 725.3 MB | 4,709.2 MB | 0.154 | 174.3 | 56.9 | 3.063 | PASS | PASS |

Exact BDH streaming at 32,768 also passes the raw execution thresholds
(about 1.14 GB vs 4.71 GB and 188.3 vs 56.9 tok/s), but VB is the intended
memory-efficient candidate.

## Interpretation

The 30% RAM/30% speed target is **real at 16K and 32K streaming decode**, but
not at 8K. It is not yet a general superiority claim because:

- this is one seed and one hardware configuration;
- the benchmark uses untrained weights;
- quality and contamination-checked capability are not evaluated here;
- training remains slower and more memory-intensive for BDH-family models;
- prefill and total end-to-end application latency are separate axes.

No report should state simply that “BDH is 30% better” without specifying the
context, decode path, hardware, precision, and the missing quality gates.
