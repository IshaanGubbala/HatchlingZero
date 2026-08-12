# HZ-0I scale profiles: 0.8B to 5B

The tiny 10–15M tests are only mechanism bring-up. The immediate target is a
~0.3B BDH model that must earn a comparison against the closest public Qwen
reference (`Qwen3-0.6B`, 596M parameters) before any
0.8B–5B scale-up. The larger profiles remain future scale gates. These profiles use the faithful BDH parameterization
(`2*vocab*d_model + 3*multiplier*d_model^2`) and shared depth weights.

| Profile | d_model | latent multiplier | heads | layers | Parameters | BF16 state* | int8 state* |
|---|---:|---:|---:|---:|---:|---:|
| HZ0I-0.3B | 768 | 144 | 12 | 8 | ~0.292B | ~1.36GB | ~0.68GB |
| HZ0I-0.8B | 1024 | 256 | 16 | 8 | 0.856B | 4.29GB | 2.15GB |
| HZ0I-1B | 1280 | 160 | 16 | 8 | 0.849B | 4.19GB | 2.10GB |
| HZ0I-3B | 1536 | 384 | 24 | 8 | 2.79B | 14.50GB | 7.25GB |
| HZ0I-5B | 2048 | 320 | 32 | 8 | 4.13B | 21.47GB | 10.74GB |

`*`State is for batch 1, all layers, BF16, before optimizer/activations. The
large state footprint is a first-class engineering constraint, not omitted from
the parameter headline.

## Immediate 0.3B versus Qwen-0.8B gate

Do not advance to the 0.8B–5B profiles until the 0.3B BDH successor has a
real comparison against Qwen 0.8B. The comparison must use the same tokenizer
when possible, or report tokenizer effects separately; the same held-out corpus
and token budget; matched validation CE/perplexity; and separate measurements
for active FLOPs, peak memory, decode tok/s, and long-context loss.

A Qwen win must be judged per parameter and per active FLOP, not raw parameter
count. A BDH win must survive at least three seeds or a predeclared confidence
interval. The 0.3B run is the primary decision gate for HZ-0I.

## Required scale work

1. Keep the full-precision BDH oracle as the correctness reference.
2. Add chunked state carry and state checkpoint/resume at every profile.
3. Add BF16/8-bit state storage with measured output drift; never silently
   change state precision.
4. Use grouped/compiled projections and triggered attention so active FLOPs do
   not scale like dense full attention.
5. Validate first at 0.8B, then 1B, before attempting 3B/5B.
6. Compare against matched GDN-2 and Transformer controls at the same model
   budget and token budget.

The current I6 10M evidence is a mechanism gate. It does not extrapolate to
these profiles without the state-memory and systems gates above.
