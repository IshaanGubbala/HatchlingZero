# HZ-0I grouped-head factorization

Added `GroupedFactorizedBDH`, which shares low-rank input/value factors across
groups of latent heads while preserving per-head right factors and BDH state.

On a 96-wide rank-16 probe with two groups:

- Parameters: `4,884,480 -> 4,878,336`
- 20-step time: `176.7ms -> 161.0ms`
- Final CE: `7.547 -> 7.606`

At the target 0.3B profile, rank-256 four-group factorization has 126.6M
parameters versus 129.8M ungrouped. The modest gain and small CE tradeoff mean
this remains optional; grouping must earn its place on longer runs.
