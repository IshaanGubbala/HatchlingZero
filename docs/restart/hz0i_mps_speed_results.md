# HZ-0I MPS training path

The MPS runner now supports dense-tied and factorized-tied 0.3B variants with
synchronized timing. On identical 10-step, 16-token target-scale probes:

| Variant | Params | Loss first -> last | tok/s |
|---|---:|---:|---:|
| Dense tied | 273.7M | 10.199 -> 6.863 | 7.08 |
| Rank-256 factorized+tied | 110.9M | 10.812 -> 7.987 | 184.7 |

The compact variant is approximately 26x faster in this short MPS probe. This
is a systems result, not a quality claim: the variants use different random
initializations and need matched long training before quality conclusions.
MPS float16 autocast remains disabled because it was slower/unstable in probes.


At sequence length 64, the factorized+tied model sustained `579.2 tok/s` for
20 MPS steps, confirming the short-sequence result is not only launch overhead.
