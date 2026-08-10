# HZ-0I factorized BDH projections

Added an experimental low-rank projection path in
`reference/hz0i_factorized_bdh.py`. Encoder, value encoder, and decoder are
factorized per head, reducing dense `D x N` matrices while preserving the BDH
outer-product and multiplicative latent interaction.

On a 96-wide probe with rank 16:

- Parameters: `5,603,328 -> 4,884,480` (12.8% reduction)
- Ten-step training: `104.9ms -> 81.4ms` (1.29x speedup)
- Forward/backward finite

At the 0.3B profile, rank 256 is expected to remove a large fraction of the
three internal projection matrices, but quality and rank sensitivity are not
validated yet. The dense BDH oracle remains the control.


Added `FactorizedBDH.from_dense`, which SVD-initializes low-rank factors from a
dense BDH checkpoint instead of starting from unrelated random factors. This
provides a warm-start path for rank ablations and avoids conflating
factorization loss with initialization loss.
