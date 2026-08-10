# HZ-0I salient latent memory writes

Added `BDHSalientMemoryWriter`, an explicit session-local memory policy. It
selects the highest-energy latent token per sequence, writes a detached key/value
into the HZ-0B immutable memory state, then reads memory back into BDH hidden
states through a learned gate. Session state is returned explicitly and can be
reset; it never silently changes model weights.

Write/read and reset tests pass. This is an inference/session mechanism, not yet
a trained knowledge-quality claim; next validation is long-context retrieval and
write precision under real domain streams.


The writer now supports configurable multi-write chunks. `writes_per_sequence=3`
stores the three highest-energy latent positions into distinct memory slots in
one pass, allowing denser session knowledge than a single summary write. The
write count is tested explicitly.


Memory writes now accept an explicit trigger mask. When supplied, only triggered
positions can be selected for writes, making supervised fact/summary storage
possible without relying on hidden-state energy as a proxy. The writer rejects
sequences with too few triggers rather than silently degrading.


Added `memory_reconstruction_loss`, an auxiliary training objective that writes
key/value facts into explicit memory and penalizes retrieval reconstruction. This
provides a direct learnable pressure for memory fidelity instead of assuming
next-token loss alone will teach session storage.


Writes now have an explicit `detach_writes` policy. Inference defaults to
detached session state to prevent graph growth; auxiliary memory training can
set `detach_writes=False` and receive gradients through key/value writes.
