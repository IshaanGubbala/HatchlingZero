# HZ-0I sparse latent BDH

Added an experimental top-k ReLU latent path (`reference/hz0i_sparse_bdh.py`).
At ratio 0.25, each latent head keeps only its largest quarter of activated
channels, preserving values and setting the rest exactly to zero. Forward and
backward finite tests pass.

This is intended to make BDH's sparse latent capacity a real systems benefit,
but the current Torch implementation still materializes dense projections;
actual FLOP/throughput benefit requires a grouped or block-sparse kernel. Quality
validation on the knowledge-dense mixture is still required.


Factorized BDH now accepts `latent_topk_ratio`, applying exact top-k ReLU sparsity
to both encoder and value latent paths in parallel and streaming execution.
The default remains dense ReLU; sparse mode is explicit until block-sparse
kernels and quality runs demonstrate a real throughput benefit.


A matched MPS kernel probe found no current speed benefit: dense latent training
ran `26,483 tok/s` versus top-k 25% `26,368 tok/s` on the 96-wide test. The
structural sparsity is real, but current dense/scatter kernels do not exploit it;
block-sparse kernels are still required before promoting top-k mode for speed.
