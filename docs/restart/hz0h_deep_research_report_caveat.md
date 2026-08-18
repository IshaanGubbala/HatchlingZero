# Caveat on `plans/deep-research-report(6).md`

That document is a general literature survey of sparse-GNN/BDH
optimization techniques (sparse formats, kernel fusion, structured
sparsity, MPS quirks, etc.) -- real, useful background reading, but
**its BDH pseudocode does not match this repo's actual, verified oracle**
(`reference/hz0h_bdh_torch.py`) and should not be treated as ground
truth for any kernel or math work here. Concretely:

- The report's pseudocode includes `A = A * v.unsqueeze(heads)` (an
  elementwise value gate) and a softmax-adjacent framing. The real
  oracle has neither: attention is `scores = (QR @ KR.mT).tril(diagonal=-1);
  return scores @ V` -- a plain matmul against `V`, no elementwise gate,
  no softmax, no `K^T @ 1` normalization anywhere. This is a real,
  deliberate, previously-documented property of the actual BDH-GPU
  source (`reference/hz0h_bdh_torch.py`'s own module docstring, and
  `docs/restart/hz0h_bdh_component_map.md`), not an oversight.
- The report describes `Y = ReLU(LayerNorm(A) @ D_y) * Q` (elementwise
  multiply by `Q`) folded directly into one step. The real oracle
  computes `x_sparse` and `y_sparse` via two separate ReLU'd projections
  (`self.encoder`, `self.encoder_v`) and only then does
  `xy_sparse = x_sparse * y_sparse` -- similar in spirit but not the
  same operation ordering/shape the report's pseudocode implies.
- The report's mask is described as applied after computing the full
  `QK^T`; the real oracle's mask is strictly lower-triangular with
  `diagonal=-1` (a position cannot attend to itself), a specific,
  previously load-bearing detail for causal-tile-skip kernel work this
  session already did (`reference/hz0h_bdh_triton_attention_torch.py`).

None of this makes the report's general optimization advice (sparse
formats, 2:4 structured sparsity, fusion priorities, MPS caveats,
benchmarking methodology) wrong -- that material is real and broadly
applicable. But any BDH-specific code sketch or tensor-shape claim in
that report should be re-derived from the actual oracle
(`reference/hz0h_bdh_torch.py`) before being trusted, exactly the same
discipline this session has applied to every other kernel change so far
(never touch the oracle, always parity-test new work against it
directly, not against a paraphrase of it).
