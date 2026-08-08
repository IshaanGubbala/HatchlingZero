# HZ-0H H0: BDH component map

Date: 2026-08-08. Companion to `docs/restart/hz0h_bdh_history_audit.md` -- that doc has the full sourcing/provenance narrative and correction history; this one is the flat, structured component list H0 requires, for direct use as H1's implementation checklist. Every row cites which audit-doc finding it's based on rather than restating the reasoning.

| Component | What it is | Label | Real HZ analog (for H1's parity design, not a claim of equivalence) |
| --- | --- | --- | --- |
| Shared low-rank matrices (paper: `E`, `Dx`, `Dy`; code: `encoder` `(nh,D,N)`, `encoder_v` `(nh,D,N)`, `decoder` `(nh*N,D)`) | Project between the model dim `D` and a wider sparse-latent dim `N = D * mlp_internal_dim_multiplier // nh` | `paper-defined` (naming) + `official-code implemented` (shapes, verified against raw `bdh.py`) | Loosely analogous in role to HZ-0A's `in_proj`/`out` per-mixer projections, but BDH's are **shared across every layer** (see "shared depth weights" row) -- HZ-0A's are per-layer, not shared |
| ReLU-lowrank positive latent representation | `x_sparse = relu(x @ encoder)`, non-negative by construction, ~5% empirical sparsity per the paper (not code-verified) | `official-code implemented` (the ReLU/projection mechanics) + `empirical paper finding` (the 5% sparsity number) | No HZ analog -- none of A-E enforce non-negative activations anywhere |
| Q=K linear attention | `attn(Q=x_sparse, K=x_sparse, V=x)`, code literally asserts `K is Q` | `official-code implemented`, verified directly | Structurally unlike HZ-0A's `CausalAttention` (real, independent Q/K projections) and unlike `gdn2_fix`'s delta-rule recurrence (no attention at all in the recurrent layers) -- BDH's linear attention is a third, distinct mechanism, not a variant of either HZ mechanism |
| RoPE / positional encoding | Present: `self.rope()`, frequencies from `get_freqs()` | `official-code implemented`, verified directly -- **corrects** an earlier wrong "no RoPE" tentative claim, see audit doc | HZ-0A has no explicit positional encoding at all (recurrence provides implicit position-dependence; the 6 attention layers are also RoPE-free per `reference/hz0a_mlx_model.py::CausalAttention`) -- a real, structural difference worth naming for H3, not glossed over |
| Persistent per-layer `rho` (paper's edge-state framing) | No `rho`/`sigma` variable exists in `bdh.py` -- the state-space/fast-weights interpretation is a mathematical equivalence claim (paper Section 3), not a literal object in the GPU tensor code | `paper-defined` (the equivalence claim) -- **UNRESOLVED** whether it holds in the literal streaming sense H2 requires; not yet verified either way | HZ-0D's `FastWeightState` (`a_fast`/`b_fast`, real, literal, explicit state carried between calls) is the closest real analog -- if BDH-GPU's forward-pass-as-parallel-attention is truly state-space-equivalent to a streaming recurrence, H2 must prove it the same way D6/D7 proved HZ-0D's own state ordering, not assume it from the paper's claim |
| Shared depth weights | **Confirmed real**: the same `encoder`/`encoder_v`/`decoder` Parameter objects are reused across the entire `for level in range(C.n_layer)` loop -- one set of matrices for the whole model depth | `official-code implemented`, verified directly -- **corrects** an earlier wrong "no weight tying reported" tentative claim, see audit doc | HZ-0A's 31 blocks each have independent parameters -- a substantial, real architectural difference. H4's "shared vs untied depth weights" ablation should treat BDH's actual default (shared) as the thing being tested against an HZ-style untied control, not the reverse |
| Effective graph matrices (`Dx E`, `Dy E` per the paper) | Read directly from the trained parameter matrices post-hoc for interpretability analysis (H6), not a separate parameter set | `paper-defined` | No HZ analog -- H6 is BDH-specific interpretability work, not a comparison axis against any HZ mechanism |
| Hebbian co-activation update (`Y(i), X(j) -> sigma(i,j)`) | The literal graph-dynamics formulation (BDH proper, not BDH-GPU) -- relationship to the code's actual forward pass not yet traced | `paper-defined` -- **UNRESOLVED** how directly this maps to `bdh.py`'s tensor ops vs. being a separate, equivalent-by-proof formulation | No HZ analog -- none of A-E claim a Hebbian formulation |
| Spiking-neuron interpretation (integrate-and-fire, ReLU-threshold) | A biological reading of the same ReLU-sparse activations, not a separate mechanism | `paper-defined` | No HZ analog |

## Complete, precise forward-pass spec (verified directly against raw `bdh.py`, not summarized)

This supersedes the earlier compressed version in the audit doc -- gaps there (`y_sparse` appearing without a defining line, an ellipsis in the decoder line) are filled in here.

```
N = mlp_internal_dim_multiplier * D // n_head   (config: n_layer=6, n_embd(D)=256, n_head=4, mlp_internal_dim_multiplier=128, vocab_size=256, dropout=0.1 -- toy defaults)

x = ln(embed(idx).unsqueeze(1))            # (B, 1, T, D) -- the "1" is a broadcast axis for nh, not a literal per-head split of D
for level in range(n_layer):                # SAME encoder/encoder_v/decoder/ln reused every iteration -- shared depth weights, confirmed
    x_latent = x @ encoder                  # (B,1,T,D) @ (nh,D,N) -> (B,nh,T,N)
    x_sparse = relu(x_latent)
    yKV = attn(Q=x_sparse, K=x_sparse, V=x) # Attention, below
    yKV = ln(yKV)                           # extra LayerNorm the earlier summary missed
    y_latent = yKV @ encoder_v              # (B,nh,T,D) @ (nh,D,N) -> (B,nh,T,N)
    y_sparse = relu(y_latent)
    xy_sparse = drop(x_sparse * y_sparse)   # elementwise gate, then dropout
    yMLP = xy_sparse.transpose(1,2).reshape(B,1,T,N*nh) @ decoder   # (B,1,T,N*nh) @ (N*nh,D) -> (B,1,T,D)
    y = ln(yMLP)
    x = ln(x + y)                           # residual, then another shared-ln
logits = x.view(B,T,D) @ lm_head            # (B,T,D) @ (D,vocab) -> (B,T,vocab)
```

Attention (`Q is K` always, asserted in code):
```
freqs = get_freqs(N, theta=2**16) = 1 / (theta ** (floor(arange(N)/2)*2 / N)) / (2*pi)     # shape (1,1,1,N)
r_phases = arange(T).view(1,1,T,1) * freqs                                                  # (1,1,T,N)
QR = rope(r_phases, Q); KR = QR                    # rope: v*cos(phases) + rotate_half_pairs(v)*sin(phases)
scores = (QR @ KR.transpose(-1,-2)).tril(diagonal=-1)   # STRICTLY lower-triangular -- self-position excluded, confirmed
return scores @ V                                        # NO softmax, NO normalization anywhere -- confirmed by direct search for "sum(" / "/ (" near this code, found nothing
```

**Two precise, real discrepancies from the paper's own prose, both confirmed by direct code search, both must be preserved faithfully in H1's port rather than "corrected" toward the paper's simplified description:**

1. The paper's stated formula is `output = (K^T V) / (K^T 1)` (a normalized average). The real code has **no such division anywhere** -- raw `scores @ V`, scale presumably controlled entirely by the surrounding `ln()` calls instead.
2. `tril(diagonal=-1)` is **strictly** lower-triangular -- position `t` cannot attend to itself, only to positions `< t`. Standard causal masks (`diagonal=0`) would be a real, silent deviation from the official implementation if used without checking this.

## How this feeds H1

H1's faithful-reference implementation (`reference/hz0h_bdh_torch.py`, `reference/hz0h_bdh_mlx.py`) should port `bdh.py`'s actual four parameters (`encoder`, `encoder_v`, `decoder`, `lm_head`), the real per-layer forward sequence documented in the audit doc, the literal `Q=K` assertion, RoPE, and the shared-depth-weights structure -- all four are now `official-code implemented` confidence, not guesses. The `rho`/state-space equivalence claim and the literal Hebbian-to-tensor mapping remain open questions H1/H2 need to resolve by testing, not by assuming the paper's narrative describes the code 1:1.
