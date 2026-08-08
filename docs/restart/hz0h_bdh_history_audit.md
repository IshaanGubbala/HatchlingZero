# HZ-0H H0: Dragon Hatchling (BDH) provenance and architecture audit

Date: 2026-08-08. First HZ-0H work, run per the plan's own H0-H2-can-proceed-in-isolation rule (`plans/HZ-0H_BDH_Reconciliation_Plan.md`) while HZ-0G's G5 comparison work is being scoped separately. Touches nothing in the canonical HZ backbone.

## Sourcing, honestly

This audit is built from two fetched, AI-summarized sources, not an independent line-by-line read of the primary PDF or source code:

1. arXiv 2509.26507's HTML rendering (`https://arxiv.org/html/2509.26507`), fetched and summarized.
2. The official code repository README (`https://github.com/pathwaycom/bdh`), fetched and summarized.

This is a real limitation, not hidden: section-level claims below (e.g. "Section 3.2, Eq. 8") are as reported by the fetch tool's summarization pass, not independently re-verified against the raw PDF. H1's actual implementation work will require reading `bdh.py` directly, not relying on a summarized README. Treat every claim below as `paper-defined` at the granularity a summarization pass can support -- real enough to plan H1 against, not yet rigorous enough to implement blindly from without checking the primary source at each step.

## Paper identity

- **Title**: "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"
- **Authors**: Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz
- **Submitted**: 2025-09-30. **arXiv**: 2509.26507
- **Official code**: `github.com/pathwaycom/bdh`, MIT license, `bdh.py` (model) + `train.py` (training script, "toy dataset" per the README's own words -- not the paper's full training setup)

## The one claim independently confirmed against the primary source, matching the plan's own stated caution

The plan's constraints section states: *"The later internal Sudoku result is not evidence for the public BDH baseline."* The official repo's own README confirms this directly, in its own words: *"the Sudoku Extreme result refers to Pathway's internal BDH implementation, not to the current open-source repository... does not reproduce the 97.4% benchmark result out of the box."* Real, sourced, not assumed -- the plan's caution was correct and is now directly cited.

## Claimed architecture, labeled per H0's own taxonomy

| Claim | Label | Note |
| --- | --- | --- |
| Core mechanism: edge-reweighting dynamics on a graph of neuron-particles, state on edges (synapses) | `paper-defined` | Abstract + Section 2 |
| BDH-GPU: three O(n·d) shared matrices `Dx`, `Dy` (project to/from n-dim neuron space) and `E` (edge-reweighting/synaptic dynamics), `d≈256` | `paper-defined` | Section 3.2, Eq. 8 per the summarized fetch -- NOT yet confirmed against `bdh.py`'s actual variable names |
| Positive-latent activations `y ∈ R^n`, ReLU-enforced non-negativity, ~5% empirical sparsity, no explicit L1 | `paper-defined` | Section 6.4 |
| "ReLU-lowrank" feedforward block (low-rank matrix products with ReLU thresholds) | `paper-defined` | Section 5.2 |
| Linear attention with Q=K (positive keys via LSH into the positive orthant), `output = (K^T V) / (K^T 1)` | `paper-defined` | Section 6.1 |
| No explicit RoPE -- positional information implicit in state evolution | `paper-defined` (absence claim) | Needs code confirmation; summarization passes are unreliable at confirming absences |
| Persistent edge-state `sigma(t)`, update `sigma(t+1) := Phi(M, sigma(t), a_t)`, "synaptic weights," parameter count = state size (O(n·d)) -- a fast-weights framing | `paper-defined` | Section 3.2 |
| No explicit cross-layer weight tying reported; standard per-layer stacking with per-layer `E`/`Dx`/`Dy` | `paper-defined` (tentative) | Section 4.1, thin summarized coverage -- verify directly before relying on this for H1 |
| Hebbian update: `Y(i), X(j) -> sigma(i,j)`, co-activation strengthens the synapse | `paper-defined` | Eq. 2, Section 1.2 |
| "Monosemantic synapses" -- individual edges empirically correspond to consistent concepts across prompts | `empirical paper finding` | Section 6.3 |
| Integrate-and-fire spiking formalization (not conductance-based Hodgkin-Huxley) | `paper-defined` | Section 2.5 |
| Model scales tested: 10M-1B params, compared against GPT-2-architecture Transformers at matched param counts | `empirical paper finding` | Section 4.2 -- exact table values not extracted, need the real PDF table for H3's baseline design |
| "Rivals GPT2 performance on language and translation tasks" | `empirical paper finding` | Abstract; no concrete loss/perplexity numbers extracted yet |
| Graph structure: high modularity, heavy-tailed (scale-free) degree distribution, read directly from trained parameter matrices | `empirical paper finding` | Section 5.5 |
| Sudoku Extreme 97.4% result | `Pathway internal-only` | Confirmed directly from the official repo's own README, not just the paper -- do not use as public-baseline evidence, per the plan's own rule |

## `official-code implemented`: verified directly against `bdh.py` (raw source, not a summarized README)

Fetched `https://raw.githubusercontent.com/pathwaycom/bdh/main/bdh.py` directly and read the actual code. Two of the paper-summary's tentative claims above were **wrong** and are corrected here -- exactly the reason those claims were labeled tentative rather than trusted:

| Real code finding | Corrects |
| --- | --- |
| `class BDH(nn.Module)`, config-driven (`BDHConfig`). Four real parameters: `self.decoder` (shape `(nh*N, D)`), `self.encoder` (`(nh, D, N)`), `self.encoder_v` (`(nh, D, N)`), `self.lm_head` (`(D, vocab_size)`), where `N = D * mlp_internal_dim_multiplier // nh`. These are the code's real names for what the paper's prose calls the shared matrices -- **not literally named `E`/`Dx`/`Dy`** in this implementation. | The paper-summary table's literal matrix names were the paper's own notation, not the code's variable names -- both are `paper-defined`/`official-code implemented` respectively, and they don't share names. |
| Forward pass, per layer: `x_latent = x @ encoder` -> `x_sparse = relu(x_latent)` -> `yKV = attn(Q=x_sparse, K=x_sparse, V=x)` -> `y_latent = yKV @ encoder_v` -> `xy_sparse = x_sparse * y_sparse` -> `yMLP = xy_sparse @ decoder` -> `x = ln(x + y)`. | Confirms the paper's ReLU-sparse-latent framing concretely, at the level of real tensor ops. |
| `Q=K` is not just a design description -- the code literally asserts it: `assert K is Q` inside the attention call. | Confirms this claim at `official-code implemented` confidence, not just `paper-defined`. |
| **RoPE is present**: `self.rope()` applies rotary position embeddings via `get_freqs()`. | **Corrects** the earlier tentative "no explicit RoPE" claim, which was wrong -- flagged as unreliable in the first version of this doc precisely because summarization passes are bad at confirming absences, and that caution was warranted. |
| **No variable named `rho` or `sigma` anywhere in this file.** The paper's "persistent edge-state" framing describes what the encoder/decoder loop's math is equivalent to (a fast-weights/state-space reinterpretation), not a literal named state object carried between forward calls in this specific file. | Refines, doesn't contradict -- `bdh.py` is the GPU tensor-formulation ("BDH-GPU"), which the paper itself describes as a mean-field/tensor reformulation of the literal graph dynamics; the state-space equivalence is likely proven mathematically (Section 3) rather than implemented as an explicit stateful object in code. Needs H2's own streaming-equivalence work to confirm this precisely, not assumed here. |
| **Weights ARE tied/shared across all layers**: the same `self.encoder`/`self.encoder_v`/`self.decoder` Parameter objects are reused inside `for level in range(C.n_layer)` -- one shared set of matrices for the whole depth, not per-layer instances. | **Corrects** the earlier tentative "no explicit weight tying reported" claim, which was wrong. This is a real, load-bearing architectural fact for H1's faithful port and H4's "shared vs untied depth weights" ablation -- H4 should treat "shared" as the paper's actual baseline, not a variant to test against an assumed-default "untied" baseline. |

## `BDHConfig` real defaults, verified directly against raw `bdh.py`

```python
@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
```

`N` (the sparse-latent width) = `mlp_internal_dim_multiplier * n_embd // n_head` = `128 * 256 // 4 = 8192` at these defaults. `vocab_size=256` is a byte-level default, matching `train.py`'s own toy setup below, not the paper's real 10M-1B-scale experiments.

## `train.py` real defaults, verified directly

Dataset: tiny Shakespeare (`raw.githubusercontent.com/karpathy/char-rnn/.../tinyshakespeare/input.txt`) -- a toy smoke-test script, not the paper's actual scaling-law training runs. Optimizer: `AdamW`, `lr=1e-3`, `weight_decay=0.1` (betas not specified in the fetched excerpt). `BATCH_SIZE=32`, `BLOCK_SIZE=512`, `MAX_ITERS=3000`, dtype `bfloat16`/`float16` depending on hardware. This script is a wiring smoke test, not a source for H3's real matched-scale configs.

## Real, disclosed gap: Section 4.2's scaling-law table

Three separate real attempts to extract Section 4.2's actual parameter-count/loss table -- arXiv's HTML rendering (twice, including a v1-pinned URL), the raw PDF (rejected: exceeds the fetch tool's 10MB limit), and the Hugging Face papers mirror -- all failed to surface the actual numbers; the table content does not survive conversion in any of these paths (likely a LaTeX table or figure that summarization-based extraction can't parse). The paper states models range "10M to 1B" parameters and that BDH-GPU "generally compares favorably to the Transformer... even on... translation," but no concrete loss/perplexity/config values were recovered.

This is not needed for H1 (mechanism-level parity at any small scale) or H2 (streaming equivalence, also scale-independent). It IS needed for H3's honest matched-scale design, but H3 is blocked on the HZ-0G G1-G5 decision regardless -- getting the real table (via direct PDF page access, or asking a human to paste the relevant page) can happen later, right before H3 actually starts, without blocking H1/H2 now.

## What H0 does not yet establish

- Section 4.2's real scaling-law table (see above) -- deferred to just-before-H3, not blocking H1/H2.
- Community MLX port exists (`github.com/severian42/BDH-MLX`) -- not evaluated, not to be trusted as a stand-in for `reference/hz0h_bdh_mlx.py`'s own from-scratch, tested implementation per H1's own requirement ("no component enters HZ-1 without a predeclared metric and a fair control" applies equally to trusting a third-party port uncritically).
- Community Hugging Face Transformers port also exists (`jploski/bdh-transformers`) -- same caveat.

## H0 status: sufficient to start H1

Real architecture (four named parameters, per-layer forward sequence, `Q=K`, RoPE, shared depth weights) and real default config values are both now verified directly against source, not guessed from paper prose. H1 can proceed.
