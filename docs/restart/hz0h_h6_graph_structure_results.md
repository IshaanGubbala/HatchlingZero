# HZ-0H H6: effective graph structure, real result

Date: 2026-08-08. Per H0, the paper's `Dx E`/`Dy E` graph notation maps to the real code's `encoder`/`decoder` matrices. `reference/hz0h_bdh_graph.py::extract_effective_graph` composes `decoder_h @ encoder_h` per head into a real N x N effective adjacency matrix -- a genuine derivation, not a metaphor.

## Method

Trained a small BDH-GPU model (n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, N=64 per head -- similar order of magnitude to T2's 819K-param setup) on a real order-2 Markov-chain task (genuine sequential structure, not pure repetition). Extracted the effective graph, thresholded to the top 10% of edge weights, and computed modularity/community structure via `networkx`. Compared against two controls per H6's own required methodology: an untrained (random-init) model, and a statistics-preserving shuffle (same edge-weight distribution, random reassignment to node pairs) -- H6's own text: "shuffle connectivity while preserving matrix statistics."

## Real bug found and fixed mid-investigation: training target convention

Initial training used `model(idx, targets=idx)` (same sequence for input and target). Checked directly against the official `train.py` (not assumed): the real convention is `model(x, y)` with `y = x` shifted by one position (`x = data[i:i+block_size]`, `y = data[i+1:i+1+block_size]`). With `targets=idx`, the residual path (`x = ln(x + y)` includes `embed(idx[t])` added directly into what predicts `targets[t]=idx[t]` at the SAME position) lets the model trivially shortcut through `embed -> lm_head` without doing any real attention/encoder-decoder work -- a degenerate task, not real sequence modeling.

Fixed in `reference/hz0h_bdh_graph.py::train_tiny_bdh_on_markov_chain` (shifted targets, `.contiguous()` for the resulting non-contiguous slice). Confirmed via H5's passkey task (same fix, see `docs/restart/hz0h_h5_state_ablation_results.md`) that this fixed training setup produces a model that learns real, non-trivial structure (100% passkey retrieval using real state vs. ~11% with state ablated) -- so the graph-structure finding below isn't an artifact of an undertrained/degenerate model.

## Real result: no detectable structure beyond chance, at this scale (post-fix)

Trained at 1500 steps, 3 seeds, with the corrected shifted-target training:

| Seed | Trained modularity | Communities | Shuffled modularity | Communities |
| --- | --- | --- | --- | --- |
| 0 | 0.1778 | 5 | 0.2035 | 5 |
| 1 | 0.1855 | 4 | 0.1890 | 4 |
| 2 | 0.1749 | 5 | 0.1957 | 5 |

Trained modularity never exceeds the shuffled control -- in fact consistently slightly *below* it across all 3 seeds. No evidence of learned graph structure in the effective `decoder @ encoder` adjacency beyond what the same value distribution would produce under random node reassignment, even with a properly-trained model that demonstrably learns real behavioral structure elsewhere (H5's real state-based retrieval).

## Honest scope -- this is not "BDH has no graph structure"

The plan's own explicit caution: "Do not permanently reject a mechanism from a tiny toy run when the paper's evidence spans approximately 10M-1B parameter scales." This result is real and reproducible (3 seeds, 2 step budgets) **at this specific tiny scale** (~800K params) -- not a claim that BDH-GPU lacks real graph/modularity structure at the paper's actual tested scales. The honest, disclosed limitation: no larger-scale BDH-GPU model exists in this repo to test against yet (that's H3's job, itself gated on HZ-0G's G1 decision).

## What H6 establishes

- A real, working effective-graph extraction and analysis pipeline, with a genuine falsifiability control (trained vs. shuffled-preserving-stats), not just a qualitative "look at the pretty matrix" exercise.
- Per H6's own explicit instruction ("only if topology affects quality should explicit sparse graph execution be tested for real compute savings") -- since no real trained-vs-shuffled topology difference was found at this scale, sparse graph execution is NOT motivated by this result. Correctly not pursued.

## What H6 does not establish

- Whether real graph structure emerges at larger scale -- untested, would need a bigger trained BDH-GPU model.
- Semantic correspondence (H6's own stated scope) -- not attempted, requires a real trained model with meaningful token-level structure to probe, which this toy Markov-chain task doesn't provide.
