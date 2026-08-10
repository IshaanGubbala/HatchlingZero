# HZ-0I BDH improvement roadmap

The objective is no longer an immediate Qwen leaderboard comparison. Keep
developing the ~0.3B BDH successor until it delivers the intended advantages:
knowledge density, persistent state, explicit memory, conditional compute,
plasticity, sparse capacity, and faster training per active FLOP.

## Mechanism lanes

1. **Persistent-state stability:** chunked O(1) state carry, RMS/scale control,
   int8/BF16 state storage with measured drift, and state checkpoint/resume.
2. **Knowledge density:** domain-balanced audited corpus, replay of rare code/math/
   JSON/terminal domains, and long-context retrieval probes.
3. **Conditional compute:** triggered attention must compute only triggered Qs;
   routed MoE must enforce capacity and report expert utilization.
4. **Session adaptation:** HZ-0B memory and HZ-0D fast weights are added only
   through explicit session-local state, never by silently changing base weights.
5. **Training speed:** MLX compile, grouped projections, fused state updates,
   BF16 activations, and no unnecessary host synchronization.
6. **Scale ladder:** improve the 0.3B profile first; only then revisit 0.8B–5B.

The faithful BDH oracle remains a control. Every algorithmic change gets a
matched control and a report of CE, active FLOPs, state bytes, throughput,
long-context behavior, and expert/trigger statistics.


## Completed improvement slices

- Chunked stateful training now carries the real BDH state across bounded chunks.
- A deterministic domain-balanced sampler now supports explicit weighting of general text, code, math, JSON, documentation, and terminal data.

- Added adaptive loss-driven domain sampling and an experimental top-k latent
  sparsity path; both remain behind explicit experiment flags.

- Added salient latent-token memory writes with explicit session reset.

- Added experimental weight tying, measured at 1.30x training speed and 42.1% fewer parameters on a vocabulary-dominated probe.

- Added low-rank factorized BDH projections; rank-16 probe measured 1.29x speedup.

- Added explicit leaky state retention for bounded plastic persistent memory.

- Combined factorized projections and tied cosine vocabulary head: 110.9M 0.3B-profile parameters.

- Added portable persistent-state checkpoint/resume for compact factorized BDH.

- Added grouped-head factor sharing as an optional capacity/speed tradeoff.

- Added layerwise conditional-attention/fast-weight/MoE composition hook.


## Current evidence checkpoint

The 0.3B compact factorized+tied BDH completed a 256k-token adaptive six-domain
MPS continuation (loss 10.738 to 7.040; 464.9 tok/s) with improved held-out CE
in every domain. This is the current development baseline; capability-layer and
int8-state variants remain separate ablations.

- Added rank-768 capacity-preserving factorized+tied 0.3B profile.

- Preferred full-capacity candidate is rank-704 untied factorized BDH after 100-step evidence.
