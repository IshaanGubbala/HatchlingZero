# HZ-0I I4/I5: optional integration primitives

Added Torch-side portable primitives for the experimental BDH shell:
conditional triggered attention residuals, bounded session-local low-rank fast
weights, and routed SwiGLU MoE. Correctness tests pass (3 tests). An explicit opt-in `HZ0IBDHIntegrated` shell now composes these components for
ablation tests; the base `HZ0IBDH` remains unchanged. Model-level quality,
active-compute, and long-context controls are required before any component is
retained. Four integration smoke tests pass.
