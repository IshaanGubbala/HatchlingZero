# HZ-0I I2: data/checkpoint runner adapter

`scripts/hz0i_bdh_stage_runner.py` now runs the BDH-centered model against the
audited JSONL token format and emits `native_metal_memory.jsonl` plus
`native_metal.json`, so the existing dashboard can consume its loss/throughput
series. Checkpoints include model state, AdamW state, step, cursor, and metric
history.

A real smoke ran 20 steps, resumed from its checkpoint, and completed at step
25. This is an adapter/infrastructure gate, not a quality comparison or a
claim that the Torch runner has replaced the optimized MLX/Metal path.


## Real corpus continuation

The adapter also completed 200 steps on the audited packed real-text corpus
(`data/packed/stage1_10m_train.jsonl`) with a 15.34M-parameter BDH topology. It
processed 12,400 tokens without NaN/Inf; mean loss over the first 20 records was
`9.3921`, versus `5.6068` over the last 20. This is still a short continuation
(12.4K tokens), but it validates the I2 runner on real data rather than only
synthetic rows.
