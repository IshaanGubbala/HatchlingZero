# HZ-0I knowledge density infrastructure

`reference/hz0i_knowledge_sampler.py` adds deterministic weighted sampling across
the audited domain streams. It is designed to prevent the general-text majority
from drowning code, mathematics, JSON/configuration, documentation, and
terminal knowledge. It has not yet changed a canonical training run; weights
and replay ratios remain explicit experiment parameters.


Added `AdaptiveKnowledgeSampler`: domain weights can be updated from per-domain
loss EMAs, upweighting difficult/underlearned knowledge while preserving an
explicit temperature and reproducible seed. This makes knowledge density an
active training policy rather than a fixed mixture guess.


The sampler now deduplicates repeated packed rows within each domain and applies
a configurable adaptive-sampling floor (default 5% per domain). Hard-domain
replay cannot silently eliminate code, math, JSON, docs, or terminal knowledge.
Sampler checkpoints preserve this floor and all adaptive state.
