# HZ-0I I3: HZ-0B memory read-only adapter

Implemented `HZ0IBDHMemory` in `reference/hz0i_bdh_model.py`. It projects the
actual BDH residual representation into the HZ-0B key space, reads an immutable
`MemoryState` using the audited Torch simulator, projects the value back, and
applies a confidence-independent gated residual contribution. The adapter
exposes no write/update path.

The real gate passes: finite logits and exact preservation of the supplied
memory state (`tests/reference/test_hz0i_bdh_model.py`). This is a mechanism and
state-isolation gate only. It does not claim quality improvement; write training
and matched BDH-with-memory controls remain open.
