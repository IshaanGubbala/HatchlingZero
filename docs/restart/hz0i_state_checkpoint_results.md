# HZ-0I persistent state checkpointing

Added explicit serialization helpers for BDH state, stream position, and an
optional model fingerprint. Factorized BDH now has a matching state initializer,
so compact models can pause/resume without dense-model assumptions.

Round-trip and factorized irregular-chunk resume tests pass. Checkpoints store
state tensors detached on CPU and restore to an explicitly requested device/dtype.


Checkpoint serialization now supports packed int8 `QuantizedState` objects,
including values, scales, shape, and dtype metadata. Quantized long-context BDH
streams can therefore be paused without expanding back to BF16/FP32 on disk.
