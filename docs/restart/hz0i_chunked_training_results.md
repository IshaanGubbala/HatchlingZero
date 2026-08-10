# HZ-0I chunked BDH training

Added `chunked_next_token_loss` and `chunked_stream_logits`. They carry the
actual BDH outer-product state across chunks, preserve full-vs-streaming logits
to `2e-5`, and permit bounded-memory training instead of materializing a full
quadratic attention matrix.

On a 512-token, 32-wide probe, chunk size 64 was finite and measured 8.69ms
versus 9.02ms for full forward. The speed result is not the main claim; the
important benefit is bounded activation/state working set for long sequences.
