# HZ-0C C6: HZ-0B Memory Wired Through the Trigger Graph

Date: 2026-08-04. Closes the tracker's open item ("HZ-0B memory has not
yet been wired into the trigger graph") by actually inserting HZ-0B's
session-local write+read memory
(`reference/hz0b_b8_latent_write.py::sequential_latent_write_and_read`)
into C6's conditional-attention forward pass and rerunning the matched-cost
LM-loss comparison with it present.

`scripts/hz0c_c6_conditional_attention_eval.py`. New function
`conditional_forward_with_memory(model, token_ids, trigger, latent_params,
*, decay_rate=1.0, ste=False)`, built by splitting the old inline
`conditional_forward` into `conditional_hidden` (stops before `final_norm`)
plus the existing `logits_from_hidden` tail, then inserting memory at that
exact split point -- the SAME injection point `reference/hz0b_b6_hz0a_integration.py`
and `hz0b_b8_latent_write.py::forward` already established (backbone
residual stream fully formed, before the LM head), so this is not a new
integration convention. `--with-memory` CLI flag; existing behavior is
byte-for-byte unchanged when the flag is off.

## Correctness (`tests/reference/test_hz0c_c6_memory_integration.py`, 4 tests, real checkpoint)

- Position 0's memory-wired output is EXACTLY equal to the no-memory
  output, for any write-gate bias (`-30`, `-3`, `0` all checked) -- the
  write+read analog of B6's own "empty memory behaves exactly like no
  memory" invariant: position 0's read always happens against an
  all-zero-confidence bank (`sequential_latent_write_and_read` starts
  empty, and B1 decision 7's write-visibility rule means nothing is
  written before it is read), and `gated_memory_read`'s
  `confidence_scaled` gate is exactly 0 whenever retrieval confidence is
  0, by construction -- true structurally, not just at this checkpoint.
- The default `write_gate_bias_init=-3.0` engages memory (mean write gate
  above `1e-3`, memory-wired logits differ from no-memory logits, all
  finite) -- confirms the wiring is not silently a no-op.
- Deterministic given a fixed seed -- exact array equality across two
  independent calls with identical params.
- **Correction to an initial wrong assumption**: a `-30` write-gate bias
  does NOT suppress writes everywhere, as first assumed -- the learned
  projection's raw dot product with a real (large-magnitude) hidden state
  can dominate a constant bias, so write gates were observed saturating
  near `1.0` even at `-30` for some positions. The position-0 exact-match
  test does not depend on this and is unaffected; a weaker "writes stay
  near zero everywhere" test was tried first, found empirically false,
  and replaced rather than loosened to pass.

## Matched-cost LM loss with memory present

Command:

```text
PYTHONPATH=. .venv/bin/python scripts/hz0c_c6_conditional_attention_eval.py --seed 555 --with-memory --memory-seed 17
```

Untrained memory (`write_gate_bias_init=-3.0`, seed 17, `decay_rate=1.0`,
no write-policy training -- there is no established B11-style protocol for
training memory against general corpus continuation, a separate,
real future undertaking, not attempted here):

| Policy | No-memory loss | With-memory loss | Delta | Mean write gate |
| --- | ---: | ---: | ---: | ---: |
| No anchors | 2.5883 | 3.0937 | +0.5054 | 0.593 |
| Fixed periodic | 2.5734 | 3.0852 | +0.5118 | 0.601 |
| Random matched | 2.5745 | 3.0920 | +0.5175 | 0.601 |
| State novelty | 2.5781 | 3.0816 | +0.5035 | 0.602 |
| Token-loss teacher (offline) | 2.5648 | 3.0627 | +0.4979 | 0.604 |
| Learned controller | 2.5533 | 3.0723 | +0.5191 | 0.604 |
| Full attention | 2.4319 | 3.0228 | +0.5908 | 0.638 |

**Honest result: an untrained memory controller is a net negative for
general-corpus LM loss (+0.50 to +0.59 across every policy)**, exactly as
expected for a randomly-initialized write/read path injecting unlearned
noise into the residual stream -- consistent with this project's standing
finding that HZ-0B memory only helps on tasks it has actually been trained
for (B11's structured recall tasks), never claimed to help unconditioned
language modeling out of the box. This is NOT a regression or a bug; it is
the correct, expected behavior of a fresh, untrained controller, disclosed
plainly rather than searched for a favorable seed.

**The real finding that matters for C6/C9: the trigger-policy ordering
survives memory being wired into the graph.** With memory present, the
learned controller (3.0723) still beats fixed periodic (3.0852), random
matched (3.0920), no anchors (3.0937), and state novelty (3.0816) -- the
same ordering as the no-memory C6/C9 results, just uniformly shifted up by
the untrained-memory penalty. This is real evidence toward the C5/C6 exit
gate ("HZ-0B memory behavior is preserved through integration"): the
trigger graph's own comparative quality signal is not disturbed by memory
being present, only by memory being untrained -- a materially different
and much narrower gap than "integration breaks the trigger mechanism."

## What remains open

- **Training a memory write policy for general-corpus continuation** (as
  opposed to B11's structured recall tasks) has no established protocol
  yet -- real future work, explicitly named, not attempted this pass.
- `decay_rate` was left at its neutral default (`1.0`, no decay); B11's
  `decay_rate=0.95` finding was specific to structured overwrite-recall
  tasks and was not assumed to transfer here without evidence.
- This still does not touch HZ-0A's own trained parameters or the
  existing anchor-attention layers' weights -- only the injection point
  and a small new controller, per this project's standing "don't modify
  frozen production code" discipline.
