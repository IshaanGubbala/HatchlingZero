# HZ-0E E7: Interaction Guards

Date: 2026-08-05. The initial E7 interaction contract is now locked by
`tests/reference/test_hz0e_e7_interactions.py` plus the existing D7 ordering
tests.

- MoE routing has no trigger argument and repeated routing on identical hidden
  states is bit-identical.
- E6 target layers `27, 28, 30` are disjoint from HZ-0D fast-weight attention
  layers `4, 9, 14, 19, 24, 29`.
- The E6 integration API accepts no HZ-0C trigger or HZ-0D fast-state input,
  preventing an accidental feedback path.
- Existing D7 tests continue to enforce one memory write per token and no
  same-call fast-update feedback.

Fresh combined verification: **7 passed** (`4` E6 integration and `3` E7
interaction tests). This closes the structural E7 guard; PMetal dispatch and
specialization quality remain later E9/E8 work.
