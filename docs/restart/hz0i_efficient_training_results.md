# HZ-0I efficient BDH training

The compact rank-256 + tied-vocabulary model completed a real target-scale
knowledge-dense training probe:

- Parameters: `110,886,913`
- 30-step packed-corpus probe: loss `10.633 -> 9.922`, 44.6 tok/s, finite
- 20-step adaptive six-domain probe: loss `10.722 -> 9.850`, 67.3 tok/s,
  with general/code/math/JSON/docs/terminal samples all present

The short-sequence throughput is not a final benchmark, but it confirms the
compact model can train with the knowledge-density machinery rather than only
passing isolated forwards.
