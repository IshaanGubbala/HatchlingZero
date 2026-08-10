# HZ-0I speed and knowledge-density updates

- Added a knowledge-dense runner with explicit weighted replay across general,
  code, math, JSON, documentation, and terminal streams. A 10-step 15.3M
  smoke processed all domains with finite parameters and loss `10.142 -> 9.901`.
- Tested `torch.compile` on the BDH training step. On the tiny CPU probe, eager
  20-step training took 28.8ms and AOT-eager compile took 31.3ms. This
  hypothesis is rejected for that backend/shape; MLX `mx.compile` remains the
  promising speed path.


The runner now supports adaptive domain weighting. It computes per-example loss
from one forward pass, updates loss EMAs, and shifts future sampling toward
difficult domains without an extra model evaluation. A 5-step adaptive smoke
was finite (`10.142 -> 10.085`).
