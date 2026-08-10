# HZ-0I target scale status

The target has been corrected from “10M bring-up” to 0.8B–5B model classes.
`specs/hz0i_scale_profiles.json` defines 0.856B, 0.849B, 2.79B, and 4.13B
profiles. The dominant new constraint is persistent BDH state: 4.19–21.47GB
BF16 for batch 1 across eight layers before activations/optimizer state.

An explicit int8 state-storage policy now exists in
`reference/hz0i_state_storage.py`; a random-state round trip has <2% relative
error in the regression gate. This is not yet approved for training—the next
step must measure state quantization drift on actual logits/loss and determine
whether per-head/per-block scales are required.

The 10M and 15M experiments remain mechanism and systems bring-up, not target
scale evidence.


## State quantization drift probe

On a real two-chunk BDH stream, symmetric per-tensor int8 state storage gave
mean state relative error `1.37%` and maximum next-chunk logit drift `0.00158`,
with finite outputs. This is encouraging for the 0.3B/large-state memory gate,
but requires validation at the target profile and across long contexts before
being enabled in training.


Scale accounting now includes int8 state estimates. The 0.3B profile drops from
~1.36GB BF16 persistent state to ~0.68GB int8 storage; the 5B profile drops from
~21.47GB to ~10.74GB before scales/metadata.
