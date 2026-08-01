"""HZ-0B reopening criterion 6: "Bad writes can be detected or rolled
back, not just hoped away" (plans/HZ-0B_Progress_Tracker.md).

Pure B2 simulator test, no LM needed. Tests THREE real, already-existing
mechanisms (not new features -- this criterion asks whether the B1
contract already provides what's needed, and it does):

1. DETECTION via write_source: every write records whether it came from
   a trusted/supervised source (SOURCE_SUPERVISED) or a learned/latent
   one (SOURCE_LATENT) -- a real, per-slot audit trail a caller can use
   to flag latent writes for extra scrutiny, without needing a new field.
2. ROLLBACK via serialize()/restore(): a full-state snapshot/restore
   round-trip -- if a write (or a batch of writes) turns out to be bad,
   restoring a pre-write snapshot undoes it completely and exactly.
3. ROLLBACK via delete(): a narrower, single-slot undo -- doesn't
   require a snapshot, just targets the one bad slot.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import SOURCE_LATENT, SOURCE_SUPERVISED, delete, read, reset, restore, serialize, write

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 16, 16


def _onehot(dim: int, index: int) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row])


def test_detection_via_write_source():
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    trusted_key, trusted_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, trusted_slot, _ = write(state, trusted_key, trusted_value, mx.array([1.0]), step=0, source=SOURCE_SUPERVISED, slot_idx=mx.array([0]))
    latent_key, latent_value = _onehot(KEY_DIM, 1), _onehot(VALUE_DIM, 1) * 5.0
    state, latent_slot, _ = write(state, latent_key, latent_value, mx.array([1.0]), step=1, source=SOURCE_LATENT, slot_idx=mx.array([1]))

    trusted_source = int(state.write_source[0, int(trusted_slot[0])])
    latent_source = int(state.write_source[0, int(latent_slot[0])])
    print(f"Slot {int(trusted_slot[0])} (trusted write): write_source={trusted_source} (expect {SOURCE_SUPERVISED})")
    print(f"Slot {int(latent_slot[0])} (latent write):   write_source={latent_source} (expect {SOURCE_LATENT})")
    ok = trusted_source == SOURCE_SUPERVISED and latent_source == SOURCE_LATENT
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- write provenance is queryable per-slot, "
          "a caller can flag/scrutinize latent writes without a new field.")
    return ok


def test_rollback_via_snapshot_restore():
    """Checks the bad slot's RAW state directly after restore, not via a
    similarity-based read() -- a confidence-weighted hard read against a
    mostly-empty memory doesn't return "nothing found" for an unrelated
    query, it returns the highest-CONFIDENCE slot regardless of true
    similarity (a real, correct property of read()'s own "best guess"
    design, not a bug -- but it makes read() the wrong tool to verify
    "is this specific slot empty," which is what rollback actually
    needs to guarantee). A direct raw-state check is unambiguous."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    good_key, good_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, _, _ = write(state, good_key, good_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    snapshot = serialize(state)

    bad_key, bad_value = _onehot(KEY_DIM, 1), _onehot(VALUE_DIM, 1) * 99.0
    state, bad_slot, _ = write(state, bad_key, bad_value, mx.array([1.0]), step=1, slot_idx=mx.array([1]))
    bad_write_present = float(state.confidence[0, int(bad_slot[0])]) > 0.0
    print(f"Bad write present before rollback (slot {int(bad_slot[0])} confidence > 0): {bad_write_present}")

    restored_state = restore(snapshot)
    bad_slot_confidence_after = float(restored_state.confidence[0, int(bad_slot[0])])
    bad_gone = bad_slot_confidence_after == 0.0
    readout_good_after_restore, _ = read(restored_state, good_key, slot_idx=mx.array([0]), hard=True)
    good_intact = bool(mx.all(mx.abs(readout_good_after_restore - good_value) < 1e-3))
    print(f"After restore(): bad slot confidence={bad_slot_confidence_after} (gone={bad_gone}), good write still intact exactly={good_intact}")
    ok = bad_write_present and bad_gone and good_intact
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- snapshot/restore fully and exactly undoes a bad write "
          "without touching unrelated, already-good state.")
    return ok


def test_rollback_via_delete():
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    bad_key, bad_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 99.0
    state, bad_slot, _ = write(state, bad_key, bad_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state = delete(state, bad_slot)
    readout, _ = read(state, bad_key, slot_idx=bad_slot, hard=True)
    confidence_after = float(state.confidence[0, int(bad_slot[0])])
    ok = bool(mx.all(mx.abs(readout) < 1e-6)) and confidence_after == 0.0
    print(f"After delete() on the bad slot: readout={readout.tolist()}, confidence={confidence_after}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- delete() cleanly undoes a single bad write "
          "without needing a full-state snapshot.")
    return ok


def main():
    print("=== 1. Detection via write_source ===")
    r1 = test_detection_via_write_source()
    print("\n=== 2. Rollback via serialize()/restore() ===")
    r2 = test_rollback_via_snapshot_restore()
    print("\n=== 3. Rollback via delete() ===")
    r3 = test_rollback_via_delete()

    print(f"\n--- Summary --- detection: {'PASS' if r1 else 'FAIL'}, snapshot rollback: {'PASS' if r2 else 'FAIL'}, single-slot rollback: {'PASS' if r3 else 'FAIL'}")
    if r1 and r2 and r3:
        print("RESULT: criterion 6 MET -- the B1 contract already provides real, working detection "
              "(write_source) and rollback (serialize/restore, delete) mechanisms; nothing new needed "
              "at the mechanism level. A real end-to-end 'when should a caller actually trigger rollback' "
              "policy (e.g. based on a trained confidence/anomaly signal) is separate, real future work.")
    else:
        print("RESULT: criterion 6 NOT fully met -- see failing case(s) above.")


if __name__ == "__main__":
    main()
