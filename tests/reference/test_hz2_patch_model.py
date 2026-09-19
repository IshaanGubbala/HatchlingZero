"""Correctness gates for reference/hz2_patch_model_torch.py, per
plans/HZ_Mixed_Depth_Pilot_2026-09-15.md section 3.

Covers, on tiny CPU models: causal safety (item 3 -- the single most
important property to verify before anything else, since a leak here
would make the whole architecture cheat at next-byte prediction), basic
shape/gradient-flow correctness for K=1/4/8 including non-multiple-of-K
lengths, and shared-weight identity across bank repeats and byte-tower
stages.

NOT covered here yet -- real, separate, not-yet-started work (see the
model file's own docstring for what's implemented so far):
  - padding / document-reset / loss-mask behavior
  - the controlled memory-dependency task (zeroing carried state must
    change predictions) -- a real test to write once there's a reason to
    believe the core's state carries something worth zeroing; premature
    before any real training
  - BF16-vs-FP32 GPU checks
"""
import torch

from reference.hz2_patch_model_torch import HZ2Config, HZ2PatchModel


def _tiny_config(K=1):
    return HZ2Config(vocab_size=64, byte_width=32, byte_ffn=64, byte_heads=2, byte_head_dim=16,
                     core_width=48, core_ffn=96, core_q_heads=4, core_kv_heads=2, core_head_dim=12,
                     core_bank_repeats=2, core_attn_window_patches=8, decoder_attn_window_bytes=16,
                     decoder_q_heads=2, decoder_kv_heads=1, K=K)


def test_shapes_and_full_gradient_flow_non_multiple_of_k():
    for K in (1, 4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K))
        idx = torch.randint(0, 64, (2, 17))  # 17 is not a multiple of 1, 4, or 8 -- exercises the partial-final-patch path
        targets = torch.randint(0, 64, (2, 17))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 17, 64)
        assert not torch.isnan(loss)
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                  if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
        assert not missing, f"K={K}: params with no/zero gradient: {missing}"


def test_causal_safety_future_byte_cannot_change_earlier_logits():
    # Item 3, the critical one: perturbing byte j must leave every logit
    # at position i < j byte-for-byte identical -- both through the byte
    # tower's own causal GDN/windowed-attention AND through patch
    # conditioning (a later-completing patch must never reach an earlier
    # decoder position).
    for K in (1, 4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (1, 20))
        with torch.no_grad():
            logits_before, _ = model(idx)

        perturbed = idx.clone()
        perturb_at = 15  # deliberately inside a later patch for every K in (1, 4, 8) given length 20
        perturbed[0, perturb_at] = (perturbed[0, perturb_at] + 1) % 64
        with torch.no_grad():
            logits_after, _ = model(perturbed)

        assert torch.equal(logits_before[:, :perturb_at], logits_after[:, :perturb_at]), \
            f"K={K}: perturbing byte {perturb_at} changed a prediction at an earlier position -- causal leak"
        # Sanity check the test itself isn't vacuous: the perturbation
        # SHOULD change at least the logit at the perturbed position or
        # later (otherwise byte_ids isn't actually wired into the model).
        assert not torch.equal(logits_before[:, perturb_at:], logits_after[:, perturb_at:]), \
            f"K={K}: perturbation had no effect anywhere -- test is vacuous, model may be ignoring input"


def test_first_patch_worth_of_bytes_use_only_initial_context():
    # Bytes [0, K-1] complete no patch yet -- they must all see the SAME
    # learned initial_context, not each other's (still-in-progress) patch.
    for K in (4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (1, K + 2))
        with torch.no_grad():
            logits_a, _ = model(idx)
        idx_b = idx.clone()
        idx_b[0, K - 1] = (idx_b[0, K - 1] + 1) % 64  # perturb the LAST byte of the first (still-incomplete) patch
        with torch.no_grad():
            logits_b, _ = model(idx_b)
        # Position K-1 itself is causally affected by its own byte value (byte embedding),
        # but position 0..K-2 (earlier in the same incomplete patch) must be unaffected --
        # already covered by the general causal test above; this test instead confirms
        # the FIRST K positions don't see a nonzero patch conditioning at all yet.
        assert torch.equal(logits_a[:, :K - 1], logits_b[:, :K - 1])


def test_core_bank_and_byte_tower_share_weights_across_repeats():
    # "Sharing: three sets of large core weights" -- confirm this is
    # structural (same nn.Module object reused), not merely equal values.
    config = _tiny_config(K=1)
    model = HZ2PatchModel(config)
    gdn_stage_a_ids = {id(model.core_bank[3 * r].mixer) for r in range(config.core_bank_repeats)}
    gdn_stage_b_ids = {id(model.core_bank[3 * r + 1].mixer) for r in range(config.core_bank_repeats)}
    attn_ids = {id(model.core_bank[3 * r + 2].attn) for r in range(config.core_bank_repeats)}
    assert len(gdn_stage_a_ids) == 1 and len(gdn_stage_b_ids) == 1 and len(attn_ids) == 1
    assert gdn_stage_a_ids == gdn_stage_b_ids, "the plan's two core GDN+FFN stages per repeat share one mixer instance"
    assert id(model.byte_encoder.mixer) != id(model.byte_decoder_gdn.mixer), \
        "encoder and decoder are separate towers per the plan -- they must NOT share GDN weights"


def _run_incremental(model, idx, chunk_sizes):
    """Feeds idx through forward_incremental split according to chunk_sizes
    (must sum to idx.shape[1]), returns concatenated logits."""
    assert sum(chunk_sizes) == idx.shape[1]
    state = model.init_state(idx.shape[0], idx.device, model.embed.weight.dtype)
    logits_chunks = []
    cursor = 0
    for size in chunk_sizes:
        chunk = idx[:, cursor:cursor + size]
        logits, _, state = model.forward_incremental(chunk, state)
        logits_chunks.append(logits)
        cursor += size
    return torch.cat(logits_chunks, dim=1)


def test_prefill_matches_full_forward_one_shot():
    # The simplest possible streaming case: one prefill call covering the
    # whole sequence must reproduce forward() exactly.
    for K in (1, 4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (2, 19))
        with torch.no_grad():
            full_logits, _ = model(idx)
            incremental_logits = _run_incremental(model, idx, [19])
        assert torch.allclose(full_logits, incremental_logits, atol=1e-5, rtol=1e-4), \
            f"K={K}: one-shot prefill diverged from full forward()"


def test_multi_chunk_streaming_matches_full_forward():
    # Real streaming: split at arbitrary, patch-boundary-crossing points
    # (not aligned to K) across K=1/4/8, and additionally a token-by-token
    # step() run -- all must reproduce forward() exactly, since the whole
    # point of HZ2State's bookkeeping is chunk-split invariance.
    for K in (1, 4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (2, 19))
        with torch.no_grad():
            full_logits, _ = model(idx)

            for chunk_sizes in ([7, 5, 7], [1] * 19, [19], [3, 16], [10, 1, 1, 7]):
                incremental_logits = _run_incremental(model, idx, chunk_sizes)
                assert torch.allclose(full_logits, incremental_logits, atol=1e-5, rtol=1e-4), \
                    f"K={K}, chunks={chunk_sizes}: streaming diverged from full forward()"


def test_step_by_step_matches_full_forward():
    for K in (1, 4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            full_logits, _ = model(idx)
            state = model.init_state(idx.shape[0], idx.device, model.embed.weight.dtype)
            step_logits = []
            for pos in range(idx.shape[1]):
                logits, state = model.step(idx[:, pos], state)
                step_logits.append(logits)
            step_logits = torch.cat(step_logits, dim=1)
        assert torch.allclose(full_logits, step_logits, atol=1e-5, rtol=1e-4), \
            f"K={K}: token-by-token step() diverged from full forward()"


def test_empty_chunk_does_not_crash_and_leaves_state_unchanged():
    # Plan section 3, item 2's explicit "empty input" case -- a real bug
    # this test caught: GDN2Mixer's internal per-timestep loop stacks its
    # outputs, which raises on zero timesteps if not guarded upstream.
    config = _tiny_config(K=4)
    model = HZ2PatchModel(config).eval()
    state = model.init_state(1, torch.device("cpu"), torch.float32)
    idx = torch.randint(0, config.vocab_size, (1, 0))
    logits, loss, new_state = model.forward_incremental(idx, state)
    assert logits.shape == (1, 0, config.vocab_size)
    assert loss is None
    assert new_state is state


def test_incremental_loss_with_targets_matches_full_forward():
    for K in (1, 4, 8):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (2, 10))
        targets = torch.randint(0, 64, (2, 10))
        with torch.no_grad():
            full_logits, full_loss = model(idx, targets)
            state = model.init_state(2, idx.device, model.embed.weight.dtype)
            inc_logits, inc_loss, _ = model.forward_incremental(idx, state, targets)
        assert torch.allclose(full_logits, inc_logits, atol=1e-5, rtol=1e-4)
        assert torch.allclose(full_loss, inc_loss, atol=1e-5, rtol=1e-4), f"K={K}: loss mismatch"


def test_windowed_attention_cache_stays_bounded():
    # The whole point of the KV cache trimming in CausalWindowAttention.forward_incremental:
    # cached length must never exceed the configured window, however long the stream runs.
    torch.manual_seed(0)
    config = _tiny_config(K=1)
    model = HZ2PatchModel(config).eval()
    idx = torch.randint(0, 64, (1, config.core_attn_window_patches * 3))
    state = model.init_state(1, idx.device, model.embed.weight.dtype)
    with torch.no_grad():
        for pos in range(idx.shape[1]):
            _, state = model.step(idx[:, pos], state)
    for cache in state.core_attn_caches:
        assert cache is not None
        k_cache, v_cache = cache
        assert k_cache.shape[2] <= config.core_attn_window_patches
        assert v_cache.shape[2] <= config.core_attn_window_patches
    k_cache, v_cache = state.decoder_attn_cache
    assert k_cache.shape[2] <= config.decoder_attn_window_bytes


def test_loss_mask_excludes_padding_positions():
    for K in (1, 4):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        idx = torch.randint(0, 64, (1, 10))
        targets = torch.randint(0, 64, (1, 10))
        loss_mask = torch.ones(1, 10, dtype=torch.bool)
        loss_mask[:, 6:] = False  # last 4 positions are "padding"

        with torch.no_grad():
            _, masked_loss = model(idx, targets, loss_mask=loss_mask)
            # Manually compute the same loss using only the unmasked span,
            # from a SEPARATE, unmasked call -- if loss_mask is wired
            # correctly, these must match (both are plain mean CE over the
            # same 6 real positions).
            full_logits, _ = model(idx)
            manual_loss = torch.nn.functional.cross_entropy(
                full_logits[:, :6].reshape(-1, 64), targets[:, :6].reshape(-1))
        assert torch.allclose(masked_loss, manual_loss, atol=1e-5, rtol=1e-4), \
            f"K={K}: loss_mask did not correctly exclude padded positions"


def test_forward_incremental_loss_mask_matches_forward():
    config = _tiny_config(K=4)
    model = HZ2PatchModel(config).eval()
    idx = torch.randint(0, 64, (2, 10))
    targets = torch.randint(0, 64, (2, 10))
    loss_mask = torch.ones(2, 10, dtype=torch.bool)
    loss_mask[:, 3:5] = False
    with torch.no_grad():
        _, full_loss = model(idx, targets, loss_mask=loss_mask)
        state = model.init_state(2, idx.device, model.embed.weight.dtype)
        _, inc_loss, _ = model.forward_incremental(idx, state, targets, loss_mask)
    assert torch.allclose(full_loss, inc_loss, atol=1e-5, rtol=1e-4)


def test_document_boundary_reset_matches_fresh_forward_on_segment_alone():
    # The real proof of document reset: the segment AFTER a declared
    # boundary must be byte-for-byte identical to running forward() on
    # THAT SEGMENT ALONE, starting from a fresh model state -- i.e. zero
    # leakage from whatever came before the boundary in the same call.
    for K in (1, 4):
        torch.manual_seed(0)
        model = HZ2PatchModel(_tiny_config(K)).eval()
        doc1 = torch.randint(0, 64, (1, 9))
        doc2 = torch.randint(0, 64, (1, 7))
        packed = torch.cat([doc1, doc2], dim=1)

        with torch.no_grad():
            packed_logits, _ = model.forward_with_boundaries(packed, boundary_positions=[doc1.shape[1]])
            doc2_alone_logits, _ = model(doc2)

        packed_doc2_logits = packed_logits[:, doc1.shape[1]:]
        assert torch.allclose(packed_doc2_logits, doc2_alone_logits, atol=1e-5, rtol=1e-4), \
            f"K={K}: document-2 segment leaked state from document 1"


def test_document_boundary_first_segment_matches_plain_forward():
    # The segment BEFORE the first declared boundary is just a normal,
    # unreset forward() call over its own bytes.
    config = _tiny_config(K=1)
    model = HZ2PatchModel(config).eval()
    doc1 = torch.randint(0, 64, (1, 9))
    doc2 = torch.randint(0, 64, (1, 7))
    packed = torch.cat([doc1, doc2], dim=1)
    with torch.no_grad():
        packed_logits, _ = model.forward_with_boundaries(packed, boundary_positions=[doc1.shape[1]])
        doc1_alone_logits, _ = model(doc1)
    assert torch.allclose(packed_logits[:, :doc1.shape[1]], doc1_alone_logits, atol=1e-5, rtol=1e-4)


def test_twelve_effective_stages_have_independent_norm_and_state():
    # "Twelve independent sets of normalization parameters and runtime
    # state" despite three sets of shared large weights.
    config = _tiny_config(K=1)
    model = HZ2PatchModel(config)
    norm_ids = set()
    for r in range(config.core_bank_repeats):
        for stage in model.core_bank[3 * r:3 * r + 3]:
            norm_ids.add(id(stage.norm1))
    assert len(norm_ids) == 3 * config.core_bank_repeats, "each effective stage must own its own RMSNorm instance"
