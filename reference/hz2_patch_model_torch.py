"""HZ2 fixed-patch model: small byte encoder -> fixed K-byte patches ->
shared hybrid core (GDN+FFN, GDN+FFN, local-attn+FFN, x4) -> small byte
decoder, per plans/HZ_Mixed_Depth_Pilot_2026-09-15.md section 2.

Trains a NEW architecture from scratch (no claim of equivalent behavior to
the existing S/H path in reference/hz_language_model_torch.py, whose
corpus lm_forward performs one workspace update per byte and leaves
persistent S unchanged -- the actual verified bug this pilot is meant to
route around by training memory through ordinary next-byte prediction).

Reuses reference/hz0a_torch_model.py's real, already-validated GDN2Mixer/
RMSNorm/SwiGLU (its per-timestep sequential recurrence, sequential/chunk/
FLA-backed variants already cross-checked against each other there) --
this file does not re-derive the delta-rule math.

Scope of THIS pass, stated plainly rather than silently simplified:
  - K=1 (patch size 1, "isolates patching" per the plan) is the primary
    target; K=4/8 patch SELECTION is implemented (see PatchFormer) but not
    yet exercised by a real training run.
  - Core/decoder attention here is windowed causal WITHOUT RoPE -- the
    plan specifies RoPE for the core's attention block; RoPE is NOT yet
    added (real, separate, not-yet-done work, not faked as present).
  - The "convolution width 4" GDN-geometry detail (a short causal conv on
    q/k/v before the recurrence, common in real GDN/KDA implementations)
    is NOT yet implemented -- GDN2Mixer is used as-is, unmodified.
  - prefill/step (streaming decode) and HZ2State exist and are claimed
    mathematically equivalent to slicing forward()'s output -- verified by
    tests/reference/test_hz2_patch_model.py's equivalence tests (full vs.
    one-shot prefill vs. multi-chunk vs. token-by-token step), not just
    asserted here. HZ2State has no save/load (de)serialization yet
    (tensors could be torch.save'd directly, but no dedicated helper
    exists) and no explicit reset/detach/clone API beyond Python's own
    object copying -- real, smaller, not-yet-done pieces.
  - Sharing: ONE core bank (three GDN2Mixer/CausalWindowAttention
    instances + their SwiGLUs) is applied 4 times in sequence, matching
    "sharing: three sets of large core weights" -- NOT twelve independent
    weight sets. Per-effective-depth state/norm independence (the plan's
    "twelve independent sets of normalization parameters and runtime
    state") IS implemented: each of the 12 effective stages gets its own
    RMSNorm parameters and its own GDN state tensor.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0a_torch_model import GDN2Mixer, HZ0AConfig, RMSNorm, SwiGLU


@dataclasses.dataclass(frozen=True)
class HZ2Config:
    vocab_size: int = 256
    byte_width: int = 192
    byte_ffn: int = 512
    byte_heads: int = 3
    byte_head_dim: int = 64
    core_width: int = 768
    core_ffn: int = 2048
    core_q_heads: int = 12
    core_kv_heads: int = 3
    core_head_dim: int = 64
    core_bank_repeats: int = 4
    core_attn_window_patches: int = 128
    decoder_attn_window_bytes: int = 256
    decoder_q_heads: int = 3
    decoder_kv_heads: int = 1
    K: int = 1  # patch size in bytes


class CausalWindowAttention(nn.Module):
    """Windowed causal attention -- grouped-query (core_q_heads may exceed
    core_kv_heads), no RoPE yet (see module docstring). Window is applied
    via an additive mask so a real chunked/streaming implementation can
    later replace this with a real sliding-window kernel without changing
    the module's observable forward behavior at prefill-length inputs.
    """

    def __init__(self, width: int, q_heads: int, kv_heads: int, head_dim: int, window: int):
        super().__init__()
        assert q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads for grouped-query attention"
        self.q_heads, self.kv_heads, self.head_dim, self.window = q_heads, kv_heads, head_dim, window
        self.q_proj = nn.Linear(width, q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(q_heads * head_dim, width, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        groups = self.q_heads // self.kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        positions = torch.arange(t, device=x.device)
        causal = positions[None, :] <= positions[:, None]
        within_window = positions[None, :] > positions[:, None] - self.window
        mask = causal & within_window
        bias = torch.zeros(t, t, device=x.device, dtype=x.dtype).masked_fill(~mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(out.transpose(1, 2).reshape(b, t, self.q_heads * self.head_dim))

    def forward_incremental(self, x_chunk, kv_cache, pos_offset: int):
        """Mathematically equivalent to slicing forward()'s output at the
        positions [pos_offset, pos_offset+t_new) of a full sequence that
        happened to start with whatever produced `kv_cache` -- kv_cache
        holds up to `window` most-recent (already window-length-bounded)
        projected key/value tensors, NOT raw hidden states (saves
        recomputing k_proj/v_proj for old positions every call). pos_offset
        is the running absolute-position counter for THIS attention
        instance's occurrence (each of the 4 bank repeats' attn stages
        tracks its own, independent per the plan's per-effective-depth
        state requirement), not derived from cache length once trimming
        has happened.
        """
        b, t_new, _ = x_chunk.shape
        q = self.q_proj(x_chunk).view(b, t_new, self.q_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_chunk).view(b, t_new, self.kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_chunk).view(b, t_new, self.kv_heads, self.head_dim).transpose(1, 2)
        if kv_cache is None:
            k_all, v_all, cache_len = k_new, v_new, 0
        else:
            k_cache, v_cache = kv_cache
            cache_len = k_cache.shape[2]
            k_all = torch.cat([k_cache, k_new], dim=2)
            v_all = torch.cat([v_cache, v_new], dim=2)
        groups = self.q_heads // self.kv_heads
        k_rep = k_all.repeat_interleave(groups, dim=1)
        v_rep = v_all.repeat_interleave(groups, dim=1)
        q_positions = pos_offset + torch.arange(t_new, device=x_chunk.device)
        k_positions = (pos_offset - cache_len) + torch.arange(cache_len + t_new, device=x_chunk.device)
        causal = k_positions[None, :] <= q_positions[:, None]
        within_window = k_positions[None, :] > (q_positions[:, None] - self.window)
        mask = causal & within_window
        bias = torch.zeros(t_new, cache_len + t_new, device=x_chunk.device, dtype=x_chunk.dtype).masked_fill(~mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=bias)
        new_cache = (k_all[:, :, -self.window:].contiguous(), v_all[:, :, -self.window:].contiguous())
        return self.out_proj(out.transpose(1, 2).reshape(b, t_new, self.q_heads * self.head_dim)), new_cache


class GDNFFNBlock(nn.Module):
    """One 'GDN + FFN' effective stage, its own RMSNorm/state per the
    plan's 'twelve independent sets of normalization parameters and
    runtime state' requirement -- `mixer` is the SHARED module passed in
    (not owned by this block), so weight sharing across repeated stages
    is structural (same nn.Module object), not just equal values.
    """

    def __init__(self, width, ffn, mixer: GDN2Mixer):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.mixer = mixer  # shared across whichever stages pass the same instance in
        self.mlp = SwiGLU(width, ffn)

    def forward(self, x, state):
        mixed, next_state = self.mixer(self.norm1(x), state)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), next_state


class AttnFFNBlock(nn.Module):
    def __init__(self, width, ffn, attn: CausalWindowAttention):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn = attn
        self.mlp = SwiGLU(width, ffn)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def forward_incremental(self, x, kv_cache, pos_offset: int):
        attn_out, new_cache = self.attn.forward_incremental(self.norm1(x), kv_cache, pos_offset)
        x = x + attn_out
        return x + self.mlp(self.norm2(x)), new_cache


class ByteTower(nn.Module):
    """Two GDN+FFN blocks, SHARED weights (one GDN2Mixer instance reused
    for both stages, each with its own norm/state) -- used for both the
    byte encoder and byte decoder's GDN portion, per the plan's identical
    spec for each ('two independent Gated DeltaNet + SwiGLU blocks').
    """

    def __init__(self, width, ffn, heads, head_dim):
        super().__init__()
        gdn_config = HZ0AConfig(vocab_size=1, d_model=width, num_layers=1, num_heads=heads,
                                d_k=head_dim, d_v=head_dim, d_ff=ffn, attention_layer_indices=())
        self.mixer = GDN2Mixer(gdn_config)
        self.stage1 = GDNFFNBlock(width, ffn, self.mixer)
        self.stage2 = GDNFFNBlock(width, ffn, self.mixer)
        self.heads, self.head_dim = heads, head_dim

    def init_states(self, batch_size, device, dtype):
        shape = (batch_size, self.heads, self.head_dim, self.head_dim)
        return [torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)]

    def forward(self, x, states):
        x, s1 = self.stage1(x, states[0])
        x, s2 = self.stage2(x, states[1])
        return x, [s1, s2]


class PatchFormer(nn.Module):
    """Selects the last causal byte-encoder output of each completed
    K-byte patch, then projects byte_width -> core_width. K=1 makes every
    byte its own (trivially 'completed') patch -- the isolating case the
    plan calls out explicitly ('K=1 isolates patching').
    """

    def __init__(self, byte_width: int, core_width: int, K: int):
        super().__init__()
        self.K = K
        self.proj = nn.Linear(byte_width, core_width, bias=False)

    def forward(self, byte_hidden: torch.Tensor) -> torch.Tensor:
        # byte_hidden: (B, T_bytes, byte_width). A patch at byte positions
        # [i*K, i*K+K-1] is "completed" once byte i*K+K-1 has been seen --
        # select every K-th position (0-indexed: K-1, 2K-1, ...), causal by
        # construction since ByteTower's GDN blocks are themselves causal.
        b, t, _ = byte_hidden.shape
        n_complete = t // self.K
        if n_complete == 0:
            return byte_hidden.new_zeros(b, 0, self.proj.out_features)
        idx = torch.arange(self.K - 1, n_complete * self.K, self.K, device=byte_hidden.device)
        selected = byte_hidden.index_select(1, idx)
        return self.proj(selected)


@dataclasses.dataclass
class HZ2State:
    """Everything needed to resume forward_incremental() across separate
    calls, chunk-split-invariant by construction (bookkeeping is driven by
    `bytes_seen`, an absolute counter, never by how a caller happened to
    split the input into chunks).

    No `partial_patch_bytes` buffer is needed: PatchFormer's real spec is
    'select the LAST causal encoder output of each completed patch' (not
    an aggregation/pool over the patch's K bytes) -- ByteTower's GDN
    recurrence already causally folds in all K bytes' information by the
    time it reaches that last position, so there is nothing left over to
    buffer across a chunk boundary.
    """
    enc_gdn_states: list
    core_gdn_states: list
    core_attn_caches: list  # len core_bank_repeats, each None or (k_cache, v_cache)
    core_attn_pos: list     # len core_bank_repeats, running absolute position per attn occurrence
    dec_gdn_states: list
    decoder_attn_cache: tuple | None
    decoder_attn_pos: int
    last_conditioning: torch.Tensor  # (B, byte_width) -- active until the next patch completes
    bytes_seen: int


class HZ2PatchModel(nn.Module):
    def __init__(self, config: HZ2Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.byte_width)
        nn.init.normal_(self.embed.weight, std=math.sqrt(1.0 / config.byte_width))
        self.byte_encoder = ByteTower(config.byte_width, config.byte_ffn, config.byte_heads, config.byte_head_dim)
        self.patch_former = PatchFormer(config.byte_width, config.core_width, config.K)

        core_gdn_config = HZ0AConfig(vocab_size=1, d_model=config.core_width, num_layers=1,
                                     num_heads=config.core_q_heads, d_k=config.core_head_dim,
                                     d_v=config.core_head_dim, d_ff=config.core_ffn, attention_layer_indices=())
        # "Sharing: three sets of large core weights" -- one GDN mixer used
        # for BOTH GDN+FFN stages in the bank, one attention module for the
        # local-attn+FFN stage, all three reused every bank repeat.
        self.core_gdn_mixer = GDN2Mixer(core_gdn_config)
        self.core_attn = CausalWindowAttention(config.core_width, config.core_q_heads, config.core_kv_heads,
                                               config.core_head_dim, config.core_attn_window_patches)
        self.core_bank = nn.ModuleList()
        for _ in range(config.core_bank_repeats):
            self.core_bank.append(GDNFFNBlock(config.core_width, config.core_ffn, self.core_gdn_mixer))
            self.core_bank.append(GDNFFNBlock(config.core_width, config.core_ffn, self.core_gdn_mixer))
            self.core_bank.append(AttnFFNBlock(config.core_width, config.core_ffn, self.core_attn))

        self.core_to_byte = nn.Linear(config.core_width, config.byte_width, bias=False)
        # Learned initial context before the first completed patch, per
        # the plan's decoder-conditioning spec.
        self.initial_context = nn.Parameter(torch.zeros(config.byte_width).normal_(std=0.02))

        self.byte_decoder_gdn = ByteTower(config.byte_width, config.byte_ffn, config.byte_heads, config.byte_head_dim)
        self.decoder_attn = CausalWindowAttention(config.byte_width, config.decoder_q_heads, config.decoder_kv_heads,
                                                  config.core_head_dim, config.decoder_attn_window_bytes)
        self.decoder_attn_block = AttnFFNBlock(config.byte_width, config.byte_ffn, self.decoder_attn)

        self.final_norm = RMSNorm(config.byte_width)
        self.lm_head = nn.Linear(config.byte_width, config.vocab_size, bias=False)

    def forward(self, byte_ids: torch.Tensor, targets: torch.Tensor | None = None,
               loss_mask: torch.Tensor | None = None):
        """loss_mask: (B, T) bool, True = include this position's loss (e.g.
        real bytes); False = exclude (padding). Implemented via PyTorch's
        cross_entropy ignore_index=-100 convention, applied on TOP of
        whatever `targets` already carries -- does not mutate `targets`."""
        C = self.config
        b, t = byte_ids.shape
        device, dtype = byte_ids.device, self.embed.weight.dtype

        x = self.embed(byte_ids)
        enc_states = self.byte_encoder.init_states(b, device, dtype)
        enc_hidden, _ = self.byte_encoder(x, enc_states)

        patches = self.patch_former(enc_hidden)  # (B, n_patches, core_width)
        if patches.shape[1] > 0:
            core_state_shape = (b, C.core_q_heads, C.core_head_dim, C.core_head_dim)
            core_states = [torch.zeros(core_state_shape, device=device, dtype=dtype) for _ in range(2 * C.core_bank_repeats)]
            core_hidden = patches
            for r in range(C.core_bank_repeats):
                gdn_stage_a, gdn_stage_b, attn_stage = self.core_bank[3 * r:3 * r + 3]
                core_hidden, core_states[2 * r] = gdn_stage_a(core_hidden, core_states[2 * r])
                core_hidden, core_states[2 * r + 1] = gdn_stage_b(core_hidden, core_states[2 * r + 1])
                core_hidden = attn_stage(core_hidden)
            core_conditioning = self.core_to_byte(core_hidden)  # (B, n_patches, byte_width)
        else:
            core_conditioning = enc_hidden.new_zeros(b, 0, C.byte_width)

        # Causal alignment (plan section 2): byte t is conditioned on the
        # LATEST previously-completed patch's core output, never its own
        # (still-incomplete) patch. Patch i (0-indexed) completes at byte
        # position i*K + K - 1; bytes [0, K-1] see only initial_context,
        # bytes [i*K+K, (i+1)*K+K-1] see patch i's conditioning.
        conditioning = torch.empty(b, t, C.byte_width, device=device, dtype=dtype)
        conditioning[:, :C.K] = self.initial_context
        n_complete = core_conditioning.shape[1]
        for i in range(n_complete):
            start = (i + 1) * C.K
            end = min((i + 2) * C.K, t)
            if start >= t:
                break
            conditioning[:, start:end] = core_conditioning[:, i:i + 1]

        dec_states = self.byte_decoder_gdn.init_states(b, device, dtype)
        dec_hidden, _ = self.byte_decoder_gdn(x + conditioning, dec_states)
        dec_hidden = self.decoder_attn_block(dec_hidden)

        logits = self.lm_head(self.final_norm(dec_hidden))
        loss = None
        if targets is not None:
            effective_targets = targets if loss_mask is None else targets.masked_fill(~loss_mask, -100)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), effective_targets.reshape(-1))
        return logits, loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch_size: int, device=None, dtype=None) -> HZ2State:
        C = self.config
        dtype = dtype or self.embed.weight.dtype
        core_state_shape = (batch_size, C.core_q_heads, C.core_head_dim, C.core_head_dim)
        return HZ2State(
            enc_gdn_states=self.byte_encoder.init_states(batch_size, device, dtype),
            core_gdn_states=[torch.zeros(core_state_shape, device=device, dtype=dtype)
                            for _ in range(2 * C.core_bank_repeats)],
            core_attn_caches=[None] * C.core_bank_repeats,
            core_attn_pos=[0] * C.core_bank_repeats,
            dec_gdn_states=self.byte_decoder_gdn.init_states(batch_size, device, dtype),
            decoder_attn_cache=None,
            decoder_attn_pos=0,
            last_conditioning=self.initial_context.to(dtype).expand(batch_size, -1).clone(),
            bytes_seen=0,
        )

    def forward_with_boundaries(self, byte_ids: torch.Tensor, boundary_positions: list[int],
                                targets: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None):
        """Document-boundary reset: every position in `boundary_positions`
        (0-indexed byte offsets into byte_ids, 0 < p < T) starts a fresh
        document -- state is fully reset (== init_state) right before that
        byte is processed, exactly matching the plan's 'reset all states
        and positions between documents.'

        Real, stated scope limit: boundaries apply UNIFORMLY across the
        whole batch (every row restarts its document at the same byte
        offset). Independent per-row boundaries -- different documents
        restarting at different positions within one packed batch, which
        real corpus-packing pipelines often need -- are NOT supported yet.
        `bytes_seen`/`core_attn_pos` are scalar ints shared across the
        batch dimension in this implementation, not per-row tensors; making
        them per-row is the real change needed for that case, not done
        here. Use this for batches where documents are packed with aligned
        boundaries (e.g. one document per batch row, or batch-uniform
        chunking), not arbitrary packed-and-staggered documents.
        """
        b, t = byte_ids.shape
        device, dtype = byte_ids.device, self.embed.weight.dtype
        state = self.init_state(b, device, dtype)
        cuts = sorted(set([0, t] + [p for p in boundary_positions if 0 < p < t]))
        logits_chunks = []
        weighted_losses = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            if start != 0:
                state = self.init_state(b, device, dtype)  # full reset at each declared boundary
            chunk_targets = targets[:, start:end] if targets is not None else None
            chunk_loss_mask = loss_mask[:, start:end] if loss_mask is not None else None
            logits, loss, state = self.forward_incremental(byte_ids[:, start:end], state,
                                                            chunk_targets, chunk_loss_mask)
            logits_chunks.append(logits)
            if loss is not None:
                weighted_losses.append((loss, end - start))
        all_logits = torch.cat(logits_chunks, dim=1)
        total_loss = None
        if weighted_losses:
            total_weight = sum(w for _, w in weighted_losses)
            total_loss = sum(l * w for l, w in weighted_losses) / total_weight
        return all_logits, total_loss

    def forward_incremental(self, byte_ids_chunk: torch.Tensor, state: HZ2State, targets=None,
                            loss_mask: torch.Tensor | None = None):
        """Mathematically equivalent to running forward() on the full
        sequence up through this chunk and slicing out this chunk's
        positions -- prefill(chunk) and step(single_byte) are both this
        same method, just called with t>1 or t==1. See HZ2State's
        docstring for why no leftover-byte buffer is needed.

        loss_mask: (B, t) bool for THIS chunk, same convention as forward().
        """
        C = self.config
        b, t = byte_ids_chunk.shape
        device, dtype = byte_ids_chunk.device, self.embed.weight.dtype
        if t == 0:
            # GDN2Mixer's internal per-timestep loop stacks its outputs
            # (torch.stack over zero tensors raises) -- an empty chunk is a
            # real, explicitly-called-out edge case (plan section 3, item 2:
            # "empty input"), not something to let crash upstream.
            return byte_ids_chunk.new_zeros(b, 0, C.vocab_size, dtype=dtype), None, state

        x = self.embed(byte_ids_chunk)
        enc_hidden, new_enc_states = self.byte_encoder(x, state.enc_gdn_states)

        global_idx = state.bytes_seen + torch.arange(t, device=device)
        complete_positions = (global_idx % C.K == C.K - 1).nonzero(as_tuple=True)[0].tolist()

        core_gdn_states = list(state.core_gdn_states)
        core_attn_caches = list(state.core_attn_caches)
        core_attn_pos = list(state.core_attn_pos)
        conditioning = torch.empty(b, t, C.byte_width, device=device, dtype=dtype)
        current_conditioning = state.last_conditioning
        cursor = 0
        for local_i in complete_positions:
            # Bytes [cursor, local_i], INCLUDING the patch-completing byte
            # itself, use the conditioning that was active BEFORE this
            # patch -- a patch's own completing byte never sees its own
            # completed representation (causal alignment, plan section 2).
            conditioning[:, cursor:local_i + 1] = current_conditioning.unsqueeze(1)
            patch = self.patch_former.proj(enc_hidden[:, local_i:local_i + 1])  # (B, 1, core_width)
            core_hidden = patch
            for r in range(C.core_bank_repeats):
                gdn_stage_a, gdn_stage_b, attn_stage = self.core_bank[3 * r:3 * r + 3]
                core_hidden, core_gdn_states[2 * r] = gdn_stage_a(core_hidden, core_gdn_states[2 * r])
                core_hidden, core_gdn_states[2 * r + 1] = gdn_stage_b(core_hidden, core_gdn_states[2 * r + 1])
                core_hidden, core_attn_caches[r] = attn_stage.forward_incremental(
                    core_hidden, core_attn_caches[r], core_attn_pos[r])
                core_attn_pos[r] += 1
            current_conditioning = self.core_to_byte(core_hidden).squeeze(1)  # (B, byte_width)
            cursor = local_i + 1
        conditioning[:, cursor:t] = current_conditioning.unsqueeze(1)

        dec_hidden, dec_gdn_states = self.byte_decoder_gdn(x + conditioning, state.dec_gdn_states)
        dec_hidden, decoder_attn_cache = self.decoder_attn_block.forward_incremental(
            dec_hidden, state.decoder_attn_cache, state.decoder_attn_pos)

        logits = self.lm_head(self.final_norm(dec_hidden))
        loss = None
        if targets is not None:
            effective_targets = targets if loss_mask is None else targets.masked_fill(~loss_mask, -100)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), effective_targets.reshape(-1))

        new_state = dataclasses.replace(
            state,
            enc_gdn_states=new_enc_states,
            core_gdn_states=core_gdn_states,
            core_attn_caches=core_attn_caches,
            core_attn_pos=core_attn_pos,
            dec_gdn_states=dec_gdn_states,
            decoder_attn_cache=decoder_attn_cache,
            decoder_attn_pos=state.decoder_attn_pos + t,
            last_conditioning=current_conditioning,
            bytes_seen=state.bytes_seen + t,
        )
        return logits, loss, new_state

    def prefill(self, byte_ids_chunk: torch.Tensor, state: HZ2State, targets=None):
        return self.forward_incremental(byte_ids_chunk, state, targets)

    def step(self, byte_id: torch.Tensor, state: HZ2State):
        """byte_id: (B,) or (B, 1). Returns (logits (B, 1, vocab), new_state)."""
        if byte_id.dim() == 1:
            byte_id = byte_id.unsqueeze(1)
        logits, _, new_state = self.forward_incremental(byte_id, state)
        return logits, new_state
