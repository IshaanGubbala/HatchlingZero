"""HZ-CQ core forward pass, plans/newnewplan.md section 33 (the BDH-CQ
pivot, 2026-08-31): persistent task memory + a separate reasoning
workspace, both real BDH recurrence, composed entirely from pieces this
project has already validated -- no new attention/MLP mechanism.

Two distinct recurrent phases, matching the target spec:

    S_t = U_theta(S_{t-1}, D_t)          -- task memory, built from demos
    H_0 = E_theta(x*, S_K)               -- workspace init, conditioned on memory + query
    H_{r+1} = F_theta(H_r, S_K)          -- R latent reasoning rounds
    y_hat = G_theta(H_R)                 -- decode (real bytes, teacher-forced here)

Real, disclosed mapping onto BDH's actual mechanism (BDH has no
separate "memory tensor" distinct from its own sequence positions --
recurrence happens by refining a growing sequence of positions together
across `n_rounds`, not by writing to an external slot):

  * S is realized as the model's own hidden state over the growing
    demonstration+query text -- appending demo_i's tokens as new
    sequence positions and continuing the SAME recurrent process IS
    U_theta(S_{t-1}, D_t): the exact-addressing write at each round
    already reads/writes across the whole sequence including all prior
    demos, so earlier demos genuinely influence how later ones (and the
    query) get processed. This is the identical growing-sequence-with-
    carry pattern already used and verified in
    scripts/hz0h_bdh_progressive_latentization_train.py's `_full_rounds`
    helper (reused here directly, not reimplemented).
  * H is realized the same way progressive-latentization realized a
    latent reasoning step: a NEW appended position, initialized as a
    copy of the last memory/query position (not a token embedding --
    there is no token here), advanced by real recurrent rounds. R of
    these positions in a row = R latent reasoning steps, each one a
    real one-round BDH write (exact address + adaptive-gate-controlled
    residual), NOT a full n_layer sub-forward per step.
  * The recurrence engine itself is the validated adaptive gate
    (reference/hz0h_bdh_adaptive_gate_torch.py's `_refresh_iteration`,
    real state-dependence, real R=16 stability, 1.3879 champion) at
    full refresh -- deliberately NOT combined with the newly-found K=4
    cached-schedule refresh champion (plans/newnewplan.md section 32)
    yet, since that combination (section 27-28's step D) has never
    been validated and stacking two unvalidated changes into one new
    architecture would make failures impossible to attribute.

Recomputes full rounds over the whole growing sequence at every phase
transition rather than anything incremental -- correct (BDH's exact
addressing genuinely needs the whole current sequence, there is no
causal mask to exploit for a real KV-cache-style shortcut here) and
the same real, disclosed choice progressive-latentization already made
for the same reason.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_adaptive_gate_torch import _refresh_iteration
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def _embed_bytes(model: BDHVBSubspaceDecoder, byte_ids: list[int], device) -> torch.Tensor:
    idx = torch.tensor([byte_ids], dtype=torch.long, device=device)
    return model.ln(model.embed(idx).unsqueeze(1))  # (1,1,T,D)


def _full_rounds(x: torch.Tensor, model: BDHVBSubspaceDecoder, n_rounds: int, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
    h_prev = x
    for _ in range(n_rounds):
        x_new, _e, _g = torch.utils.checkpoint.checkpoint(_refresh_iteration, x, h_prev, model, B, T, D, nh, N, use_reentrant=False)
        h_prev = x
        x = x_new
    return x


def forward_hz_cq(
    model: BDHVBSubspaceDecoder,
    memory_text: str,
    query_text: str,
    answer_text: str | None,
    n_rounds_per_phase: int,
    n_latent_rounds: int,
    device,
):
    """One HZ-CQ episode forward pass.

    memory_text: the demo IN/OUT/END blocks (real ARC demonstrations).
    query_text: the QUERY/<input grid> block.
    answer_text: the ANSWER/<output grid> block for teacher-forced
        training loss, or None for real held-out-query inference (no
        answer bytes are ever embedded in that case -- the model must
        produce them from H_R alone, matching the real eval protocol).
    n_rounds_per_phase: rounds run after each phase's tokens are
        appended (memory ingestion, query conditioning, answer decode)
        -- this project's usual n_layer=8 full-refresh depth.
    n_latent_rounds: R, the number of latent reasoning-workspace steps
        (one real recurrent round each) between query and answer.

    Returns (logits_over_answer_positions, loss_or_None, final_state_x).
    loss is None when answer_text is None (nothing to supervise).
    """
    C = model.config
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    # Phase 1: task memory S, built from the demonstration text.
    mem_bytes = list(memory_text.encode("utf-8"))
    x = _embed_bytes(model, mem_bytes, device)
    B, _, T, _ = x.shape
    x = _full_rounds(x, model, n_rounds_per_phase, B, T, D, nh, N)

    # Query conditions the memory: H_0 = E_theta(x*, S_K). Appending the
    # query tokens and running more real rounds lets the exact-addressing
    # write at each round see the query against the full memory context.
    query_bytes = list(query_text.encode("utf-8"))
    query_embed = _embed_bytes(model, query_bytes, device)
    x = torch.cat([x, query_embed], dim=2)
    T = x.shape[2]
    x = _full_rounds(x, model, n_rounds_per_phase, B, T, D, nh, N)

    # Phase 2: separate reasoning workspace H, R latent rounds. Each new
    # position starts as a copy of the current last position (no token
    # embedding exists for a latent step) and gets exactly one real
    # recurrent round -- H_{r+1} = F_theta(H_r, S_K), with S_K still
    # reachable through the same sequence's exact addressing.
    for _r in range(n_latent_rounds):
        new_pos = x[:, :, -1:, :]
        x = torch.cat([x, new_pos], dim=2)
        T = x.shape[2]
        x = _full_rounds(x, model, 1, B, T, D, nh, N)

    # Phase 3: decode. Training mode embeds the true answer bytes
    # (teacher forcing) and supervises ordinary next-byte prediction
    # over them; held-out mode stops here and returns the workspace's
    # final state for the caller to decode autoregressively instead.
    if answer_text is None:
        return None, None, x

    answer_bytes = list(answer_text.encode("utf-8"))
    answer_embed = _embed_bytes(model, answer_bytes, device)
    x = torch.cat([x, answer_embed], dim=2)
    T = x.shape[2]
    x = _full_rounds(x, model, n_rounds_per_phase, B, T, D, nh, N)

    # Position (T - len(answer_bytes) - 1 + j) predicts answer_bytes[j]
    # for j = 1..len-1 -- the first answer byte's predecessor is the
    # last query/workspace position, which has no real "next answer
    # byte" target of its own in this framing (matches the same
    # boundary convention progressive-latentization uses).
    n_answer = len(answer_bytes)
    if n_answer < 2:
        return None, None, x
    pred_start = T - n_answer - 1
    pred_positions = x[:, :, pred_start:pred_start + n_answer - 1, :].reshape(-1, D)
    logits = pred_positions @ model.lm_head
    targets = torch.tensor(answer_bytes[1:], dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, targets)
    return logits, loss, x
