"""HZ Next-Phase Plan follow-up: "Split-V BDH" -- a real architecture
variant that gives each attention head its OWN value subspace, instead
of vanilla BDH's real, measured property that every head reads the
SAME full-D-width `V` (see `reference/hz0h_bdh_torch.py`'s `Attention.
forward`: `scores @ V` broadcasts one `(B, 1, T, D)` V across every
head, never splitting it).

Real motivation, from this session's own n_head sweep
(`plans/Deep Reserach Plan.md`'s "Block-diagonal (folded) synaptic
state..." section): shrinking `N` alone via `n_head` (which only
narrows Q/K, not V) made raw-matmul attention dramatically SLOWER, not
faster (~95x at n_head=64 in bf16 on Mac/MPS) -- because `scores @ V`
costs `O(B*nh*T^2*D)` with `D` fixed regardless of `n_head`, so more
heads just repeats the same full-D matmul more times. Split-V tests
the corrected hypothesis: narrow V PER HEAD too (`D/nh` each), so
`scores @ V` becomes `O(B*nh*T^2*(D/nh)) = O(B*T^2*D)`, independent of
`nh`, and persistent streaming state shrinks from `nh*N*D` to
`nh*N*(D/nh) = N*D` -- a real `nh`-times reduction, not just a
relabeling of the same rectangle.

Real, disclosed distinction from Value Bottleneck (`reference/
hz0h_bdh_vb_torch.py`): VB has every head compress the SAME full-D
signal down to a shared `d_state < D` (lossy compression per head). Split-V
instead has the `nh` heads COLLECTIVELY span the full `D`-dimensional
value space (each head owns a disjoint, independently-learned `D/nh`
slice) -- different inductive bias (specialization vs. shared lossy
compression), untested whether it preserves quality better, which is
the real open question this file exists to let someone test.

Architecture (this file's own real addition, not upstream BDH, not
math-equivalent to it -- unlike `hz0h_bdh_fused_attention_torch.py`'s
exact/verified-equivalent chunk_gla rewrite, this genuinely changes
what the model computes and adds real new parameters):

  x  (B, 1, T, D)
   |
   +-- x_sparse = ReLU(x @ encoder)              [UNCHANGED from BDH]
   |
   +-- V_full = x @ W_v                          [NEW: D -> D, shared across depth]
   |    V_split = reshape(V_full) -> (B, nh, T, D/nh)
   |
   +-- yKV_split = Attention(Q=x_sparse, K=x_sparse, V=V_split)
   |    (reuses reference/hz0h_bdh_torch.py's own Attention module
   |    UNCHANGED -- V's last-dim width doesn't affect the RoPE/scores
   |    computation, so plugging in a narrower per-head V is a direct,
   |    exact reuse, not a reimplementation)
   |
   +-- yKV_cat = concat_heads(yKV_split) -> (B, 1, T, D)
   |    yKV = yKV_cat @ W_O                      [NEW: D -> D, "shared
   |                                               cheap output mixer",
   |                                               shared across depth]
   |
   +-- ...everything from here on is BYTE-IDENTICAL to BDH.forward
        (LN, encoder_v, ReLU, x_sparse*y_sparse, decoder, LN residual)

`W_v`/`W_O` are real new learned parameters, shared across all
`n_layer` depth iterations (same "one set of weights reused every
level" philosophy as BDH's own `encoder`/`encoder_v`/`decoder`), adding
`2*D*D` parameters versus vanilla BDH at the same config (e.g. ~524K at
D=512, ~2% of a 25.4M-param model) -- a real, disclosed parameter-count
difference that any quality/efficiency comparison against exact BDH
must account for, not ignore.

STATUS: forward/backward-tested for shape and gradient-flow correctness
only (see tests/reference/test_hz0h_bdh_split_v_torch.py). NOT yet
compared against exact BDH or VB on real quality (CE, passkey,
reassignment) or real training/decode throughput -- do not cite this as
a proven win or loss until that comparison exists.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_torch import Attention, BDHConfig


@dataclasses.dataclass
class BDHSplitVConfig(BDHConfig):
    pass


class BDHSplitV(nn.Module):
    def __init__(self, config: BDHSplitVConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        assert D % nh == 0, f"Split-V needs n_embd ({D}) divisible by n_head ({nh}) to give every head an equal value-subspace width"

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        # NEW vs vanilla BDH: per-head value projection + output mixer,
        # both shared across depth (same philosophy as encoder/encoder_v/decoder).
        self.w_v = nn.Parameter(torch.zeros((D, D)).normal_(std=0.02))
        self.w_o = nn.Parameter(torch.zeros((D, D)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        Dh = D // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for _level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N

            v_full = x @ self.w_v  # B, 1, T, D
            v_split = v_full.view(B, T, nh, Dh).transpose(1, 2)  # B, nh, T, D/nh

            yKV_split = self.attn(Q=x_sparse, K=x_sparse, V=v_split)  # B, nh, T, D/nh
            yKV_cat = yKV_split.transpose(1, 2).reshape(B, 1, T, D)
            yKV = yKV_cat @ self.w_o  # B, 1, T, D
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N
            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def split_v_parameter_count(config: BDHSplitVConfig) -> int:
    """Real parameter count as a function of config, for matching against
    exact BDH/VB at a given target -- avoids needing to instantiate a
    model just to check a candidate config's size."""
    nh = config.n_head
    D = config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    decoder = nh * N * D
    encoder = nh * D * N
    encoder_v = nh * D * N
    embed = config.vocab_size * D
    lm_head = D * config.vocab_size
    w_v = D * D
    w_o = D * D
    return decoder + encoder + encoder_v + embed + lm_head + w_v + w_o
