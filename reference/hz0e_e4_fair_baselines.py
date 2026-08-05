"""HZ-0E E4: fair baselines.

Per the plan's own E4 text: "Compare with dense MLPs at matched active
and total parameters, wider dense MLPs, domain adapters, static expert
assignment, and shared-expert-only models. Always report total and
active parameters separately."

Every baseline here is trained via the SAME real protocol E3 proved
working (`lr=1e-4`, real `mlx.optimizers.Adam`, real disjoint prose
train/val corpus splits, layer 27, frozen backbone otherwise) -- a
fair, apples-to-apples comparison, not different recipes per baseline.
A single generic trainer (`train_generic`) is reused across every
baseline so training MECHANICS (optimizer, step count, data, learning
rate) are held identical; only the forward computation and trainable
parameter set differ.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams, init_moe_layer, moe_ffn_forward
from reference.hz0e_e3_routing_objectives import params_to_dict


def _backbone_prefix_and_suffix(model, tokens: mx.array, layer_index: int):
    """Runs the real frozen backbone up through block `layer_index`'s
    own mixer+residual, returning `(ffn_input, residual_x)` -- the
    exact real FFN input and the residual stream it will be added back
    onto, matching `reference/hz0e_e2_router_simulator.py::collect_real_ffn_input`'s
    own (bug-fixed) computation."""
    x = model.embedding(tokens)
    for i in range(layer_index):
        x, _ = model.blocks[i](x, None)
    block = model.blocks[layer_index]
    mixed, _ = block.mixer(block.norm1(x), None)
    x = x + mixed
    ffn_input = block.norm2(x)
    return ffn_input, x


def _finish_and_lm_loss(model, x_after_ffn: mx.array, tokens: mx.array, layer_index: int) -> mx.array:
    for i in range(layer_index + 1, len(model.blocks)):
        x_after_ffn, _ = model.blocks[i](x_after_ffn, None)
    logits = mx.matmul(model.final_norm(x_after_ffn), model.embedding.weight.T)
    return mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))


def _swiglu(x, gate_w, gate_b, up_w, up_b, down_w, down_b):
    return (nn.silu(x @ gate_w.T + gate_b) * (x @ up_w.T + up_b)) @ down_w.T + down_b


def _init_ffn(dim: int, d_ff: int, seed: int, scale: float = 0.02) -> dict[str, mx.array]:
    key = mx.random.key(seed)
    k = mx.random.split(key, 3)
    return {
        "gate_w": mx.random.normal((d_ff, dim), key=k[0]) * scale, "gate_b": mx.zeros((d_ff,)),
        "up_w": mx.random.normal((d_ff, dim), key=k[1]) * scale, "up_b": mx.zeros((d_ff,)),
        "down_w": mx.random.normal((dim, d_ff), key=k[2]) * scale, "down_b": mx.zeros((dim,)),
    }


def dense_ffn_param_count(dim: int, d_ff: int) -> int:
    return (dim * d_ff + d_ff) * 2 + (d_ff * dim + dim)


def train_generic(model, train_batches: list[mx.array], init_fn, loss_fn, *, learning_rate: float = 1e-4) -> tuple[dict[str, mx.array], list[float]]:
    """The one shared trainer every E4 baseline (and E1/E3's own MoE
    mechanism, for a fully consistent re-comparison) runs through --
    real `mx.value_and_grad` + real `mlx.optimizers.Adam`, matching
    E3's own proven-working recipe exactly. `init_fn() -> params_dict`,
    `loss_fn(params_dict, tokens) -> scalar`."""
    params = init_fn()
    optimizer = optim.Adam(learning_rate=learning_rate)
    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    losses = []
    for tokens in train_batches:
        loss, grads = grad_fn(params, tokens)
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params)
        losses.append(float(loss))
    return params, losses


def eval_generic(model, val_batches: list[mx.array], params: dict[str, mx.array], loss_fn) -> float:
    losses = []
    for tokens in val_batches:
        loss = loss_fn(params, tokens)
        mx.eval(loss)
        losses.append(float(loss))
    return sum(losses) / len(losses)


# --- Baseline A: no adaptation (plain frozen dense FFN, untouched) ---

def no_adaptation_loss(model, tokens: mx.array, layer_index: int) -> float:
    """Not trained at all -- the real, original pretrained dense FFN at
    `layer_index`, completely untouched. The floor every other baseline
    (and MoE itself) must beat to be worth anything."""
    ffn_input, x = _backbone_prefix_and_suffix(model, tokens, layer_index)
    block = model.blocks[layer_index]
    mlp = _swiglu(ffn_input, block.gate.weight, block.gate.bias, block.up.weight, block.up.bias, block.down.weight, block.down.bias)
    x = x + mlp
    loss = _finish_and_lm_loss(model, x, tokens, layer_index)
    mx.eval(loss)
    return float(loss)


# --- Baselines B/C/D: dense FFN at a chosen width (matched-active, matched-total, wider) ---

def make_dense_baseline(model, config: MoeConfig, layer_index: int, d_ff: int, seed: int):
    dim = config.dim

    def init_fn():
        return _init_ffn(dim, d_ff, seed)

    def loss_fn(params, tokens):
        ffn_input, x = _backbone_prefix_and_suffix(model, tokens, layer_index)
        mlp = _swiglu(ffn_input, params["gate_w"], params["gate_b"], params["up_w"], params["up_b"], params["down_w"], params["down_b"])
        x = x + mlp
        return _finish_and_lm_loss(model, x, tokens, layer_index)

    return init_fn, loss_fn, dense_ffn_param_count(dim, d_ff)


# --- Baseline E: domain adapter (a real trainable low-rank ADDITIVE adapter on the FROZEN original dense FFN's output) ---

def make_adapter_baseline(model, config: MoeConfig, layer_index: int, rank: int, seed: int):
    """The frozen ORIGINAL dense FFN runs unchanged (never trained,
    matching D4's own precedent for what an "adapter" baseline means --
    the pretrained path stays frozen); a real, TRAINED low-rank
    additive delta (`a @ b`, `[dim, rank] @ [rank, dim]`) is added onto
    the FFN's output. Unlike HZ-0D's D4 `static_random_adapter`
    baseline (deliberately never trained, to isolate "extra capacity
    alone"), this adapter IS trained -- matching E4's plan text naming
    "domain adapters" as a genuinely competing, trainable alternative
    to MoE, not a static-capacity control."""
    dim = config.dim

    def init_fn():
        key = mx.random.key(seed)
        k1, k2 = mx.random.split(key)
        return {"a": mx.random.normal((dim, rank), key=k1) * 0.02, "b": mx.zeros((rank, dim))}

    def loss_fn(params, tokens):
        ffn_input, x = _backbone_prefix_and_suffix(model, tokens, layer_index)
        block = model.blocks[layer_index]
        mlp = _swiglu(ffn_input, block.gate.weight, block.gate.bias, block.up.weight, block.up.bias, block.down.weight, block.down.bias)
        adapter_delta = ffn_input @ params["a"] @ params["b"]
        x = x + mlp + adapter_delta
        return _finish_and_lm_loss(model, x, tokens, layer_index)

    adapter_params = 2 * dim * rank
    return init_fn, loss_fn, adapter_params


# --- Baseline F: static expert assignment (no router at all, fixed deterministic assignment) ---

def make_static_expert_baseline(model, config: MoeConfig, layer_index: int, seed: int):
    """The SAME 4 experts + shared fallback structure and parameter
    budget as E1's real MoE contract, but with NO router at all --
    every token is assigned to an expert by a fixed, non-learned rule
    (token position modulo `num_experts`), matching the plan's own
    "static expert assignment" naming. Isolates whether LEARNED routing
    itself (E1/E3's contribution) adds value beyond simply having 4
    separate small expert MLPs process different (arbitrarily-split)
    token subsets. No capacity/overflow/fallback logic is needed here
    -- a fixed modulo assignment is exactly balanced by construction, so
    the shared fallback never engages (0 fallback params trained)."""
    dim, rank, e = config.dim, config.expert_d_ff, config.num_experts

    def init_fn():
        moe = init_moe_layer(MoeConfig(dim=dim, dense_d_ff=config.dense_d_ff, num_experts=e, expert_d_ff=rank, capacity_factor=config.capacity_factor, init_seed=seed))
        # only the 4 experts are trainable parameters here -- no router, no fallback (never engaged)
        return {
            "expert_gate_w": moe.expert_gate_w, "expert_gate_b": moe.expert_gate_b,
            "expert_up_w": moe.expert_up_w, "expert_up_b": moe.expert_up_b,
            "expert_down_w": moe.expert_down_w, "expert_down_b": moe.expert_down_b,
        }

    def loss_fn(params, tokens):
        ffn_input, x = _backbone_prefix_and_suffix(model, tokens, layer_index)
        batch, seq, d = ffn_input.shape
        n = batch * seq
        flat = ffn_input.reshape(n, d)
        assignment = mx.arange(n) % e  # fixed, non-learned, exactly balanced
        output = mx.zeros((n, d))
        for expert in range(e):
            expert_out = _swiglu(
                flat, params["expert_gate_w"][expert], params["expert_gate_b"][expert],
                params["expert_up_w"][expert], params["expert_up_b"][expert],
                params["expert_down_w"][expert], params["expert_down_b"][expert],
            )
            mask = (assignment == expert).astype(mx.float32)
            output = output + expert_out * mask[:, None]
        mlp = output.reshape(batch, seq, d)
        x = x + mlp
        return _finish_and_lm_loss(model, x, tokens, layer_index)

    expert_params = e * dense_ffn_param_count(dim, rank)
    return init_fn, loss_fn, expert_params


# --- Baseline G: shared-expert-only (bypass the 4 specialized experts entirely, always use ONE shared dense path) ---

def make_shared_expert_only_baseline(model, config: MoeConfig, layer_index: int, seed: int):
    """No routing, no specialized experts at all -- every token is
    processed by ONE shared, full-dense-width FFN (matching E1's
    fallback's own shape, `dense_d_ff`), trained fresh (not reusing E1's
    fallback weights, so this is a clean, independent baseline, not a
    biased reuse of a jointly-trained component). Tests whether having
    just ONE (larger) shared path, with zero specialization, is
    sufficient -- the real "shared-expert" alternative E1's contract
    doc explicitly considered and rejected as the FALLBACK's own
    semantics, here tested directly as its own real baseline."""
    dim, dff = config.dim, config.dense_d_ff

    def init_fn():
        return _init_ffn(dim, dff, seed)

    def loss_fn(params, tokens):
        ffn_input, x = _backbone_prefix_and_suffix(model, tokens, layer_index)
        mlp = _swiglu(ffn_input, params["gate_w"], params["gate_b"], params["up_w"], params["up_b"], params["down_w"], params["down_b"])
        x = x + mlp
        return _finish_and_lm_loss(model, x, tokens, layer_index)

    return init_fn, loss_fn, dense_ffn_param_count(dim, dff)


# --- MoE itself (E1/E3's own mechanism), reusable here for a fully consistent E4 re-comparison ---

def make_moe_baseline(model, config: MoeConfig, layer_index: int, seed: int):
    def init_fn():
        return params_to_dict(init_moe_layer(MoeConfig(
            dim=config.dim, dense_d_ff=config.dense_d_ff, num_experts=config.num_experts,
            expert_d_ff=config.expert_d_ff, capacity_factor=config.capacity_factor, init_seed=seed,
        )))

    def loss_fn(params, tokens):
        ffn_input, x = _backbone_prefix_and_suffix(model, tokens, layer_index)
        moe_out, _diag = moe_ffn_forward(ffn_input, MoeLayerParams(**params), config)
        x = x + moe_out
        return _finish_and_lm_loss(model, x, tokens, layer_index)

    from reference.hz0e_moe_contract import moe_layer_param_counts
    counts = moe_layer_param_counts(config)
    return init_fn, loss_fn, counts["moe_layer_total"], counts["moe_layer_active_no_overflow"]
