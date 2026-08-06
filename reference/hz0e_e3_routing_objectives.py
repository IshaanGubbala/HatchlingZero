"""HZ-0E E3: routing objectives.

Per the plan's own E3 text: "Evaluate language-model loss, load
balancing, router z-loss, overflow penalty, diversity regularization,
and supervised warm starts. Exit gate: balancing does not overwhelm
task learning."

This is the first HZ-0E phase requiring REAL gradient-based training
(E1/E2 were static contract + untrained-mechanism checks). Trains
E1's MoE layer (`reference/hz0e_moe_contract.py`) at its real target
layer (27, isolated -- blocks 28/29/30 keep their original dense FFN,
matching E1/E2's own single-layer-isolated testing convention, not yet
the full 3-layer conversion) against a REAL next-token LM loss on real
corpus text, with optional auxiliary loss terms, and measures whether
adding those terms hurts real task-loss convergence -- the literal
exit-gate question, not assumed.

Gradient flow verified directly before this module was trusted (see
`docs/restart/hz0e_e3_routing_objectives_results.md`): MLX's `mx.grad`
over the discrete top-1 `argmax` routing decision itself is correctly
zero (a discrete choice has no gradient), but the router IS trained via
the differentiable softmax GATE WEIGHT that scales the selected
expert's output -- the standard top-1 MoE training mechanism, confirmed
via a direct smoke test (nonzero `router_w` gradient) before any real
training was run.
"""
from __future__ import annotations

from dataclasses import asdict, fields

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from reference.hz0e_moe_contract import MoeConfig, MoeDiagnostics, MoeLayerParams, init_moe_layer, moe_ffn_forward


def params_to_dict(params: MoeLayerParams) -> dict[str, mx.array]:
    return asdict(params)


def dict_to_params(d: dict[str, mx.array]) -> MoeLayerParams:
    return MoeLayerParams(**{f.name: d[f.name] for f in fields(MoeLayerParams)})


def _moe_prefix(model, tokens: mx.array, layer_index: int) -> tuple[mx.array, mx.array]:
    """Return the frozen residual stream and MoE input before the trainable layer."""
    x = model.embedding(tokens)
    for i in range(layer_index):
        x, _ = model.blocks[i](x, None)
    block = model.blocks[layer_index]
    mixed, _ = block.mixer(block.norm1(x), None)
    x = x + mixed
    return x, block.norm2(x)


def lm_forward_with_moe(
    params_dict: dict[str, mx.array], model, tokens: mx.array, config: MoeConfig,
    layer_index: int, cached_prefix: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, MoeDiagnostics, mx.array]:
    """Real next-token LM loss, real frozen HZ-0A backbone, with block
    `layer_index`'s dense FFN REPLACED by E1's MoE layer (trainable),
    every other block (including 28/29/30, matching E1/E2's own
    isolated-single-layer convention) left as the original frozen dense
    FFN. Returns `(lm_loss, diagnostics, router_logits)` -- the router
    logits are returned SEPARATELY (recomputed identically to
    `moe_ffn_forward`'s own internal computation, same op, same
    gradient path) since `moe_ffn_forward` itself only returns the
    top-1 gate weight in its diagnostics, not the full pre-softmax
    logit vector the z-loss and load-balance terms need."""
    moe_params = dict_to_params(params_dict)
    if cached_prefix is None:
        x, ffn_input = _moe_prefix(model, tokens, layer_index)
    else:
        x, ffn_input = cached_prefix

    batch, seq, dim = ffn_input.shape
    ffn_input_flat = ffn_input.reshape(batch * seq, dim)
    router_logits = ffn_input_flat @ moe_params.router_w.T + moe_params.router_b

    moe_out, diagnostics = moe_ffn_forward(ffn_input, moe_params, config)
    x = x + moe_out

    for i in range(layer_index + 1, len(model.blocks)):
        x, _ = model.blocks[i](x, None)
    logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
    lm_loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))
    return lm_loss, diagnostics, router_logits


def load_balance_loss(diagnostics: MoeDiagnostics, router_probs: mx.array, num_experts: int) -> mx.array:
    """The standard Switch-Transformer auxiliary load-balance loss:
    `num_experts * sum_e(f_e * P_e)`, where `f_e` is the (non-
    differentiable, stop-gradient) FRACTION of tokens whose top-1
    argmax chose expert `e`, and `P_e` is the (differentiable) MEAN
    router softmax probability mass on expert `e` across all tokens.
    Minimized at `1.0` when routing is perfectly uniform; grows above
    `1.0` as routing concentrates on fewer experts."""
    n = router_probs.shape[0]
    one_hot = (diagnostics.expert_idx[:, None] == mx.arange(num_experts)[None, :]).astype(mx.float32)
    f = mx.stop_gradient(mx.mean(one_hot, axis=0))  # [E], fraction of tokens per expert (hard, no grad)
    p = mx.mean(router_probs, axis=0)               # [E], mean soft probability per expert (differentiable)
    return num_experts * mx.sum(f * p)


def router_z_loss(router_logits: mx.array) -> mx.array:
    """ST-MoE's router z-loss: mean over tokens of
    `logsumexp(router_logits)^2` -- penalizes large router logit
    magnitudes directly (a real stability regularizer, independent of
    load balance: a router can be perfectly balanced while still having
    huge, poorly-conditioned logits)."""
    log_sum_exp = mx.logsumexp(router_logits, axis=-1)
    return mx.mean(log_sum_exp ** 2)


def overflow_penalty_loss(diagnostics: MoeDiagnostics) -> mx.array:
    """A real, DIRECTLY TRAINABLE overflow penalty, distinct from load
    balance: `mean(gate_weight * overflow_mask)`. `overflow_mask` is a
    stop-gradient 0/1 constant (which tokens overflowed is a discrete
    capacity decision), but `gate_weight` (the router's own confidence
    in its now-rejected choice) IS differentiable -- minimizing this
    pushes DOWN the router's confidence specifically for tokens whose
    chosen expert it couldn't actually get capacity from, a real signal
    distinct from the aggregate-probability-mass balance loss above."""
    overflow_mask = mx.stop_gradient(diagnostics.overflow.astype(mx.float32))
    return mx.mean(diagnostics.gate_weight * overflow_mask)


def diversity_loss(expert_gate_w: mx.array) -> mx.array:
    """Real diversity regularizer: mean SQUARED pairwise cosine
    similarity between experts' `gate_w` matrices (flattened),
    `expert_gate_w`: `[num_experts, expert_d_ff, dim]`. Minimized
    (pushed toward 0) when experts' weights are maximally dissimilar;
    squared so both positive and negative similarity are penalized
    (either extreme means experts are NOT independent)."""
    num_experts = expert_gate_w.shape[0]
    flat = expert_gate_w.reshape(num_experts, -1)
    norm = flat / mx.maximum(mx.linalg.norm(flat, axis=-1, keepdims=True), 1e-8)
    sim_matrix = norm @ norm.T  # [E, E]
    mask = 1.0 - mx.eye(num_experts)  # zero the diagonal (self-similarity)
    num_pairs = num_experts * (num_experts - 1)
    return mx.sum((sim_matrix ** 2) * mask) / num_pairs


def combined_loss(params_dict: dict[str, mx.array], model, tokens: mx.array, config: MoeConfig, layer_index: int, aux_weights: dict[str, float], cached_prefix: tuple[mx.array, mx.array] | None = None, emit_breakdown: bool = True) -> tuple[mx.array, dict[str, float]]:
    """`lm_loss + sum(weight * aux_term)` for every aux term named in
    `aux_weights` with a nonzero weight -- `{}` or all-zero weights
    reduces exactly to plain LM-loss training. Returns `(total_loss,
    breakdown_dict)` for real per-term logging, not just a scalar."""
    lm_loss, diagnostics, router_logits = lm_forward_with_moe(params_dict, model, tokens, config, layer_index, cached_prefix)
    router_probs = mx.softmax(router_logits, axis=-1)
    total = lm_loss
    breakdown = {"lm_loss": float(lm_loss)} if emit_breakdown else {}

    if aux_weights.get("balance", 0.0) != 0.0:
        term = load_balance_loss(diagnostics, router_probs, config.num_experts)
        total = total + aux_weights["balance"] * term
        if emit_breakdown:
            breakdown["balance_loss"] = float(term)
    if aux_weights.get("z_loss", 0.0) != 0.0:
        term = router_z_loss(router_logits)
        total = total + aux_weights["z_loss"] * term
        if emit_breakdown:
            breakdown["z_loss"] = float(term)
    if aux_weights.get("overflow", 0.0) != 0.0:
        term = overflow_penalty_loss(diagnostics)
        total = total + aux_weights["overflow"] * term
        if emit_breakdown:
            breakdown["overflow_loss"] = float(term)
    if aux_weights.get("diversity", 0.0) != 0.0:
        moe_params = dict_to_params(params_dict)
        term = diversity_loss(moe_params.expert_gate_w)
        total = total + aux_weights["diversity"] * term
        if emit_breakdown:
            breakdown["diversity_loss"] = float(term)

    if emit_breakdown:
        breakdown["total_loss"] = float(total)
    return total, breakdown


def train_moe_layer(model, train_batches: list[mx.array], config: MoeConfig, *, layer_index: int = 27, aux_weights: dict[str, float] | None = None, learning_rate: float = 3e-3, init_seed: int = 0, start_params: MoeLayerParams | None = None, cache_backbone: bool = False, compile_step: bool = False, record_history: bool = True, eval_interval: int = 1, weight_decay: float | None = None) -> tuple[MoeLayerParams, list[dict]]:
    """Real gradient-descent training of E1's MoE layer against real
    LM loss (+ optional auxiliary terms) on a real sequence of
    real-token batches, one gradient step per batch. Returns the final
    trained `MoeLayerParams` and a full per-step breakdown history
    (every loss term, every step -- not just a final number).

    `start_params`: if given, training CONTINUES from this state
    (e.g. the output of `supervised_warm_start`) instead of a fresh
    `init_moe_layer` call -- makes warm-start-then-train a real,
    composable two-call sequence through this module's own public API,
    not ad-hoc duplicated training-loop code."""
    aux_weights = aux_weights or {}
    if eval_interval < 1:
        raise ValueError(f"eval_interval must be positive, got {eval_interval}")
    initial = start_params if start_params is not None else init_moe_layer(MoeConfig(
        dim=config.dim, dense_d_ff=config.dense_d_ff, num_experts=config.num_experts,
        expert_d_ff=config.expert_d_ff, capacity_factor=config.capacity_factor, init_seed=init_seed,
    ))
    params_dict = params_to_dict(initial)
    cached_prefixes = None
    if cache_backbone:
        # The prefix is frozen and independent of MoE parameters. Materialize
        # it once per batch to avoid replaying 27 full transformer blocks on
        # every optimizer step. The suffix remains in the gradient graph.
        cached_prefixes = []
        for tokens in train_batches:
            prefix = _moe_prefix(model, tokens, layer_index)
            mx.eval(*prefix)
            cached_prefixes.append(prefix)
    optimizer = (optim.Adam(learning_rate=learning_rate) if weight_decay is None
                 else optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay))
    apply_update = optimizer.apply_gradients

    def loss_fn(p, tokens, cached_prefix=None):
        return combined_loss(p, model, tokens, config, layer_index, aux_weights, cached_prefix)

    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    if compile_step:
        # The frozen model and batch shapes are constant for a curriculum;
        # compile only the pure params -> loss/grads function, leaving the
        # optimizer state update outside the compiled graph.
        def scalar_loss_fn(p, tokens, cached_prefix=None):
            return combined_loss(p, model, tokens, config, layer_index, aux_weights, cached_prefix, emit_breakdown=False)[0]

        grad_fn = mx.compile(mx.value_and_grad(scalar_loss_fn, argnums=0))
    history = []
    for index, tokens in enumerate(train_batches):
        cached = None if cached_prefixes is None else cached_prefixes[index]
        if compile_step:
            _total, grads = grad_fn(params_dict, tokens, cached)
            # E8's production curriculum uses no auxiliary terms, so the
            # compiled scalar is also the complete per-step breakdown.
            if aux_weights and record_history:
                _logged_total, breakdown = loss_fn(params_dict, tokens, cached)
            elif record_history:
                breakdown = {"lm_loss": float(_total), "total_loss": float(_total)}
            else:
                breakdown = {}
        else:
            (_total, breakdown), grads = grad_fn(params_dict, tokens, cached)
        params_dict = apply_update(grads, params_dict)
        if (index + 1) % eval_interval == 0 or index == len(train_batches) - 1:
            mx.eval(params_dict)
        if record_history:
            history.append(breakdown)
    return dict_to_params(params_dict), history


def supervised_warm_start(model, domain_batches: dict[str, mx.array], target_expert: dict[str, int], config: MoeConfig, *, layer_index: int = 27, steps: int = 20, learning_rate: float = 3e-3, init_seed: int = 0, start_params: MoeLayerParams | None = None, cache_backbone: bool = False, compile_step: bool = False) -> MoeLayerParams:
    """Trains ONLY the router (`router_w`/`router_b`) via real
    cross-entropy classification against `target_expert`'s real
    domain-to-expert label assignment, for `steps` real gradient steps
    -- a genuine supervised pre-training phase before task-loss
    training begins, matching the plan's own "supervised warm starts"
    text. Expert/fallback weights are NOT touched here (they get no
    supervised target at this stage -- routing supervision comes first,
    expert specialization comes later via task loss, matching the
    plan's own phase ordering: E3 warm-starts routing, E8 is the later
    "specialization curriculum")."""
    initial = start_params or init_moe_layer(MoeConfig(
        dim=config.dim, dense_d_ff=config.dense_d_ff, num_experts=config.num_experts,
        expert_d_ff=config.expert_d_ff, capacity_factor=config.capacity_factor, init_seed=init_seed,
    ))
    params_dict = params_to_dict(initial)
    if any(label < 0 or label >= config.num_experts for label in target_expert.values()):
        raise ValueError(f"target_expert labels must be in [0, {config.num_experts}), got {target_expert}")
    optimizer = optim.Adam(learning_rate=learning_rate)
    domains = list(domain_batches.keys())
    cached_prefixes = {}
    if cache_backbone:
        for domain in domains:
            _residual, ffn_input = _moe_prefix(model, domain_batches[domain], layer_index)
            mx.eval(ffn_input)
            # Router supervision only consumes the normalized FFN input;
            # do not retain the unused residual stream in the cache.
            cached_prefixes[domain] = ffn_input

    def router_loss_fn(p, tokens, label, cached_prefix=None):
        if cached_prefix is None:
            _x, ffn_input = _moe_prefix(model, tokens, layer_index)
        else:
            ffn_input = cached_prefix
        batch, seq, dim = ffn_input.shape
        ffn_input_flat = ffn_input.reshape(batch * seq, dim)
        router_logits = ffn_input_flat @ p["router_w"].T + p["router_b"]
        labels = mx.full((ffn_input_flat.shape[0],), label, dtype=mx.int32)
        return mx.mean(nn.losses.cross_entropy(router_logits, labels))

    grad_fn = mx.value_and_grad(router_loss_fn, argnums=0)
    if compile_step:
        grad_fn = mx.compile(grad_fn)
    for step in range(steps):
        domain = domains[step % len(domains)]
        tokens = domain_batches[domain]
        label = target_expert[domain]
        cached = cached_prefixes.get(domain)
        _loss, grads = grad_fn(params_dict, tokens, label, cached)
        # only router_w/router_b receive real gradients here since only
        # they appear in router_loss_fn's forward graph; the optimizer
        # applies zero-valued updates to any params with implicit zero
        # gradient (none arise here since grads is a dict matching
        # params_dict's keys, with router-only entries populated by
        # mx.grad and the rest naturally absent from the loss graph --
        # apply_gradients only touches keys present in `grads`).
        router_only_grads = {"router_w": grads["router_w"], "router_b": grads["router_b"]}
        router_only_params = {"router_w": params_dict["router_w"], "router_b": params_dict["router_b"]}
        updated = optimizer.apply_gradients(router_only_grads, router_only_params)
        params_dict["router_w"] = updated["router_w"]
        params_dict["router_b"] = updated["router_b"]
        mx.eval(params_dict)
    return dict_to_params(params_dict)
