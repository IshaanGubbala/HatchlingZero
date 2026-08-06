"""HZ-0E E3: routing objectives tests (reference/hz0e_e3_routing_objectives.py).
Checked against the ACTUAL frozen checkpoint and REAL corpus text.
Skips if either is missing locally. Locks in the real, measured
findings from `docs/restart/hz0e_e3_routing_objectives_results.md` as
regression tests -- E3's own exit gate ("balancing does not overwhelm
task learning") plus every named item (LM loss, load balancing, router
z-loss, overflow penalty, diversity regularization, supervised warm
starts).
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e3_routing_objectives import (
    combined_loss, diversity_loss, lm_forward_with_moe, load_balance_loss, overflow_penalty_loss, params_to_dict,
    router_z_loss, supervised_warm_start, train_moe_layer,
)
from reference.hz0e_moe_contract import MoeConfig, init_moe_layer
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import load_real_sequences

TRAIN_PATH = "data/packed/repro_1024_train.jsonl"
VAL_PATH = "data/packed/repro_1024_val.jsonl"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(TRAIN_PATH).exists() or not Path(VAL_PATH).exists(),
    reason="frozen HZ-0A checkpoint / real prose train-val corpus not present locally (gitignored)",
)

CONFIG = MoeConfig()
LR = 1e-4


def _real_batches(path: str, n: int, seq_len: int = 64, offset: int = 0) -> list[mx.array]:
    seqs = load_real_sequences(path, n + offset)[offset:]
    return [mx.array([s[:min(len(s), seq_len)]]) for s in seqs]


def _eval_lm_loss(model, params, batches) -> float:
    pd = params_to_dict(params)
    losses = []
    for tokens in batches:
        loss, _diag, _logits = lm_forward_with_moe(pd, model, tokens, CONFIG, 27)
        mx.eval(loss)
        losses.append(float(loss))
    return sum(losses) / len(losses)


def test_gradient_flows_to_router_through_the_differentiable_gate_weight():
    """The core mechanism E3's training depends on: `mx.grad` through
    the discrete top-1 `argmax` is correctly zero (a discrete choice
    has no gradient), but `router_w` still receives a REAL nonzero
    gradient via the differentiable softmax gate weight that scales
    the selected expert's output -- the standard top-1 MoE training
    signal, confirmed directly rather than assumed."""
    toy_config = MoeConfig(dim=8, dense_d_ff=16, num_experts=4, expert_d_ff=4, capacity_factor=10.0, init_scale=0.3)
    params = params_to_dict(init_moe_layer(toy_config))
    x = mx.random.normal((1, 8, 8), key=mx.random.key(1))
    target = mx.random.normal((1, 8, 8), key=mx.random.key(2))

    from reference.hz0e_moe_contract import MoeLayerParams, moe_ffn_forward

    def loss_fn(p):
        out, _diag = moe_ffn_forward(x, MoeLayerParams(**p), toy_config)
        return mx.mean((out - target) ** 2)

    _loss, grads = mx.value_and_grad(loss_fn)(params)
    mx.eval(grads)
    assert float(mx.sqrt(mx.sum(grads["router_w"] ** 2))) > 0.0
    assert bool(mx.all(mx.isfinite(grads["router_w"])))


def test_lm_loss_training_genuinely_improves_over_fresh_untrained_baseline():
    """The plan's first named item (language-model loss): real
    gradient-descent training on real prose train data must produce a
    GENUINE, held-out-validated improvement over a fresh (untrained)
    MoE layer -- not just "the loop runs," and not just batch-to-batch
    noise (checked on a FIXED held-out validation set, before vs after
    training, matching the results doc's own controlled protocol).
    Checked across 2 seeds for consistency."""
    model, _payload = load_frozen_model()
    train_batches = _real_batches(TRAIN_PATH, 50)
    val_batches = _real_batches(VAL_PATH, 10)

    for seed in (0, 1):
        fresh = init_moe_layer(MoeConfig(init_seed=seed))
        fresh_val = _eval_lm_loss(model, fresh, val_batches)
        trained, _history = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, init_seed=seed)
        trained_val = _eval_lm_loss(model, trained, val_batches)
        assert trained_val < fresh_val, f"seed={seed}: expected real improvement, fresh={fresh_val} trained={trained_val}"


def test_fast_cached_training_is_parameter_exact():
    """Deferred synchronization and skipped Python logging must not
    change the optimizer update sequence for a frozen-backbone run."""
    model, _payload = load_frozen_model()
    batches = _real_batches(TRAIN_PATH, 24)
    detailed, _ = train_moe_layer(
        model, batches, CONFIG, learning_rate=LR, init_seed=11,
        cache_backbone=True, compile_step=True,
    )
    fast, _ = train_moe_layer(
        model, batches, CONFIG, learning_rate=LR, init_seed=11,
        cache_backbone=True, compile_step=True,
        record_history=False, eval_interval=8,
    )
    detailed_dict = params_to_dict(detailed)
    fast_dict = params_to_dict(fast)
    mx.eval(*detailed_dict.values(), *fast_dict.values())
    max_error = max(float(mx.max(mx.abs(detailed_dict[name] - fast_dict[name]))) for name in detailed_dict)
    assert max_error == 0.0, f"fast mode changed parameters: max_error={max_error}"


def test_compiled_scalar_loss_keeps_auxiliary_objectives():
    """Disabling Python breakdown logging must not remove an auxiliary
    objective from the scalar loss used by the compiled gradient path."""
    model, _payload = load_frozen_model()
    tokens = _real_batches(TRAIN_PATH, 1)[0]
    params = params_to_dict(init_moe_layer(CONFIG))
    plain, _ = combined_loss(params, model, tokens, CONFIG, 27, {})
    logged, _ = combined_loss(params, model, tokens, CONFIG, 27, {"z_loss": 0.01})
    scalar, _ = combined_loss(params, model, tokens, CONFIG, 27, {"z_loss": 0.01}, emit_breakdown=False)
    mx.eval(plain, logged, scalar)
    assert float(mx.abs(logged - scalar)) == 0.0
    assert float(mx.abs(logged - plain)) > 0.0


def test_cached_router_warm_start_stays_within_numerical_tolerance():
    """Caching the frozen prefix must preserve supervised router updates
    within the small MLX materialization/kernel-selection tolerance."""
    model, _payload = load_frozen_model()
    domains = {
        "prose": _real_batches(TRAIN_PATH, 2)[0],
        "code": _real_batches(TRAIN_PATH, 2, offset=2)[0],
    }
    labels = {"prose": 0, "code": 1}
    from reference.hz0e_e3_routing_objectives import supervised_warm_start
    initial = init_moe_layer(CONFIG)
    uncached = supervised_warm_start(model, domains, labels, CONFIG, steps=4,
                                     learning_rate=1e-3, start_params=initial)
    cached = supervised_warm_start(model, domains, labels, CONFIG, steps=4,
                                   learning_rate=1e-3, start_params=initial,
                                   cache_backbone=True)
    left, right = params_to_dict(uncached), params_to_dict(cached)
    mx.eval(*left.values(), *right.values())
    max_error = max(float(mx.max(mx.abs(left[name] - right[name]))) for name in left)
    assert max_error <= 2e-4, f"cached warm-start drifted beyond tolerance: max_error={max_error}"


def test_load_balance_auxiliary_loss_reduces_max_expert_share_without_hurting_lm_loss():
    """Load balancing: training WITH the balance loss (weight 0.01,
    the calibrated default) must genuinely reduce the maximum expert
    share on held-out data relative to plain LM-loss-only training,
    confirming the term does what it claims -- and must NOT make LM
    loss meaningfully worse (a small, generous tolerance, not zero
    tolerance, since some real interaction is expected)."""
    model, _payload = load_frozen_model()
    train_batches = _real_batches(TRAIN_PATH, 50)
    val_batches = _real_batches(VAL_PATH, 10)

    plain, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, init_seed=0)
    balanced, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={"balance": 0.01}, learning_rate=LR, init_seed=0)

    def max_share(params, batches):
        pd = params_to_dict(params)
        shares = []
        for tokens in batches:
            _loss, diag, _logits = lm_forward_with_moe(pd, model, tokens, CONFIG, 27)
            mx.eval(diag.expert_idx)
            n = diag.expert_idx.shape[0]
            counts = [int(mx.sum((diag.expert_idx == e).astype(mx.int32))) for e in range(CONFIG.num_experts)]
            shares.append(max(counts) / n)
        return sum(shares) / len(shares)

    plain_share = max_share(plain, val_batches)
    balanced_share = max_share(balanced, val_batches)
    plain_lm = _eval_lm_loss(model, plain, val_batches)
    balanced_lm = _eval_lm_loss(model, balanced, val_batches)

    assert balanced_share < plain_share, f"balance loss should reduce max expert share: plain={plain_share} balanced={balanced_share}"
    assert balanced_lm < plain_lm * 1.05, f"balance loss should not meaningfully hurt LM loss: plain={plain_lm} balanced={balanced_lm}"


def test_router_z_loss_reduces_logit_magnitude_without_hurting_lm_loss():
    """Router z-loss: training WITH z-loss (weight 0.001) must reduce
    the mean `logsumexp(router_logits)^2` on held-out data versus plain
    training, and must not meaningfully hurt LM loss."""
    model, _payload = load_frozen_model()
    train_batches = _real_batches(TRAIN_PATH, 50)
    val_batches = _real_batches(VAL_PATH, 10)

    plain, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, init_seed=0)
    z_trained, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={"z_loss": 0.001}, learning_rate=LR, init_seed=0)

    def mean_z(params, batches):
        pd = params_to_dict(params)
        zs = []
        for tokens in batches:
            _loss, _diag, router_logits = lm_forward_with_moe(pd, model, tokens, CONFIG, 27)
            zs.append(float(router_z_loss(router_logits)))
        return sum(zs) / len(zs)

    plain_z = mean_z(plain, val_batches)
    trained_z = mean_z(z_trained, val_batches)
    plain_lm = _eval_lm_loss(model, plain, val_batches)
    trained_lm = _eval_lm_loss(model, z_trained, val_batches)

    assert trained_z < plain_z, f"z-loss training should reduce logit magnitude: plain={plain_z} trained={trained_z}"
    assert trained_lm < plain_lm * 1.05, f"z-loss should not meaningfully hurt LM loss: plain={plain_lm} trained={trained_lm}"


def test_overflow_penalty_reduces_overflow_rate_without_hurting_lm_loss():
    """Overflow penalty: training WITH the overflow penalty (weight
    1.0) must reduce the mean overflow fraction on held-out data versus
    plain training, and must not meaningfully hurt LM loss."""
    model, _payload = load_frozen_model()
    train_batches = _real_batches(TRAIN_PATH, 50)
    val_batches = _real_batches(VAL_PATH, 10)

    plain, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, init_seed=0)
    overflow_trained, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={"overflow": 1.0}, learning_rate=LR, init_seed=0)

    def mean_overflow(params, batches):
        pd = params_to_dict(params)
        fracs = []
        for tokens in batches:
            _loss, diag, _logits = lm_forward_with_moe(pd, model, tokens, CONFIG, 27)
            mx.eval(diag.overflow)
            fracs.append(float(mx.mean(diag.overflow.astype(mx.float32))))
        return sum(fracs) / len(fracs)

    plain_overflow = mean_overflow(plain, val_batches)
    trained_overflow = mean_overflow(overflow_trained, val_batches)
    plain_lm = _eval_lm_loss(model, plain, val_batches)
    trained_lm = _eval_lm_loss(model, overflow_trained, val_batches)

    assert trained_overflow < plain_overflow, f"overflow penalty should reduce overflow rate: plain={plain_overflow} trained={trained_overflow}"
    assert trained_lm < plain_lm * 1.05, f"overflow penalty should not meaningfully hurt LM loss: plain={plain_lm} trained={trained_lm}"


def test_diversity_regularization_reduces_expert_similarity_without_hurting_lm_loss():
    """Diversity regularization: training WITH the diversity term
    (weight 1000, calibrated against its naturally tiny raw magnitude
    at small-random init) must reduce pairwise expert cosine similarity
    versus plain training, and must not meaningfully hurt LM loss."""
    model, _payload = load_frozen_model()
    train_batches = _real_batches(TRAIN_PATH, 50)
    val_batches = _real_batches(VAL_PATH, 10)

    plain, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, init_seed=0)
    diverse, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={"diversity": 1000.0}, learning_rate=LR, init_seed=0)

    plain_div = float(diversity_loss(plain.expert_gate_w))
    diverse_div = float(diversity_loss(diverse.expert_gate_w))
    plain_lm = _eval_lm_loss(model, plain, val_batches)
    diverse_lm = _eval_lm_loss(model, diverse, val_batches)

    assert diverse_div < plain_div, f"diversity loss should reduce expert similarity: plain={plain_div} diverse={diverse_div}"
    assert diverse_lm < plain_lm * 1.05, f"diversity loss should not meaningfully hurt LM loss: plain={plain_lm} diverse={diverse_lm}"


def test_extreme_auxiliary_weights_do_not_diverge_lm_loss():
    """The exit gate's own stress test: even at weights 1000-10000x the
    calibrated defaults, none of the four auxiliary terms may cause LM
    loss to diverge (NaN/Inf) or blow up far beyond the natural
    baseline range -- confirming the exit gate genuinely holds across a
    wide margin, not just at one hand-picked comfortable weight."""
    model, _payload = load_frozen_model()
    train_batches = _real_batches(TRAIN_PATH, 50)
    val_batches = _real_batches(VAL_PATH, 10)

    extreme_configs = {
        "balance": 10.0, "z_loss": 1.0, "overflow": 10000.0, "diversity": 1e8,
    }
    for name, weight in extreme_configs.items():
        trained, history = train_moe_layer(model, train_batches, CONFIG, aux_weights={name: weight}, learning_rate=LR, init_seed=0)
        assert all(h["total_loss"] == h["total_loss"] for h in history), f"{name}: NaN encountered during training"
        val_loss = _eval_lm_loss(model, trained, val_batches)
        assert val_loss < 5.0, f"{name} at extreme weight {weight}: val LM loss {val_loss} far exceeds the natural baseline range"


def test_supervised_warm_start_does_not_meaningfully_change_final_lm_loss():
    """Supervised warm start: a real, honest NEUTRAL finding -- 20
    steps of router-only supervised classification against hand-
    assigned real domain-to-expert labels, followed by real LM-loss
    training, does not meaningfully change the final held-out LM loss
    versus training from scratch without a warm start. Checked
    directly rather than assumed positive or negative -- the real
    result is "no measurable difference," reported as such."""
    model, _payload = load_frozen_model()
    domain_batches = {}
    for name, path in DOMAIN_DATA_PATHS.items():
        seqs = load_real_sequences(path, 4)
        min_len = min(min(len(s) for s in seqs), 64)
        domain_batches[name] = mx.array([s[:min_len] for s in seqs])
    target_expert = {"prose": 0, "code": 1, "math": 2, "json": 3, "tools": 0}

    train_batches = _real_batches(TRAIN_PATH, 30)
    val_batches = _real_batches(VAL_PATH, 10)

    no_warm, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, init_seed=0)
    warm_params = supervised_warm_start(model, domain_batches, target_expert, CONFIG, steps=20, learning_rate=LR, init_seed=0)
    warm_then_trained, _ = train_moe_layer(model, train_batches, CONFIG, aux_weights={}, learning_rate=LR, start_params=warm_params)

    no_warm_val = _eval_lm_loss(model, no_warm, val_batches)
    warm_val = _eval_lm_loss(model, warm_then_trained, val_batches)
    assert abs(warm_val - no_warm_val) < 0.05, f"expected no large difference: no_warm={no_warm_val} warm={warm_val}"


def test_supervised_warm_start_only_updates_router_weights():
    """Structural check: `supervised_warm_start` must touch ONLY
    `router_w`/`router_b` -- expert and fallback weights must be
    bit-identical to a fresh `init_moe_layer` call, matching the
    module's own documented design (expert specialization is E8's job,
    not E3's warm-start)."""
    model, _payload = load_frozen_model()
    domain_batches = {}
    for name, path in DOMAIN_DATA_PATHS.items():
        seqs = load_real_sequences(path, 4)
        min_len = min(min(len(s) for s in seqs), 64)
        domain_batches[name] = mx.array([s[:min_len] for s in seqs])
    target_expert = {"prose": 0, "code": 1, "math": 2, "json": 3, "tools": 0}

    fresh = init_moe_layer(MoeConfig(init_seed=0))
    warmed = supervised_warm_start(model, domain_batches, target_expert, CONFIG, steps=5, learning_rate=LR, init_seed=0)

    assert bool(mx.array_equal(warmed.expert_gate_w, fresh.expert_gate_w))
    assert bool(mx.array_equal(warmed.expert_up_w, fresh.expert_up_w))
    assert bool(mx.array_equal(warmed.expert_down_w, fresh.expert_down_w))
    assert bool(mx.array_equal(warmed.fallback_gate_w, fresh.fallback_gate_w))
    assert not bool(mx.array_equal(warmed.router_w, fresh.router_w)), "router_w should have actually changed"
