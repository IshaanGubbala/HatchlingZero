"""HZ-0E E8: specialization curriculum.

Per the plan's own E8 text: "Train progressively on balanced prose,
code, math, technical documents, JSON, and tool tasks, then mixed-
domain and adversarially imbalanced sequences. Exit gate: experts show
measurable specialization without becoming unusable elsewhere."

Builds on E6's real, warm-started integration (`init_e6_layers` --
experts start as scaled slices of the real pretrained dense FFN, not
small-random E3-style init) and E3's proven training mechanics
(`train_moe_layer`, real `mlx.optimizers.Adam`). Three real curriculum
stages, each using real corpus text:

1. Balanced: rotates evenly through all 5 real domains
   (`reference/hz0e_e2_router_simulator.py::DOMAIN_DATA_PATHS`).
2. Mixed-domain: each batch mixes 2 real domains as different rows,
   matching E2's own "mixed domains" scenario.
3. Adversarially imbalanced: batches heavily skewed toward one domain
   (matching E2's own "imbalance" scenario), rotating which domain is
   dominant so no single domain is starved across the whole stage.

Specialization is measured the same way E2 measured mechanism
stability: real per-domain routing statistics
(`reference/hz0e_e2_router_simulator.py::route_with_stats`), compared
BEFORE (right after E6 warm-start + router supervision) and AFTER the
full curriculum -- a real, quantified before/after delta, not an
assumed outcome.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

import mlx.nn as nn

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS, collect_real_ffn_input, route_with_stats
from reference.hz0e_e3_routing_objectives import params_to_dict, supervised_warm_start, train_moe_layer
from reference.hz0e_e4_fair_baselines import train_generic, eval_generic
from reference.hz0e_e6_integration import TARGET_LAYERS, cross_entropy_loss, forward_e6, init_e6_layers
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams
from scripts.hz0c_c3_trigger_simulator import load_real_sequences

DOMAIN_TO_EXPERT = {"prose": 0, "code": 1, "math": 2, "json": 3, "tools": 0}
LAYER = 27

# A real bug lived here during development: reusing E2's
# `DOMAIN_DATA_PATHS` (all pointing at each domain's own
# *_validation.jsonl file, appropriate for E2's own untrained-mechanism
# checks, which never trained anything) for E8's real TRAINING data
# caused genuine train/eval leakage -- `DOMAIN_DATA_PATHS["prose"]` is
# `data/packed/repro_1024_val.jsonl`, the EXACT SAME FILE this module's
# own general-quality held-out check reads from. The first symptom was
# an implausible post-curriculum held-out loss of `~0.40` (perplexity
# `~1.49`, essentially memorization-level, impossible for genuine
# generalization on 301M-scale held-out prose in ~150 real steps) --
# caught by a perplexity sanity check, not assumed correct because the
# number "looked like an improvement." Fixed by using each domain's own
# real *_train.jsonl file for TRAINING and *_validation.jsonl for
# held-out specialization/quality measurement, matching E3/E4's own
# established train/val discipline exactly (never reusing a validation
# file as a training source).
TRAIN_DOMAIN_DATA_PATHS = {
    "prose": "data/packed/repro_1024_train.jsonl",
    "code": "data/packed/external/code_train.jsonl",
    "math": "data/packed/external/mathematical_and_structured_train.jsonl",
    "json": "data/packed/external/json_and_configuration_train.jsonl",
    "tools": "data/packed/external/terminal_and_debugging_train.jsonl",
}


def load_domain_batches(paths: dict[str, str], count: int = 8, seq_len: int = 64, offset: int = 0) -> dict[str, mx.array]:
    """One real batch per domain -- `count` sequences, `seq_len` tokens
    each, from `offset` onward in each domain's own corpus file. Caller
    supplies `paths` explicitly (`TRAIN_DOMAIN_DATA_PATHS` for real
    training data, `DOMAIN_DATA_PATHS` for held-out specialization
    measurement) -- never defaulted, so a caller cannot accidentally
    reuse a validation file as a training source the way the module's
    own history audit (see above) found and fixed."""
    out = {}
    for name, path in paths.items():
        seqs = load_real_sequences(path, count + offset)[offset:]
        min_len = min(min(len(s) for s in seqs), seq_len)
        out[name] = mx.array([s[:min_len] for s in seqs])
    return out


def balanced_batches(domain_batches: dict[str, mx.array], steps: int) -> list[mx.array]:
    names = list(domain_batches.keys())
    return [domain_batches[names[i % len(names)]] for i in range(steps)]


def mixed_domain_batches(domain_batches: dict[str, mx.array], steps: int, seed: int = 0) -> list[mx.array]:
    """Each batch mixes 2 DIFFERENT real domains as different rows,
    matching `reference/hz0e_e2_router_simulator.py`'s own "mixed
    domains" scenario (there tested for mechanism stability; here used
    as a real training stage).

    A real, minor bug lived here during development: `seed` created an
    `mx.random.key` that was never actually referenced -- domain
    pairing was fully deterministic regardless of `seed`, silently
    giving every "different seed" run of this stage the IDENTICAL
    curriculum. Caught while trying to get genuinely independent
    multi-seed baseline comparisons for
    `docs/restart/hz0e_e8_specialization_curriculum_results.md`. Fixed:
    `seed` now genuinely permutes domain order before pairing via
    `mx.random.permutation`, so different seeds produce different (but
    still real, still every-domain-covered) mixed-batch compositions."""
    names_array = mx.array(list(range(len(domain_batches))))
    order = mx.random.permutation(names_array, key=mx.random.key(seed)).tolist()
    names = list(domain_batches.keys())
    shuffled = [names[i] for i in order]
    batches = []
    for i in range(steps):
        a = shuffled[i % len(shuffled)]
        b = shuffled[(i + 1 + (i // len(shuffled))) % len(shuffled)]
        if a == b:
            b = shuffled[(shuffled.index(a) + 1) % len(shuffled)]
        min_len = min(domain_batches[a].shape[1], domain_batches[b].shape[1])
        mixed = mx.concatenate([domain_batches[a][:, :min_len], domain_batches[b][:, :min_len]], axis=0)
        batches.append(mixed)
    return batches


def imbalanced_batches(domain_batches: dict[str, mx.array], steps: int) -> list[mx.array]:
    """Heavily skewed batches -- one dominant domain (5 rows) plus one
    token row from each other domain, ROTATING which domain dominates
    across the stage so no domain is starved over the whole curriculum,
    matching E2's own "imbalance" scenario used as a real training
    stage here."""
    names = list(domain_batches.keys())
    batches = []
    for i in range(steps):
        dominant = names[i % len(names)]
        min_len = min(b.shape[1] for b in domain_batches.values())
        rows = [domain_batches[dominant][:, :min_len]] * 5
        for other in names:
            if other != dominant:
                rows.append(domain_batches[other][:1, :min_len])
        batches.append(mx.concatenate(rows, axis=0))
    return batches


@dataclass(frozen=True)
class SpecializationReport:
    before_utilization: dict[str, list[float]]
    after_utilization: dict[str, list[float]]
    before_pairwise_tv_distance: float
    after_pairwise_tv_distance: float
    general_val_loss_before: float
    general_val_loss_after: float
    before_domain_loss: dict[str, float]
    after_domain_loss: dict[str, float]


def _total_variation(a: list[float], b: list[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def measure_specialization(model, params: MoeLayerParams, config: MoeConfig, domain_batches: dict[str, mx.array]) -> dict[str, list[float]]:
    """Real per-domain routing utilization (post-capacity-served
    fraction per expert), computed directly via E2's own
    `route_with_stats` on real held-out domain activations."""
    out = {}
    for name, tokens in domain_batches.items():
        x = collect_real_ffn_input(model, tokens, LAYER)
        mx.eval(x)
        _out, _diag, stats = route_with_stats(x, params, config)
        out[name] = stats.utilization
    return out


def mean_pairwise_tv_distance(per_domain_utilization: dict[str, list[float]]) -> float:
    """Mean pairwise total-variation distance between every pair of
    domains' routing-utilization distributions -- the real
    specialization metric: 0.0 means every domain routes identically
    (no specialization), higher means domains route to measurably
    DIFFERENT experts."""
    names = list(per_domain_utilization.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    distances = [_total_variation(per_domain_utilization[a], per_domain_utilization[b]) for a, b in pairs]
    return sum(distances) / len(distances)


def run_curriculum(model, config: MoeConfig, *, balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50, learning_rate: float = 1e-5, seed: int = 0, warm_start_steps: int = 40, aux_weights: dict[str, float] | None = None, weight_decay: float | None = None) -> tuple[MoeLayerParams, SpecializationReport]:
    """The full real 3-stage curriculum, built on E6's warm-started
    init and E3's proven trainer. Returns the final trained
    `MoeLayerParams` and a `SpecializationReport` comparing
    BEFORE (post-warm-start) vs. AFTER (post-curriculum) on real,
    HELD-OUT domain data (a disjoint slice from what training used).

    `learning_rate=1e-5` (NOT E3's `1e-4`) is the real, measured,
    working default for this specific regime -- found via a direct
    learning-rate sweep, not assumed. E6's warm-started experts start
    at a much LARGER output magnitude (pretrained-weight slices, scaled
    5-7x to compensate for top-1 gate attenuation) than E3's
    small-random init (`scale=0.02`); `lr=1e-4`, stable for the latter,
    causes real, measured divergence here (held-out loss `2.5677 ->
    2.7640` over just 60 real steps in the diagnostic sweep) -- the
    same "learning rate is not scale/init-invariant" lesson this
    project already found in HZ-0D's D6 and HZ-0E's own E3, confirmed
    again in a new regime rather than assumed to carry over."""
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    # offset=1: a real, narrow, pre-existing corpus quirk was found
    # while writing this module's own test suite --
    # `json_and_configuration_train.jsonl` record 0 and
    # `json_and_configuration_validation.jsonl` record 0 are IDENTICAL
    # (checked directly; every other domain's train/validation files
    # were scanned 20 records deep and found clean). Not something
    # this project's own code can fix (it is a pre-existing artifact of
    # how the corpus was originally split), so the held-out load here
    # simply skips record 0 to avoid it, rather than silently trusting
    # an unverified assumption that offset=0 was leakage-free.
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)

    e6_layers = init_e6_layers(model, seed=seed)
    start = e6_layers[LAYER]
    warm = supervised_warm_start(
        model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=LAYER,
        steps=warm_start_steps, learning_rate=1e-3, start_params=start,
    )

    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]
    from reference.hz0e_e3_routing_objectives import lm_forward_with_moe
    warm_dict = params_to_dict(warm)
    val_losses_before = [float(lm_forward_with_moe(warm_dict, model, tb, config, LAYER)[0]) for tb in general_val]
    general_before = sum(val_losses_before) / len(val_losses_before)

    before_util = measure_specialization(model, warm, config, held_out_domains)
    before_tv = mean_pairwise_tv_distance(before_util)
    before_domain_loss = {name: float(lm_forward_with_moe(warm_dict, model, tb, config, LAYER)[0]) for name, tb in held_out_domains.items()}

    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    all_batches = stage1 + stage2 + stage3

    trained, _history = train_moe_layer(
        model, all_batches, config, layer_index=LAYER, aux_weights=aux_weights or {},
        learning_rate=learning_rate, start_params=warm, cache_backbone=True, compile_step=True,
        record_history=False, eval_interval=8,
        weight_decay=weight_decay,
    )

    trained_dict = params_to_dict(trained)
    val_losses_after = [float(lm_forward_with_moe(trained_dict, model, tb, config, LAYER)[0]) for tb in general_val]
    general_after = sum(val_losses_after) / len(val_losses_after)

    after_util = measure_specialization(model, trained, config, held_out_domains)
    after_tv = mean_pairwise_tv_distance(after_util)
    after_domain_loss = {name: float(lm_forward_with_moe(trained_dict, model, tb, config, LAYER)[0]) for name, tb in held_out_domains.items()}

    report = SpecializationReport(
        before_utilization=before_util, after_utilization=after_util,
        before_pairwise_tv_distance=before_tv, after_pairwise_tv_distance=after_tv,
        general_val_loss_before=general_before, general_val_loss_after=general_after,
        before_domain_loss=before_domain_loss, after_domain_loss=after_domain_loss,
    )
    return trained, report


# --- The fair-comparison baseline this module's own results depend on:
# a dense FFN warm-started the SAME way MoE's experts are (a real slice
# of the pretrained FFN, scaled to compensate for the same kind of
# attenuation MoE's top-1 gate causes), trained via the SAME curriculum
# and the SAME real trainer. Without this, "MoE vs. dense" would
# compare a warm-started mechanism against a cold-started one -- not a
# fair test of MoE's own real contribution. ---

def warm_dense_init(model, layer_index: int, d_ff: int, scale: float = 5.0) -> dict[str, mx.array]:
    """A dense FFN's trainable parameters, initialized as a real slice
    of the model's own pretrained FFN weights at `layer_index` (the
    first `d_ff` rows of `gate`/`up`, first `d_ff` columns of `down`),
    with `down` scaled by `scale` -- the same compensation idea E6's
    `init_e6_layers` uses for MoE's experts (a top-1 gate typically
    passes well under `1.0` of an expert's raw output; a dense FFN with
    no gate at all needs no such compensation in principle, but using
    the SAME scale as MoE's own warm-start keeps this a fair,
    apples-to-apples comparison rather than independently re-tuning
    two different scale conventions)."""
    block = model.blocks[layer_index]
    return {
        "gate_w": block.gate.weight[:d_ff], "gate_b": block.gate.bias[:d_ff],
        "up_w": block.up.weight[:d_ff], "up_b": block.up.bias[:d_ff],
        "down_w": block.down.weight[:, :d_ff] * scale, "down_b": block.down.bias,
    }


def make_warm_dense_loss_fn(model, layer_index: int):
    """Returns a `loss_fn(params, tokens) -> scalar` compatible with
    `reference/hz0e_e4_fair_baselines.py::train_generic`/`eval_generic`,
    for a dense FFN using `warm_dense_init`'s parameter dict."""
    block = model.blocks[layer_index]

    def loss_fn(params: dict[str, mx.array], tokens: mx.array) -> mx.array:
        x = model.embedding(tokens)
        for i in range(layer_index):
            x, _ = model.blocks[i](x, None)
        mixed, _ = block.mixer(block.norm1(x), None)
        x = x + mixed
        ffn_input = block.norm2(x)
        mlp = (nn.silu(ffn_input @ params["gate_w"].T + params["gate_b"]) * (ffn_input @ params["up_w"].T + params["up_b"])) @ params["down_w"].T + params["down_b"]
        x = x + mlp
        for i in range(layer_index + 1, len(model.blocks)):
            x, _ = model.blocks[i](x, None)
        logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))

    return loss_fn


def run_warm_dense_baseline(model, *, d_ff: int = 577, layer_index: int = LAYER, balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50, learning_rate: float = 1e-5, seed: int = 0) -> tuple[dict[str, mx.array], float, float]:
    """The SAME 3-stage real curriculum `run_curriculum` trains MoE on,
    applied to a warm-started dense-matched-active FFN instead --
    E8's own fair-baseline re-check of E4's flagged risk, not just
    assumed to carry over from E4's small-random-init comparison.
    Returns `(trained_params, general_val_loss_before, general_val_loss_after)`."""
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]
    loss_fn = make_warm_dense_loss_fn(model, layer_index)

    before_params = warm_dense_init(model, layer_index, d_ff)
    before = eval_generic(model, general_val, before_params, loss_fn)

    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    trained, _losses = train_generic(model, stage1 + stage2 + stage3, lambda: warm_dense_init(model, layer_index, d_ff), loss_fn, learning_rate=learning_rate)
    after = eval_generic(model, general_val, trained, loss_fn)
    return trained, before, after


# --- Full 3-layer joint integration: the actually-scoped E1 contract
# (layers 27, 28, 30 converted SIMULTANEOUSLY), not just one layer in
# isolation. Every earlier E6/E8 quality measurement -- including the
# 150-step and 450-step single-layer results this module and its
# results doc report -- only ever tested layer 27 alone. This checks
# whether the AGGREGATE effect of all 3 real target layers differs
# from a single isolated layer, using E6's own `forward_e6` (which
# already supports simultaneous multi-layer MoE dispatch) rather than
# building new integration machinery. ---

def _pack_layers(layers: dict[int, MoeLayerParams]) -> dict[str, mx.array]:
    flat: dict[str, mx.array] = {}
    for index, params in layers.items():
        for key, value in params_to_dict(params).items():
            flat[f"{index}.{key}"] = value
    return flat


def _unpack_layers(flat: dict[str, mx.array], target_layers: tuple[int, ...] = TARGET_LAYERS) -> dict[int, MoeLayerParams]:
    layers = {}
    for index in target_layers:
        prefix = f"{index}."
        sub = {key[len(prefix):]: value for key, value in flat.items() if key.startswith(prefix)}
        layers[index] = MoeLayerParams(**sub)
    return layers


def run_joint_multilayer_curriculum(model, config: MoeConfig, *, balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50, learning_rate: float = 1e-5, seed: int = 0, warm_start_steps: int = 40, target_layers: tuple[int, ...] = TARGET_LAYERS) -> tuple[dict[int, MoeLayerParams], float, float, float]:
    """Trains ALL of `target_layers` (default: E1's real 3-layer
    contract, 27/28/30) TOGETHER via one shared gradient step per real
    batch -- not 3 separate single-layer runs. Returns
    `(trained_layers, pure_dense_val_loss, warm_start_only_val_loss,
    after_curriculum_val_loss)`, all on the SAME real held-out prose
    set every other E4/E8 quality number in this project uses."""
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]

    pure_dense_losses = [float(cross_entropy_loss(forward_e6(model, tb, enabled=False).logits, tb)) for tb in general_val]
    pure_dense_val = sum(pure_dense_losses) / len(pure_dense_losses)

    e6_layers = init_e6_layers(model, seed=seed, target_layers=target_layers)
    warmed_layers = {
        index: supervised_warm_start(model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=index, steps=warm_start_steps, learning_rate=1e-3, start_params=e6_layers[index])
        for index in target_layers
    }
    warm_losses = [float(cross_entropy_loss(forward_e6(model, tb, moe_layers=warmed_layers, enabled=True, target_layers=target_layers).logits, tb)) for tb in general_val]
    warm_val = sum(warm_losses) / len(warm_losses)

    flat_params = _pack_layers(warmed_layers)

    def loss_fn(flat_p: dict[str, mx.array], tokens: mx.array) -> mx.array:
        layers = _unpack_layers(flat_p, target_layers)
        result = forward_e6(model, tokens, moe_layers=layers, enabled=True, target_layers=target_layers)
        return cross_entropy_loss(result.logits, tokens)

    import mlx.optimizers as optim
    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    optimizer = optim.Adam(learning_rate=learning_rate)

    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    for tokens in stage1 + stage2 + stage3:
        _loss, grads = grad_fn(flat_params, tokens)
        flat_params = optimizer.apply_gradients(grads, flat_params)
        mx.eval(flat_params)

    trained_layers = _unpack_layers(flat_params, target_layers)
    after_losses = [float(cross_entropy_loss(forward_e6(model, tb, moe_layers=trained_layers, enabled=True, target_layers=target_layers).logits, tb)) for tb in general_val]
    after_val = sum(after_losses) / len(after_losses)
    return trained_layers, pure_dense_val, warm_val, after_val


# --- Per-domain (in-distribution) quality comparison. Every earlier
# comparison in this module and in E4 used ONE external, general-prose
# held-out set (`repro_1024_val.jsonl`) as the sole quality metric --
# a real, valid measure of OUT-OF-DISTRIBUTION robustness, but not a
# direct test of whether specialization (the actual purpose of a
# curriculum that trains on 5 real domains) helps on those SAME real
# domains. This section adds that direct test: mean held-out loss
# across all 5 real domains the curriculum trains on, held-out data
# disjoint from training (`DOMAIN_DATA_PATHS`, offset=1 past the known
# JSON duplicate) -- a different, equally real question from general-
# prose robustness, not a replacement for it. ---

def per_domain_mean_loss(losses_by_domain: dict[str, float]) -> float:
    return sum(losses_by_domain.values()) / len(losses_by_domain)


def evaluate_moe_per_domain(model, params: MoeLayerParams, config: MoeConfig, layer_index: int = LAYER) -> dict[str, float]:
    """Real per-domain held-out loss for a single-layer MoE state."""
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    from reference.hz0e_e3_routing_objectives import lm_forward_with_moe
    params_dict = params_to_dict(params)
    return {name: float(lm_forward_with_moe(params_dict, model, tb, config, layer_index)[0]) for name, tb in held_out_domains.items()}


def evaluate_dense_per_domain(model, params: dict[str, mx.array], layer_index: int = LAYER) -> dict[str, float]:
    """Real per-domain held-out loss for a single-layer warm-started
    dense baseline (from `warm_dense_init`/`make_warm_dense_loss_fn`)."""
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    loss_fn = make_warm_dense_loss_fn(model, layer_index)
    return {name: float(loss_fn(params, tb)) for name, tb in held_out_domains.items()}


def evaluate_joint_moe_per_domain(model, trained_layers: dict[int, MoeLayerParams], target_layers: tuple[int, ...] = TARGET_LAYERS) -> dict[str, float]:
    """Real per-domain held-out loss for the full joint multi-layer MoE
    state (`run_joint_multilayer_curriculum`'s own trained output)."""
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    return {
        name: float(cross_entropy_loss(forward_e6(model, tb, moe_layers=trained_layers, enabled=True, target_layers=target_layers).logits, tb))
        for name, tb in held_out_domains.items()
    }


def run_joint_multilayer_dense_baseline(model, *, d_ff: int = 577, target_layers: tuple[int, ...] = TARGET_LAYERS, balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50, learning_rate: float = 1e-5, seed: int = 0) -> dict[str, mx.array]:
    """The fair 3-layer-scope counterpart to `warm_dense_init` -- a
    warm-started, matched-active dense FFN at EVERY one of
    `target_layers` simultaneously, trained via the SAME 3-stage
    curriculum `run_joint_multilayer_curriculum` uses for MoE. Without
    this, "3-layer MoE vs. dense" would only be testable against the
    UNTOUCHED original network (a much cheaper reference, not a fair
    matched-active comparison at 3-layer scope)."""
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)

    def init_layers() -> dict[str, mx.array]:
        flat: dict[str, mx.array] = {}
        for index in target_layers:
            block = model.blocks[index]
            flat[f"{index}.gate_w"] = block.gate.weight[:d_ff]
            flat[f"{index}.gate_b"] = block.gate.bias[:d_ff]
            flat[f"{index}.up_w"] = block.up.weight[:d_ff]
            flat[f"{index}.up_b"] = block.up.bias[:d_ff]
            flat[f"{index}.down_w"] = block.down.weight[:, :d_ff] * 5.0
            flat[f"{index}.down_b"] = block.down.bias
        return flat

    def loss_fn(params: dict[str, mx.array], tokens: mx.array) -> mx.array:
        x = model.embedding(tokens)
        for index, block in enumerate(model.blocks):
            if index not in target_layers:
                x, _ = block(x, None)
            else:
                mixed, _ = block.mixer(block.norm1(x), None)
                residual = x + mixed
                ffn_input = block.norm2(residual)
                mlp = (nn.silu(ffn_input @ params[f"{index}.gate_w"].T + params[f"{index}.gate_b"]) * (ffn_input @ params[f"{index}.up_w"].T + params[f"{index}.up_b"])) @ params[f"{index}.down_w"].T + params[f"{index}.down_b"]
                x = residual + mlp
        logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
        return cross_entropy_loss(logits, tokens)

    params = init_layers()
    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    import mlx.optimizers as optim
    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    optimizer = optim.Adam(learning_rate=learning_rate)
    for tokens in stage1 + stage2 + stage3:
        _loss, grads = grad_fn(params, tokens)
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params)

    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    self_loss_fn = loss_fn
    return {name: float(self_loss_fn(params, tb)) for name, tb in held_out_domains.items()}


# --- Replay/rehearsal: interleaves extra general-prose batches
# throughout the curriculum -- a real, standard continual-learning
# technique for the exact problem E8 found (specialization training
# costs general/out-of-distribution robustness). Tested directly here
# to check whether it CLOSES the gap between MoE and dense on general
# quality, or whether the in-distribution/out-of-distribution tradeoff
# is structural and persists even under a genuine, principled
# mitigation attempt -- not assumed either way. Real result (see
# `docs/restart/hz0e_moe_per_domain_significance_results.md`'s own
# addendum): replay improves BOTH mechanisms' absolute general-quality
# numbers substantially, but does NOT close the RELATIVE gap between
# them -- dense still wins on general/out-of-distribution quality,
# MoE still wins on per-domain/in-distribution quality, with both gaps
# roughly preserved. This confirms the tradeoff is a real, structural
# property of specialization, not a training-recipe artifact fixable
# by more rehearsal. ---

def interleave_replay(curriculum_batches: list[mx.array], replay_batches: list[mx.array]) -> list[mx.array]:
    """Evenly interleaves `replay_batches` throughout
    `curriculum_batches` (roughly one replay batch per
    `len(curriculum_batches) // len(replay_batches)` curriculum
    batches). `replay_batches` should be drawn from a REAL, DISJOINT
    slice of general text (e.g. `TRAIN_DOMAIN_DATA_PATHS["prose"]` at
    an offset beyond what the curriculum's own "prose" domain data
    already uses) -- not the same records the curriculum trains on,
    which would just be double-counting one domain rather than genuine
    rehearsal of general robustness."""
    if not replay_batches:
        return list(curriculum_batches)
    insert_every = max(1, len(curriculum_batches) // len(replay_batches))
    interleaved = []
    replay_index = 0
    for i, batch in enumerate(curriculum_batches):
        interleaved.append(batch)
        if (i + 1) % insert_every == 0 and replay_index < len(replay_batches):
            interleaved.append(replay_batches[replay_index])
            replay_index += 1
    return interleaved


def load_replay_batches(count: int = 20, seq_len: int = 64, domain_train_count: int = 8) -> list[mx.array]:
    """Real general-prose batches DISJOINT from what
    `TRAIN_DOMAIN_DATA_PATHS["prose"]` (offset=0..`domain_train_count`)
    already trains on -- offset starts right after that slice."""
    seqs = load_real_sequences(TRAIN_DOMAIN_DATA_PATHS["prose"], domain_train_count + count)[domain_train_count:]
    return [mx.array([s[:seq_len]]) for s in seqs]
