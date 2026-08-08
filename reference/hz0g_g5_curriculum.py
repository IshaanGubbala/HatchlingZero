"""HZ-0G G5: the real Dense vs. MoE vs. domain-adapter curriculum, run
on the FULL A+B+C+D+(E) integration, not A+E alone.

Every prior HZ-0E MoE curriculum (`reference/hz0e_e8_curriculum.py`,
which this module adapts rather than reimplements) trained and
evaluated through `forward_e6` -- backbone (A) + MoE (E) only, B/C/D
never in the loop. G5's own plan text is explicit: compare HZ-Dense vs.
HZ-MoE vs. Dense+domain-adapter "on the corrected, INTEGRATED
checkpoint" -- meaning B (session memory), C (surprise-triggered
attention), and D (fast weights) must be live in the same forward pass
E trains and evaluates against, via
`reference/hz0g_g5_full_integration.py::full_integration_forward`, not
bypassed.

B/C/D are held FIXED for this comparison (matching G4's "do not
retrain permanent weights to accommodate D" and this repo's own
integration-order discipline -- G5 isolates E's contribution, it does
not re-optimize B/C/D): a fresh, untrained `LatentWriteControllerParams`
(B), the standard 15%-rate `fixed_matched_trigger` (C), and INACTIVE
(zero) fast weights (D) -- the same fixed setup G2-G4 used to test each
mechanism against the frozen backbone. Only E's own parameters (MoE
experts, or the domain-adapter baseline's weights) are trained here.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim

from reference.hz0b_b8_latent_write import init_latent_write_controller
from reference.hz0d_d6_integration import d6_fast_weight_config
from reference.hz0d_fast_weights import init_fast_weights
from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e3_routing_objectives import supervised_warm_start
from reference.hz0e_e6_integration import TARGET_LAYERS, init_e6_layers
from reference.hz0e_e8_curriculum import (
    DOMAIN_TO_EXPERT, TRAIN_DOMAIN_DATA_PATHS, _pack_layers, _unpack_layers,
    balanced_batches, imbalanced_batches, load_domain_batches, mixed_domain_batches,
)
from reference.hz0e_moe_contract import MoeConfig
from reference.hz0g_g5_full_integration import full_integration_forward
from scripts.hz0c_c3_trigger_simulator import load_real_sequences
from scripts.hz0c_c6_conditional_attention_eval import fixed_matched_trigger


def _fixed_bcd(model, batch: int, seq_len: int, *, seed: int = 0):
    """The frozen B/C/D setup every G5 arm shares -- built once, reused
    identically across HZ-Dense/HZ-MoE/adapter so E is the only real
    variable."""
    fast_config = d6_fast_weight_config()
    fast_state = init_fast_weights(fast_config)
    latent_params = init_latent_write_controller(d_model=model.dim, key_dim=64, value_dim=64, seed=seed)
    trigger = fixed_matched_trigger(batch, seq_len, 0.15)
    return trigger, latent_params, fast_state, fast_config


def _ce(logits: mx.array, token_ids: mx.array) -> mx.array:
    import mlx.nn as nn
    return mx.mean(nn.losses.cross_entropy(logits[:, :-1], token_ids[:, 1:]))


def integrated_loss(model, tokens: mx.array, *, moe_layers=None, moe_enabled: bool, moe_target_layers=TARGET_LAYERS, bcd_seed: int = 0) -> mx.array:
    batch, seq_len = tokens.shape
    trigger, latent_params, fast_state, fast_config = _fixed_bcd(model, batch, seq_len, seed=bcd_seed)
    result = full_integration_forward(
        model, tokens, trigger, latent_params, fast_state, fast_config,
        moe_layers=moe_layers, moe_enabled=moe_enabled, moe_target_layers=moe_target_layers,
    )
    return _ce(result.logits, tokens)


def run_integrated_moe_curriculum(
    model, config: MoeConfig, *, balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50,
    learning_rate: float = 1e-5, seed: int = 0, warm_start_steps: int = 40, target_layers: tuple = TARGET_LAYERS,
) -> tuple[dict, float, float, float]:
    """HZ-MoE's real training curriculum on the FULL integration --
    adapts hz0e_e8_curriculum.py::run_joint_multilayer_curriculum,
    replacing every forward_e6 call with integrated_loss/
    full_integration_forward so B/C/D are live throughout training and
    evaluation, not just at the end. Returns (trained_layers,
    pure_dense_val, warm_val, after_val), same real held-out general-
    prose set every other E4/E8/G5 quality number in this project uses."""
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]

    pure_dense_losses = [float(integrated_loss(model, tb, moe_enabled=False, bcd_seed=seed)) for tb in general_val]
    pure_dense_val = sum(pure_dense_losses) / len(pure_dense_losses)

    e6_layers = init_e6_layers(model, seed=seed, target_layers=target_layers)
    warmed_layers = {
        index: supervised_warm_start(model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=index, steps=warm_start_steps, learning_rate=1e-3, start_params=e6_layers[index])
        for index in target_layers
    }
    warm_losses = [float(integrated_loss(model, tb, moe_layers=warmed_layers, moe_enabled=True, moe_target_layers=target_layers, bcd_seed=seed)) for tb in general_val]
    warm_val = sum(warm_losses) / len(warm_losses)

    flat_params = _pack_layers(warmed_layers)

    def loss_fn(flat_p: dict, tokens: mx.array) -> mx.array:
        layers = _unpack_layers(flat_p, target_layers)
        return integrated_loss(model, tokens, moe_layers=layers, moe_enabled=True, moe_target_layers=target_layers, bcd_seed=seed)

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
    after_losses = [float(integrated_loss(model, tb, moe_layers=trained_layers, moe_enabled=True, moe_target_layers=target_layers, bcd_seed=seed)) for tb in general_val]
    after_val = sum(after_losses) / len(after_losses)
    return trained_layers, pure_dense_val, warm_val, after_val


def evaluate_integrated_moe_per_domain(model, trained_layers: dict, target_layers: tuple = TARGET_LAYERS, *, seed: int = 0) -> dict[str, float]:
    """Real per-domain held-out loss for the trained HZ-MoE arm, through
    the full integration -- the direct specialization test, same
    held-out domain split E8 established (offset=1, disjoint from
    training)."""
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    return {name: float(integrated_loss(model, tb, moe_layers=trained_layers, moe_enabled=True, moe_target_layers=target_layers, bcd_seed=seed)) for name, tb in held_out_domains.items()}


def run_integrated_dense_baseline(
    model, *, d_ff: int = 577, target_layers: tuple = TARGET_LAYERS,
    balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50,
    learning_rate: float = 1e-5, seed: int = 0,
) -> dict[str, mx.array]:
    """The fair 'Dense + domain adapter' arm -- E4's own dangerous
    baseline (a small trained adapter can beat MoE outright), now
    through the full A+B+C+D integration rather than A alone. A
    warm-started dense FFN at every target layer, trained with the
    SAME 3-stage curriculum and the SAME fixed B/C/D as the MoE arm, so
    the only real difference between this arm and HZ-MoE is dense-vs-
    routed at the FFN layers -- everything else matched."""
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

    flat_params = init_layers()
    grad_fn = mx.value_and_grad(lambda params, tokens: _dense_loss(model, params, tokens, target_layers, seed), argnums=0)
    optimizer = optim.Adam(learning_rate=learning_rate)

    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    for tokens in stage1 + stage2 + stage3:
        _loss, grads = grad_fn(flat_params, tokens)
        flat_params = optimizer.apply_gradients(grads, flat_params)
        mx.eval(flat_params)
    return flat_params


def _dense_loss(model, flat_params: dict, tokens: mx.array, target_layers: tuple, seed: int) -> mx.array:
    """The dense-adapter arm's own forward -- runs the SAME fixed B/C/D
    as the MoE arm, but with a trained dense FFN patched into
    `target_layers` instead of MoE, via a lightweight monkeypatch of
    the block's gate/up/down at call time. Deliberately NOT
    full_integration_hidden's own MoE branch (a dense adapter is a
    different real mechanism, not an MoE variant)."""
    import mlx.nn as nn
    from reference.hz0d_d6_integration import ATTENTION_INDICES, fast_masked_anchor_attention, logits_from_hidden
    from reference.hz0b_b8_latent_write import sequential_latent_write_and_read

    batch, seq_len = tokens.shape
    trigger, latent_params, fast_state, fast_config = _fixed_bcd(model, batch, seq_len, seed=seed)
    x = model.embedding(tokens)
    for index, block in enumerate(model.blocks):
        if index in ATTENTION_INDICES:
            fast_layer = ATTENTION_INDICES.index(index)
            normed = block.norm1(x)
            anchor = fast_masked_anchor_attention(
                normed, trigger, qkv_w=block.mixer.qkv.weight, qkv_b=block.mixer.qkv.bias,
                out_w=block.mixer.out.weight, out_b=block.mixer.out.bias, heads=model.heads,
                fast_a=fast_state.a_fast[fast_layer], fast_b=fast_state.b_fast[fast_layer],
            )
            x = x + anchor
            normed2 = block.norm2(x)
            x = x + block.down(nn.silu(block.gate(normed2)) * block.up(normed2))
        elif index in target_layers:
            mixed, _ = block.mixer(block.norm1(x), None)
            residual = x + mixed
            ffn_input = block.norm2(residual)
            p = flat_params
            gated = nn.silu(ffn_input @ p[f"{index}.gate_w"].T + p[f"{index}.gate_b"]) * (ffn_input @ p[f"{index}.up_w"].T + p[f"{index}.up_b"])
            ffn_out = gated @ p[f"{index}.down_w"].T + p[f"{index}.down_b"]
            x = residual + ffn_out
        else:
            x, _ = block(x, None)
    hidden, _memory_state, _write_gates = sequential_latent_write_and_read(latent_params, x)
    logits = logits_from_hidden(model, hidden)
    return _ce(logits, tokens)


def evaluate_integrated_dense_per_domain(model, flat_params: dict, target_layers: tuple = TARGET_LAYERS, *, seed: int = 0) -> dict[str, float]:
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    return {name: float(_dense_loss(model, flat_params, tb, target_layers, seed)) for name, tb in held_out_domains.items()}
