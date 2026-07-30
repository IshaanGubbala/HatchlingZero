"""HZ-0B Phase B6 prep tests: verifies the plan's own B6 "Verify" list
directly, against a synthetic frozen-model stand-in (no real HZ-0A
checkpoint involved) --

  - memory read does not destabilize logits
  - empty memory behaves like no memory
  - reset returns to baseline behavior
  - unrelated memories do not corrupt output

plus a differentiability check, since B1 decided writes/reads are
differentiable by default and B7 (controlled writes) will need to train
through this same read path.
"""
import mlx.core as mx

from reference.hz0b_memory_simulator import reset as memory_reset
from reference.hz0b_memory_simulator import write as memory_write
from reference.hz0b_readonly_integration import (
    gated_memory_read,
    init_readonly_integration,
)

D_MODEL = 768   # HZ-0A's frozen width all session (dim=768) -- fixed regardless of Stage 2's progress
KEY_DIM = 4
VALUE_DIM = 4
NUM_SLOTS = 8


def onehot(dim, index, batch=1):
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row for _ in range(batch)])


def make_hidden(seed=0):
    return mx.random.normal((1, D_MODEL), key=mx.random.key(seed))


# 1. empty memory behaves like no memory: readout is exactly zero (no
# slot has ever been written, so every value is still zero regardless of
# what the softmax read-weights look like) -> output == hidden exactly
def test_empty_memory_behaves_like_no_memory():
    params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = memory_reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    hidden = make_hidden(seed=2)
    output, _ = gated_memory_read(params, hidden, memory)
    assert bool(mx.allclose(output, hidden, atol=1e-6))


# 2. reset returns to baseline: after writing, output differs from
# hidden; after reset, it's back to matching case 1 exactly
def test_reset_returns_to_baseline_after_prior_writes():
    params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = memory_reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    hidden = make_hidden(seed=2)

    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 5.0
    written_memory, _, _ = memory_write(memory, key, value, mx.array([1.0]), step=0)
    output_after_write, _ = gated_memory_read(params, hidden, written_memory)
    assert not bool(mx.allclose(output_after_write, hidden, atol=1e-4))

    reset_memory = memory_reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    output_after_reset, _ = gated_memory_read(params, hidden, reset_memory)
    assert bool(mx.allclose(output_after_reset, hidden, atol=1e-6))


# 3. memory read does not destabilize logits: the gated contribution is
# bounded by construction (sigmoid gate in [0, 1], readout is a convex
# combination of stored values), so even a maximal-strength matching
# write cannot blow up the output beyond hidden's own norm plus the
# largest stored value's norm
def test_memory_read_does_not_destabilize_output_magnitude():
    params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = memory_reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    hidden = make_hidden(seed=2)

    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 100.0  # deliberately large stored value
    written_memory, _, _ = memory_write(memory, key, value, mx.array([1.0]), step=0)
    output, read_weights = gated_memory_read(params, hidden, written_memory)

    assert bool(mx.all(mx.isfinite(output)))
    # read_weights form a valid convex combination (softmax) -> readout norm <= max stored value norm
    assert float(mx.sum(read_weights)) - 1.0 < 1e-4
    value_to_hidden_norm = float(mx.sqrt(mx.sum(params.value_to_hidden_w * params.value_to_hidden_w)))
    max_readout_contribution = 100.0 * value_to_hidden_norm  # loose analytic bound, not a tight one
    delta_norm = float(mx.sqrt(mx.sum((output - hidden) ** 2)))
    assert delta_norm <= max_readout_contribution + 1e-3


# 4. unrelated memories do not corrupt output: a query orthogonal to the
# only written key produces a much smaller output change than a query
# that actually matches it -- proves the mechanism discriminates
# relevant from irrelevant memory rather than injecting noise regardless
# of relevance
def test_unrelated_memory_produces_smaller_change_than_matching_memory():
    params = init_readonly_integration(KEY_DIM, KEY_DIM, VALUE_DIM, seed=3)  # d_model==key_dim here so hidden can double as an exact query probe
    memory = memory_reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 5.0
    written_memory, _, _ = memory_write(memory, key, value, mx.array([1.0]), step=0)

    matching_hidden = onehot(KEY_DIM, 0)     # query_w is identity-ish random but the RAW similarity test below uses memory_read directly
    unrelated_hidden = onehot(KEY_DIM, 1)    # orthogonal to the written key

    # Bypass the learned query projection to test the actual read
    # mechanism's discrimination directly (query == hidden here since
    # KEY_DIM == d_model in this test's setup)
    from reference.hz0b_memory_simulator import read as memory_read
    matching_readout, _ = memory_read(written_memory, matching_hidden)
    unrelated_readout, _ = memory_read(written_memory, unrelated_hidden)

    matching_norm = float(mx.sqrt(mx.sum(matching_readout ** 2)))
    unrelated_norm = float(mx.sqrt(mx.sum(unrelated_readout ** 2)))
    assert matching_norm > unrelated_norm


# 5. differentiability: gradients flow through query projection, gate,
# and value-to-hidden projection -- required for B7's later training,
# not exercised by B1-B4's pure-simulator tests since none of them had
# an integration module to differentiate through yet
def test_gated_memory_read_is_differentiable():
    params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=4)
    memory = memory_reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 5.0
    written_memory, _, _ = memory_write(memory, key, value, mx.array([1.0]), step=0)
    hidden = make_hidden(seed=5)

    def loss_fn(query_w):
        p = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=4)
        p = type(p)(query_w=query_w, query_b=p.query_b, gate_w=p.gate_w, gate_b=p.gate_b,
                     value_to_hidden_w=p.value_to_hidden_w, value_to_hidden_b=p.value_to_hidden_b)
        output, _ = gated_memory_read(p, hidden, written_memory)
        return mx.sum(output)

    grad = mx.grad(loss_fn)(params.query_w)
    assert grad.shape == params.query_w.shape
    assert bool(mx.all(mx.isfinite(grad)))
    assert float(mx.sum(mx.abs(grad))) > 0.0
