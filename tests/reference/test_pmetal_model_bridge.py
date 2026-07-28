import json
import numpy as np

from restart.hz0a_pmetal.python.model_bridge import PmetalModelBridge


def test_full_topology_bridge_updates_all_parameter_arrays_and_carries_state(tmp_path):
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"vocab_size": 32, "d_model": 16, "num_layers": 3, "num_heads": 2, "head_dim_qk": 8, "head_dim_v": 8, "d_ff": 32, "attention_layer_indices": [1]}))
    bridge = PmetalModelBridge(config, seed=5)
    tokens = (np.arange(8).reshape(2, 4) + 1) % 32
    targets = np.roll(tokens, -1, axis=1)
    before = bridge.fingerprint()
    loss, metric, states = bridge.train_microbatch(tokens, targets, tokens_seen=tokens.size)
    assert np.isfinite(loss)
    assert metric is not None
    assert bridge.fingerprint() != before
    assert states[0] is not None and states[1] is None and states[2] is not None
    continued_loss, _, continued_states = bridge.train_microbatch(tokens[:, :1], targets[:, :1], tokens_seen=2, states=states)
    assert np.isfinite(continued_loss)
    assert continued_states[0].shape == states[0].shape


def test_full_topology_bridge_100_step_replay_resumes_exactly(tmp_path):
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"vocab_size": 32, "d_model": 16, "num_layers": 3, "num_heads": 2, "head_dim_qk": 8, "head_dim_v": 8, "d_ff": 32, "attention_layer_indices": [1]}))
    batches = [((np.arange(8).reshape(2, 4) + step) % 32).astype(np.int64) for step in range(100)]
    targets = [np.roll(batch, -1, axis=1) for batch in batches]
    full = PmetalModelBridge(config, seed=17)
    resumed = PmetalModelBridge(config, seed=17)
    for step, (batch, target) in enumerate(zip(batches, targets)):
        full.train_microbatch(batch, target, tokens_seen=batch.size)
        resumed.train_microbatch(batch, target, tokens_seen=batch.size)
        if step == 49:
            resumed.checkpoint(tmp_path / "bridge.json")
            resumed = PmetalModelBridge.restore(config, tmp_path / "bridge.json")
    assert full.fingerprint() == resumed.fingerprint()
    assert full.optimizer.state.optimizer_step == resumed.optimizer.state.optimizer_step == 100
    assert full.optimizer.state.tokens_seen == resumed.optimizer.state.tokens_seen == 800
