# Live model representation view

This view is driven by real HZ-0A inference, not a mock visualization. The probe loads the newest model checkpoint from a running native-Metal run, executes the supplied token sequence, and records per-layer recurrent-state magnitude plus actual top logits.

Start the probe on the machine running the checkpoint:

```bash
archive/.venv/bin/python scripts/hz0a_live_representation.py \
  --run-dir outputs/<run-name> \
  --tokens 101,42,17,9,88,3 \
  --interval 5
```

In another terminal, start the existing dashboard:

```bash
archive/.venv/bin/python scripts/hz0a_dashboard_server.py --port 8014
```

Open `http://127.0.0.1:8014`. The recurrent heatmap and top-logit panel update whenever a newer checkpoint is available. The heatmap is the actual checkpoint's recurrent state; the top-logit list is the actual model output for the probe tokens.

The probe intentionally runs separately from the training process to avoid adding large host/device synchronization overhead to the training loop.
