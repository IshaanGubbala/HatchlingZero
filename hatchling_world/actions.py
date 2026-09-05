"""Human-readable action decoding, shared by the rollout demo and the
BC training script (plans/Hatchling world.md section 23's suggested
`hatchling_world/actions.py`)."""
from __future__ import annotations

from hatchling_world.state import WorldConfig


def decode_action(a: int, config: WorldConfig) -> dict:
    if a < config.n_rooms:
        return {"type": "move", "text": f"MOVE to room {a}", "target_room": a, "color": None}
    if a < config.n_rooms + config.n_colors * config.n_rooms:
        color, target = config.decode_use_key(a)
        return {"type": "use_key", "text": f"USE key {chr(65 + color)} on door -> room {target}",
                "target_room": target, "color": color}
    if a == config.action_pickup:
        return {"type": "pickup", "text": "PICK UP keys in this room", "target_room": None, "color": None}
    return {"type": "inspect", "text": "INSPECT surroundings", "target_room": None, "color": None}
