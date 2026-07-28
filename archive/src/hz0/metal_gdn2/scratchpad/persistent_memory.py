"""
Memory persistence API for scratchpad state.

Modes:
- sequence_local: reset per forward call
- chunk_persistent: carry across chunks in batch
- session_persistent: external state management
"""

import mlx.core as mx
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for memory persistence."""

    mode: str  # "sequence_local" | "chunk_persistent" | "session_persistent"
    num_slots: int
    slot_dim: int
    reset_on_forward: bool = False
    enable_diagnostics: bool = False


class PersistentMemory:
    """
    Scratchpad memory with configurable persistence.

    Replaces implicit internal reset with explicit API.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.state = mx.zeros((config.num_slots, config.slot_dim))
        self.session_id = 0
        self.step_count = 0

    def forward(
        self,
        values: mx.array,  # [B, T, slot_dim] or [B, num_slots, slot_dim]
        external_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Process values with configured persistence.

        Args:
            values: encoded values to store
            external_state: optional [num_slots, slot_dim] state to restore

        Returns:
            output: processed memory content
            state: updated memory state
        """
        # Restore state if provided (session_persistent mode)
        if external_state is not None:
            self.state = external_state

        # Reset if configured (sequence_local mode)
        if self.config.reset_on_forward and self.config.mode == "sequence_local":
            self.state = mx.zeros_like(self.state)

        # Store values
        self.state = self._update_memory(self.state, values)

        # Return output and state
        output = self._read_memory(self.state)
        self.step_count += 1

        return output, self.state

    def _update_memory(self, state: mx.array, values: mx.array) -> mx.array:
        """Update memory with new values."""
        # Simple sum/accumulate approach
        # Production: use routing + selective write
        # Average over batch and sequence dimensions
        mean_val = mx.mean(values, axis=(0, 1), keepdims=False)  # [slot_dim]
        return state + mx.expand_dims(mean_val, axis=0)  # [num_slots, slot_dim]

    def _read_memory(self, state: mx.array) -> mx.array:
        """Read from memory."""
        return state

    def reset_session(self) -> None:
        """Start new session (clear persistent memory)."""
        self.state = mx.zeros_like(self.state)
        self.session_id += 1
        self.step_count = 0

    def get_state(self) -> mx.array:
        """Export current state for external management."""
        return self.state

    def set_state(self, state: mx.array) -> None:
        """Import state from external source."""
        self.state = state


class MemoryInterface:
    """
    High-level API for persistent memory.

    Usage:
        memory = MemoryInterface(mode="session_persistent", ...)
        output, _ = memory(values)
        memory.persist()  # Save session
        memory.reset_session()
        output, _ = memory(new_values)
    """

    def __init__(
        self,
        mode: str = "sequence_local",
        num_slots: int = 64,
        slot_dim: int = 32,
    ):
        self.config = MemoryConfig(
            mode=mode,
            num_slots=num_slots,
            slot_dim=slot_dim,
            reset_on_forward=(mode == "sequence_local"),
        )
        self.memory = PersistentMemory(self.config)
        self.session_cache: Dict[int, mx.array] = {}

    def __call__(
        self,
        values: mx.array,
        memory: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward with optional external memory state.

        Args:
            values: [B, T, slot_dim]
            memory: [num_slots, slot_dim] or None

        Returns:
            output: processed values
            updated_memory: new memory state
        """
        return self.memory.forward(values, external_state=memory)

    def persist_session(self) -> mx.array:
        """Save current session state."""
        state = self.memory.get_state()
        sid = self.memory.session_id
        self.session_cache[sid] = state
        return state

    def restore_session(self, session_id: int) -> None:
        """Restore previous session state."""
        if session_id in self.session_cache:
            self.memory.set_state(self.session_cache[session_id])
            self.memory.session_id = session_id

    def reset_session(self) -> None:
        """Clear session and start fresh."""
        self.memory.reset_session()

    @property
    def state(self) -> mx.array:
        """Current memory state."""
        return self.memory.get_state()

    @state.setter
    def state(self, value: mx.array) -> None:
        """Set memory state directly."""
        self.memory.set_state(value)


# Mode-specific helpers

def sequence_local_memory(
    num_slots: int = 64,
    slot_dim: int = 32,
) -> MemoryInterface:
    """Memory resets per forward call."""
    return MemoryInterface(mode="sequence_local", num_slots=num_slots, slot_dim=slot_dim)


def chunk_persistent_memory(
    num_slots: int = 64,
    slot_dim: int = 32,
) -> MemoryInterface:
    """Memory carries across chunks in batch."""
    return MemoryInterface(mode="chunk_persistent", num_slots=num_slots, slot_dim=slot_dim)


def session_persistent_memory(
    num_slots: int = 64,
    slot_dim: int = 32,
) -> MemoryInterface:
    """Memory managed externally (user provides state)."""
    return MemoryInterface(mode="session_persistent", num_slots=num_slots, slot_dim=slot_dim)
