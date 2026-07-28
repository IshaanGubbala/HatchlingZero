"""
GDN-2 debug: Trace NaN source step-by-step.

Log every intermediate value to find where NaN originates.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, Optional


class GDN2DebugModule(nn.Module):
    """GDN-2 with detailed logging for NaN debugging."""

    def __init__(self, dim: int = 64, hidden_dim: Optional[int] = None, num_heads: int = 2):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads

        # Projections
        self.to_qkv = nn.Linear(dim, 3 * self.hidden_dim)
        self.to_decay_erase_write = nn.Linear(dim, self.hidden_dim * 3)
        self.to_out = nn.Linear(self.hidden_dim, dim)

    def _check_tensor(self, name: str, x: mx.array, verbose: bool = True):
        """Check for NaN/inf and log stats."""
        has_nan = bool(mx.any(mx.isnan(x)))
        has_inf = bool(mx.any(mx.isinf(x)))
        min_val = float(mx.min(x))
        max_val = float(mx.max(x))
        mean_val = float(mx.mean(x))

        status = "OK"
        if has_nan:
            status = "NaN"
        elif has_inf:
            status = "INF"
        elif abs(mean_val) > 1e6:
            status = "LARGE"

        if verbose or status != "OK":
            print(f"  {name:30s}: shape={str(x.shape):20s} {status:6s} "
                  f"min={min_val:10.4f} max={max_val:10.4f} mean={mean_val:10.4f}")

        return has_nan or has_inf

    def __call__(
        self,
        x: mx.array,  # [B, T, D]
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward with debugging."""
        print("=" * 80)
        print("GDN2 DEBUG FORWARD")
        print("=" * 80)

        B, T, D = x.shape
        print(f"\nInput: shape={x.shape}")
        self._check_tensor("input_x", x)

        # QKV projection
        print("\n[QKV Projection]")
        qkv = self.to_qkv(x)
        self._check_tensor("qkv (before split)", qkv)

        qkv = mx.reshape(qkv, (B, T, 3, self.hidden_dim))
        q, k, v = mx.split(qkv, 3, axis=2)
        q = mx.squeeze(q, axis=2)
        k = mx.squeeze(k, axis=2)
        v = mx.squeeze(v, axis=2)

        self._check_tensor("q (before reshape)", q)
        self._check_tensor("k (before reshape)", k)
        self._check_tensor("v (before reshape)", v)

        # Reshape to heads
        q = mx.reshape(q, (B, T, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, T, self.num_heads, self.head_dim))
        v = mx.reshape(v, (B, T, self.num_heads, self.head_dim))

        self._check_tensor("q (after reshape)", q)
        self._check_tensor("k (after reshape)", k)
        self._check_tensor("v (after reshape)", v)

        # Decay/Erase/Write projection
        print("\n[Decay/Erase/Write Projection]")
        decay_erase_write = self.to_decay_erase_write(x)
        self._check_tensor("decay_erase_write (raw)", decay_erase_write)

        decay_erase_write = mx.reshape(decay_erase_write, (B, T, 3, self.hidden_dim))
        d, e, w = mx.split(decay_erase_write, 3, axis=2)
        d = mx.squeeze(d, axis=2)
        e = mx.squeeze(e, axis=2)
        w = mx.squeeze(w, axis=2)

        self._check_tensor("d (before sigmoid)", d)
        self._check_tensor("e (before sigmoid)", e)
        self._check_tensor("w (before sigmoid)", w)

        # Apply sigmoid
        d_sig = mx.sigmoid(d)
        e_sig = mx.sigmoid(e)
        w_sig = mx.sigmoid(w)

        self._check_tensor("d (after sigmoid)", d_sig)
        self._check_tensor("e (after sigmoid)", e_sig)
        self._check_tensor("w (after sigmoid)", w_sig)

        # Reshape gates
        d_reshaped = mx.reshape(d_sig, (B, T, self.num_heads, self.head_dim))
        e_reshaped = mx.reshape(e_sig, (B, T, self.num_heads, self.head_dim))
        w_reshaped = mx.reshape(w_sig, (B, T, self.num_heads, self.head_dim))

        self._check_tensor("d (after reshape)", d_reshaped)
        self._check_tensor("e (after reshape)", e_reshaped)
        self._check_tensor("w (after reshape)", w_reshaped)

        # Initialize state
        print("\n[State Initialization]")
        if state is None:
            state = mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=x.dtype)
        self._check_tensor("state (initial)", state)

        # Process sequence
        print("\n[Sequence Processing]")
        outputs = []

        for t in range(min(T, 1)):  # Just first timestep for debugging
            print(f"\n  Timestep {t}:")

            q_t = q[:, t]  # [B, H, Dk]
            k_t = k[:, t]
            v_t = v[:, t]
            d_t = d_reshaped[:, t]
            e_t = e_reshaped[:, t]
            w_t = w_reshaped[:, t]

            self._check_tensor(f"  q_t", q_t, verbose=False)
            self._check_tensor(f"  k_t", k_t, verbose=False)
            self._check_tensor(f"  v_t", v_t, verbose=False)
            self._check_tensor(f"  d_t", d_t, verbose=False)
            self._check_tensor(f"  e_t", e_t, verbose=False)
            self._check_tensor(f"  w_t", w_t, verbose=False)

            # Decay step
            state_decayed = state * d_t[:, :, None, :]
            self._check_tensor(f"    state (after decay)", state_decayed, verbose=False)

            # Stop after first step if NaN
            if bool(mx.any(mx.isnan(state_decayed))):
                print("\n⚠ NaN detected at decay step!")
                break

            state = state_decayed

        # For now, just return dummy output
        print("\n[Output Projection]")
        dummy_output = mx.zeros((B, T, self.hidden_dim))
        output = self.to_out(dummy_output)
        self._check_tensor("output", output)

        print("\n" + "=" * 80)
        return output, state


def test_gdn2_debug():
    """Test GDN-2 debug module."""
    print("Testing GDN-2 Debug Module\n")

    model = GDN2DebugModule(dim=64, num_heads=2)
    x = mx.random.normal((1, 4, 64))

    try:
        output, state = model(x)
        print("\n✓ Forward pass completed")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    test_gdn2_debug()
