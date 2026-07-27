"""
Debug full GDN2LanguageModel to find where NaN originates.
"""

import mlx.core as mx
import numpy as np
import sys

sys.path.insert(0, '/Users/ishaangubbala/Documents/Training')

from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel


def test_gdn2_model():
    """Test model initialization and forward pass."""
    print("=" * 80)
    print("Testing GDN2LanguageModel")
    print("=" * 80)
    print()

    print("1. Creating model...")
    try:
        model = GDN2LanguageModel(
            vocab_size=32768,  # Real size
            model_dim=768,     # Real size
            num_layers=4,      # Reduced for debugging
            num_heads=12,      # Real heads
        )
        print("   ✓ Model created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    print("\n2. Checking model parameters...")
    params = model.parameters()
    if isinstance(params, dict):
        for name, param in params.items():
            if hasattr(param, 'shape'):
                has_nan = bool(mx.any(mx.isnan(param)))
                has_inf = bool(mx.any(mx.isinf(param)))
                status = "OK"
                if has_nan:
                    status = "NaN"
                elif has_inf:
                    status = "INF"
                min_val = float(mx.min(param))
                max_val = float(mx.max(param))
                print(f"   {status:6s} {name:40s} shape={str(param.shape):20s} "
                      f"min={min_val:10.4f} max={max_val:10.4f}")
    else:
        print("   (parameters() returns non-dict)")

    print("\n3. Testing forward pass...")
    batch_ids = mx.array(np.random.randint(0, 256, (1, 8)), dtype=mx.int32)
    print(f"   Input shape: {batch_ids.shape}")

    try:
        logits, state = model(batch_ids)
        print(f"   ✓ Forward pass completed")
        print(f"   Logits shape: {logits.shape}")

        has_nan = bool(mx.any(mx.isnan(logits)))
        has_inf = bool(mx.any(mx.isinf(logits)))
        min_val = float(mx.min(logits))
        max_val = float(mx.max(logits))

        print(f"   Logits: min={min_val:.4f} max={max_val:.4f}")
        if has_nan:
            print("   ✗ Logits contain NaN!")
        elif has_inf:
            print("   ✗ Logits contain inf!")
        else:
            print("   ✓ Logits valid")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gdn2_model()
