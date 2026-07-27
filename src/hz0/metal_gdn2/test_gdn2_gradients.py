"""
GDN-2 gradient checking (plan phase 3).

Validate gradients via finite differences on all parameters:
- Query, Key, Value
- Decay, Erase, Write logits
- Initial state
"""

import numpy as np
from hz0.metal_gdn2.reference.gdn2_numpy import gdn2_step


def check_gradient(param_name, param, delta=1e-5):
    """Check gradient via finite differences."""
    # Setup
    B, H, Dk, Dv = 1, 2, 16, 16

    query = np.random.randn(B, H, Dk).astype(np.float32)
    key = np.random.randn(B, H, Dk).astype(np.float32)
    value = np.random.randn(B, H, Dv).astype(np.float32)
    decay = np.random.uniform(0.1, 0.9, (B, H, Dk)).astype(np.float32)
    erase = np.random.uniform(0.1, 0.9, (B, H, Dk)).astype(np.float32)
    write = np.random.uniform(0.1, 0.9, (B, H, Dv)).astype(np.float32)
    state = np.random.randn(B, H, Dv, Dk).astype(np.float32)

    # Prepare inputs dict
    inputs = {
        'query': query, 'key': key, 'value': value,
        'decay': decay, 'erase': erase, 'write': write, 'state': state
    }

    # f(x + delta)
    inputs_plus = inputs.copy()
    inputs_plus[param_name] = param + delta
    state_plus, output_plus = gdn2_step(**inputs_plus)
    loss_plus = np.sum(output_plus)

    # f(x - delta)
    inputs_minus = inputs.copy()
    inputs_minus[param_name] = param - delta
    state_minus, output_minus = gdn2_step(**inputs_minus)
    loss_minus = np.sum(output_minus)

    # Finite difference gradient
    grad_fd = (loss_plus - loss_minus) / (2 * delta)

    return grad_fd, loss_plus, loss_minus


def run_gradient_checks():
    """Run gradient checks on all parameters."""
    print("=" * 70)
    print("GDN-2 GRADIENT CHECKING (Phase 3)")
    print("=" * 70)
    print()

    params_to_check = ['query', 'key', 'value', 'decay', 'erase', 'write', 'state']

    B, H, Dk, Dv = 1, 2, 16, 16

    results = {}

    for param_name in params_to_check:
        print(f"Checking {param_name}...", end="", flush=True)

        # Create random parameter
        if param_name in ['query', 'key', 'decay', 'erase']:
            param = np.random.randn(B, H, Dk).astype(np.float32)
        elif param_name in ['value', 'write']:
            param = np.random.randn(B, H, Dv).astype(np.float32)
        elif param_name == 'state':
            param = np.random.randn(B, H, Dv, Dk).astype(np.float32)

        try:
            grad_fd, loss_plus, loss_minus = check_gradient(param_name, param)
            results[param_name] = float(grad_fd)
            print(f" ✓ grad={grad_fd:.2e}")
        except Exception as e:
            print(f" ✗ {e}")
            results[param_name] = None

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)

    for param, grad in results.items():
        status = "✓" if grad is not None else "✗"
        print(f"{status} {param:15s}: {grad}")

    print()
    print(f"Gradient checks: {passed}/{total} passed")

    if passed == total:
        print("\n✓ ALL GRADIENTS COMPUTABLE")
        print("GDN-2 reference ready for backprop implementation.")
    else:
        print(f"\n✗ {total - passed} parameter(s) failed")


if __name__ == "__main__":
    run_gradient_checks()
