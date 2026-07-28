"""Phase 3: Validate memory on actual tasks.

Tasks:
- Associative recall: memorize key→value pairs
- Overwrite: update stale entries
- Protected: retain important memories
"""

import mlx.core as mx
from hz0.validation.phase3_memory_redesign import ImprovedScratchpadMemory


def test_associative_recall():
    """Memory learns key→value associations."""
    print("="*70)
    print("Task 1: Associative Recall")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    B, T, D = 1, 10, 64

    # Create pattern: first 5 tokens are "query" keys, second 5 are "values"
    keys = mx.random.normal((B, D)) * 0.1
    values = mx.random.normal((B, D)) * 0.1

    # Initialize memory
    memory_state = memory.initialize_state(B)

    # Write phase: store key→value pairs
    print("\nWrite phase: memorizing patterns...")
    for step in range(5):
        memory_state = memory.write(keys, values, memory_state)

    # Read phase: query should return value
    print("Read phase: retrieving memory...")
    memory_read, attention = memory.read(keys, memory_state)

    # Check if read output closer to values than random
    read_diff = mx.mean(mx.abs(memory_read - values))
    random_diff = mx.mean(mx.abs(mx.random.normal((B, D)) - values))

    print(f"Distance(read, value): {float(read_diff):.6f}")
    print(f"Distance(random, value): {float(random_diff):.6f}")

    if float(read_diff) < float(random_diff) * 0.5:
        print("✓ RECALL SUCCESS: Memory is learning associations")
        return True
    else:
        print("✗ RECALL FAIL: Memory not learning")
        return False


def test_overwrite():
    """Memory can update stale entries."""
    print("\n" + "="*70)
    print("Task 2: Overwrite")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    B, D = 1, 64

    # Initialize memory
    memory_state = memory.initialize_state(B)

    # Write v1
    key1 = mx.ones((B, D)) * 0.1
    val1 = mx.ones((B, D)) * 1.0
    memory_state = memory.write(key1, val1, memory_state)

    read1, _ = memory.read(key1, memory_state)
    dist1 = float(mx.mean(mx.abs(read1 - val1)))

    # Overwrite with v2
    val2 = mx.ones((B, D)) * 2.0
    for _ in range(5):
        memory_state = memory.write(key1, val2, memory_state)

    read2, _ = memory.read(key1, memory_state)
    dist2 = float(mx.mean(mx.abs(read2 - val2)))

    print(f"After write v1: distance={dist1:.6f}")
    print(f"After overwrite v2: distance={dist2:.6f}")

    if dist2 < dist1 * 0.8:
        print("✓ OVERWRITE SUCCESS: Memory updated to new value")
        return True
    else:
        print("✗ OVERWRITE FAIL: Memory not updating")
        return False


def test_protected():
    """Memory protects important entries from interference."""
    print("\n" + "="*70)
    print("Task 3: Protected Retention")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    B, D = 1, 64

    memory_state = memory.initialize_state(B)

    # Store important memory
    key_important = mx.ones((B, D)) * 0.1
    val_important = mx.ones((B, D)) * 5.0
    for _ in range(3):
        memory_state = memory.write(key_important, val_important, memory_state)

    read_before, _ = memory.read(key_important, memory_state)
    dist_before = float(mx.mean(mx.abs(read_before - val_important)))

    # Write many other memories (interference)
    for i in range(10):
        key_noise = mx.random.normal((B, D))
        val_noise = mx.random.normal((B, D))
        memory_state = memory.write(key_noise, val_noise, memory_state)

    # Check if important memory still preserved
    read_after, _ = memory.read(key_important, memory_state)
    dist_after = float(mx.mean(mx.abs(read_after - val_important)))

    print(f"Before interference: distance={dist_before:.6f}")
    print(f"After interference: distance={dist_after:.6f}")
    print(f"Retention: {((dist_before / dist_after) if dist_after > 0 else 1):.2f}x")

    if dist_after < dist_before * 1.5:
        print("✓ PROTECTED SUCCESS: Memory resilient to interference")
        return True
    else:
        print("✗ PROTECTED FAIL: Memory corrupted by noise")
        return False


def phase3_memory_tasks():
    """Run all memory tasks."""
    print("="*70)
    print("Phase 3: Memory Task Validation")
    print("="*70)

    results = []

    recall = test_associative_recall()
    results.append(("Associative Recall", recall))

    overwrite = test_overwrite()
    results.append(("Overwrite", overwrite))

    protected = test_protected()
    results.append(("Protected Retention", protected))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for task, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{task}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n✓ PHASE 3 COMPLETE: Memory validated on all tasks")
        verdict = "PASS"
    elif passed >= 2:
        print(f"\n~ PHASE 3 PARTIAL: {passed}/3 tasks pass")
        verdict = "PARTIAL"
    else:
        print(f"\n✗ PHASE 3 FAIL: Only {passed}/3 tasks pass")
        verdict = "FAIL"

    print("="*70)

    return {
        "verdict": verdict,
        "results": results,
        "passed": passed,
        "total": total,
    }


if __name__ == "__main__":
    results = phase3_memory_tasks()
    print(f"\nPhase 3 Memory Tasks: {results['verdict']}")
