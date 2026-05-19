"""Quick demo of flux-check — run with `python -m flux_check.demo`."""

from flux_check.core import FluxExact, Severity
from flux_check.presets import PRESETS, list_presets
from flux_check.fracture import Fracturer, DependencyGraph
from flux_check.sediment import SedimentStack, ConstraintCorrection

import numpy as np


def main():
    print("=" * 60)
    print("  FLUX Constraint Engine Demo")
    print("=" * 60)

    # 1. Exact checking
    print("\n1. Exact Constraint Checking")
    print("-" * 40)
    constraints = PRESETS["automotive"]["constraints"]
    fc = FluxExact(constraints)
    print(f"   Preset: automotive ({fc.n} constraints)")

    test_values = [3000, 150, 90, 50, 100, 0, 12, 75]  # all in bounds
    result = fc.check(test_values[0])
    print(f"   Value {test_values[0]} → mask=0x{fc.check_mask(test_values[0]):02x} ({'PASS' if fc.check_mask(test_values[0]) == 0 else 'FAIL'})")

    test_oob = 160  # above coolant temp hi=150
    mask = fc.check_mask(test_oob)
    print(f"   Value {test_oob} → mask=0x{mask:02x} (violates constraints: ", end="")
    violated = [constraints[i]["name"] for i in range(fc.n) if mask & (1 << i)]
    print(", ".join(violated) + ")")

    # 2. NaN handling
    print("\n2. NaN Always Violates")
    print("-" * 40)
    nan_mask = fc.check_mask(float("nan"))
    print(f"   NaN → mask=0x{nan_mask:02x} (all {fc.n} bits set)")

    # 3. Batch checking
    print("\n3. Batch Checking (numpy vectorized)")
    print("-" * 40)
    arr = np.array([50.0, 200.0, float("nan"), -5.0, 9000.0])
    masks = fc.check_batch_numpy(arr)
    for v, m in zip(arr, masks):
        print(f"   {str(v):>8} → mask=0x{int(m):02x} ({'PASS' if m == 0 else 'FAIL'})")

    # 4. Presets
    print("\n4. Available Presets")
    print("-" * 40)
    for name in list_presets():
        n = len(PRESETS[name]["constraints"])
        print(f"   {name:<12} — {n} constraints")

    # 5. Fracture-coalesce
    print("\n5. Fracture-Coalesce")
    print("-" * 40)
    # 4 constraints sharing 3 dimensions → 2 independent blocks
    graph = DependencyGraph.from_masks([
        np.array([0, 1]),   # c0 touches dims 0,1
        np.array([0]),      # c1 touches dim 0
        np.array([2]),      # c2 touches dim 2 (independent)
        np.array([2]),      # c3 touches dim 2
    ], constraint_names=["c0", "c1", "c2", "c3"])
    fracturer = Fracturer()
    result = fracturer.fracture(graph)
    print(f"   Input: 4 constraints, 3 dimensions")
    print(f"   Output: {result.n_blocks} independent blocks")
    for i, block in enumerate(result.blocks):
        print(f"   Block {i}: constraints {block.constraint_indices}, dims {block.dimension_indices}")
    print(f"   Speedup potential: {result.speedup_potential:.1f}x")

    # 6. Sediment
    print("\n6. Sediment Layers")
    print("-" * 40)
    stack = SedimentStack()
    stack.add_layer(
        input_context={"crisis": "cold start false alarm"},
        corrections=[ConstraintCorrection(
            constraint_name="coolant_temp_c",
            override_pass=True,
            reason="Engine cold start allows temps below -40 briefly",
        )],
        provenance="demo",
    )
    stats = stack.coverage_stats()
    print(f"   Layers: {stats['active_layers']} active, {stats['total_corrections']} corrections")

    print("\n" + "=" * 60)
    print("  Demo complete. Install and use: flux-check --help")
    print("=" * 60)


if __name__ == "__main__":
    main()
