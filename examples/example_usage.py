#!/usr/bin/env python3
"""Example: Using flux-check as a library."""

from flux_check import FluxExact, check_batch
from flux_check.presets import get_preset
import numpy as np

# 1. Load a preset
constraints = get_preset("automotive")
fc = FluxExact(constraints)

# 2. Check a single value (against all constraints)
print("Single value check:")
mask = fc.check_mask(160)
print(f"  160 → mask=0x{mask:02x} → {'PASS' if mask == 0 else 'FAIL'}")

# 3. Full detail
print("\nFull detail:")
result = fc.check(160)
for d in result.details:
    print(f"  {d.name}: {d.value} in [{d.lo}, {d.hi}] → {'PASS' if d.passed else 'FAIL'}")

# 4. Batch check with numpy
print("\nBatch check:")
values = np.array([50.0, 3000.0, 9000.0, float("nan"), -100.0])
masks = fc.check_batch_numpy(values)
for v, m in zip(values, masks):
    print(f"  {str(v):>8} → mask=0x{int(m):02x}")
