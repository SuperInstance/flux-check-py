# flux-check-py

Python CLI and library for exact constraint checking — zero false negatives, vectorized batch operations, and constraint graph fracture-coalesce analysis.

## What It Does

`flux-check-py` is a constraint engine that checks whether values fall within defined bounds. It guarantees **zero false negatives**: any value outside bounds is always detected, including NaN. It ships with 6 industry presets (automotive, aviation, medical, energy, IoT, financial), supports NumPy-vectorized batch checking, and can fracture constraint graphs into independent blocks for parallel evaluation.

## Installation

```bash
pip install flux-constraint-check
```

Requires Python ≥ 3.9 and NumPy ≥ 1.21.

## Quick Start

### CLI

```bash
# List available presets
flux-check presets

# Check a sensor vector against the automotive preset
flux-check check-vector --preset automotive --values 3000,120,90,50,100,0,12,75

# Batch check from CSV
flux-check batch --preset automotive --input sensors.csv --output results.csv

# Fracture a constraint dependency graph
flux-check fracture --graph graph.json

# Benchmark
flux-check bench --preset automotive --iterations 1000000
```

### Python Library

```python
from flux_check import FluxExact, check_batch
from flux_check.presets import get_preset
import numpy as np

# Load a preset
constraints = get_preset("automotive")
fc = FluxExact(constraints)

# Single value check (zero-alloc hot path)
mask = fc.check_mask(160)  # Returns error bitmask (0 = all pass)

# Full detail result
result = fc.check(160)
for d in result.details:
    print(f"{d.name}: {'PASS' if d.passed else 'FAIL'}")

# Vectorized batch check
values = np.array([50.0, 3000.0, 9000.0, float("nan")])
masks = fc.check_batch_numpy(values)

# Check a sensor vector (value[i] vs constraint[i])
from flux_check.core import check_vector
result = check_vector([3000, 120, 90, 50, 100, 0, 12, 75], constraints)
```

## API

### `FluxExact(constraints)`

Core engine. Accepts a list of `{"lo": float, "hi": float, "name": str}` dicts (max 8).

| Method | Returns | Description |
|--------|---------|-------------|
| `check_mask(value)` | `int` | Zero-alloc error bitmask (0 = pass) |
| `check(value)` | `ExactResult` | Full result with severity + per-constraint details |
| `check_detail(value)` | `dict` | Allocating full-detail check |
| `check_batch_numpy(array)` | `np.ndarray[uint8]` | Vectorized batch, one mask per value |
| `benchmark(iterations)` | `float` | Checks/sec throughput |

### `check_vector(values, constraints)` → `ExactResult`

Check a sensor array where `value[i]` maps to `constraint[i]`. Returns combined mask with per-constraint detail.

### `check_vector_batch(matrix, constraints)` → `np.ndarray[uint8]`

Batch version of the vector check for 2D arrays (rows = samples, cols = constraints).

### Fracture-Coalesce

```python
from flux_check.fracture import Fracturer, DependencyGraph

# Define which constraints share dimensions
graph = DependencyGraph.from_masks([
    np.array([0, 1]),  # constraint 0 touches dims 0, 1
    np.array([0]),     # constraint 1 touches dim 0
    np.array([2]),     # constraint 2 touches dim 2
], constraint_names=["c0", "c1", "c2"])

result = Fracturer().fracture(graph)
# result.blocks → independent blocks for parallel evaluation
# result.speedup_potential → parallelism factor
```

### Sediment Layers

Accumulated edge-case corrections as immutable layers:

```python
from flux_check.sediment import SedimentStack, ConstraintCorrection

stack = SedimentStack()
stack.add_layer(
    input_context={"scenario": "cold start"},
    corrections=[ConstraintCorrection(
        constraint_name="coolant_temp_c",
        override_pass=True,
        reason="Cold start allows brief sub-range readings",
    )],
)
```

## Industry Presets

| Preset | Constraints | Domain |
|--------|------------|--------|
| `automotive` | 8 | CAN bus sensor ranges |
| `aviation` | 8 | ADS-B / flight data |
| `medical` | 8 | Vital signs (FHIR-compatible) |
| `energy` | 8 | Grid SCADA ranges |
| `iot` | 8 | MQTT environmental sensors |
| `financial` | 8 | FIX protocol ranges |

## Severity Levels

| Severity | Violated Constraints |
|----------|---------------------|
| PASS (0) | 0 |
| CAUTION (1) | 1 |
| WARNING (2) | 2–3 |
| CRITICAL (3) | 4+ |

## Testing

```bash
pip install pytest && pytest tests/
```

74 tests covering core checking, presets, NaN handling, batch operations, fracture-coalesce, sediment layers, and CLI commands.

## Related Repos

- **[flux-fracture-c](https://github.com/SuperInstance/flux-fracture-c)** — C99 single-header fracture-coalesce library
- **[constraint-theory-rust-python](https://github.com/SuperInstance/constraint-theory-rust-python)** — Rust engine with PyO3 Python bindings
- **[constraint-theory-engine-cpp-lua](https://github.com/SuperInstance/constraint-theory-engine-cpp-lua)** — C++ engine with LuaJIT orchestration, CDCL solver, AVX-512

## License

MIT
