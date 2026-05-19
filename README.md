# flux-check

A command-line tool for exact constraint checking. Feed it values and bounds, and it tells you — unambiguously — whether everything is within spec.

```bash
pip install -e .
```

## How It Works

Every check boils down to one question: *is this value between its lower and upper bound?* The answer is always exact — no approximations, no probabilities. If a value is outside bounds, the check fails. NaN always fails.

Results come back as a **bitmask**: bit 0 corresponds to constraint 0, bit 1 to constraint 1, and so on. A mask of `0` means everything passed. A mask of `5` (binary `0b101`) means constraints 0 and 2 both violated. This representation is cheap to compute, cheap to store, and cheap to combine.

Three subsystems layered on top of that core:

1. **core** — The hot path. Zero allocations on the single-value check. Numpy-vectorized for batch.
2. **fracture** — Splits a set of constraints into independent blocks (ones that share no dimensions). Each block can be checked in parallel, then results are merged with bitwise OR.
3. **sediment** — Accumulates edge-case corrections as immutable layers. Think of it as geological strata: each layer corrects known false alarms without rewriting the ones below it.

## What This Module Does

It's a CLI and Python library for checking whether sensor values, financial data, or any numeric measurements fall within defined tolerance ranges. It ships with six industry presets (automotive, aviation, medical, energy, IoT, financial) so you can start checking immediately.

## CLI Usage

```bash
# See available presets
flux-check presets

# Check 8 values against automotive's 8 constraints
flux-check check --preset automotive --values 3000,50,90,50,100,0,12,75
# → PASS

# Force a failure (9000 RPM exceeds the 8000 RPM limit)
flux-check check --preset automotive --values 9000,50,90,50,100,0,12,75
# → CRITICAL (1 constraint violated)

# Batch check from CSV (numpy-vectorized under the hood)
flux-check batch --preset automotive --input examples/automotive_data.csv --output results.csv

# Fracture a constraint graph into independent blocks
flux-check fracture --graph examples/graph_example.json

# Benchmark your system
flux-check bench --preset automotive --iterations 1000000
```

## As a Library

```python
from flux_check import FluxExact
from flux_check.presets import get_preset

fc = FluxExact(get_preset("automotive"))

# Fast path: returns a uint8 bitmask (0 = all pass)
mask = fc.check_mask(9000)  # → 1 (bit 0 set: first constraint violated)

# Full detail
result = fc.check(9000)
print(result.passed)          # False
print(result.severity.name)   # "CRITICAL"
print(result.details[0])      # ExactDetail with name, lo, hi, value, passed

# Batch: numpy vectorized
import numpy as np
masks = fc.check_batch_numpy(np.array([3000.0, 9000.0, float("nan")]))
# → array([0, 1, 255], dtype=uint8)
# NaN violates everything → all bits set
```

## Fracture-Coalesce

When constraints touch different dimensions (e.g., temperature and RPM don't depend on each other), you can split them into independent blocks:

```python
from flux_check.fracture import Fracturer, DependencyGraph
import numpy as np

graph = DependencyGraph.from_masks([
    np.array([0, 1]),  # c0: touches dims 0 and 1
    np.array([0]),     # c1: touches dim 0 (linked to c0)
    np.array([2]),     # c2: touches dim 2 (independent)
    np.array([2]),     # c3: touches dim 2 (linked to c2)
])

result = Fracturer().fracture(graph)
# → 2 blocks: {c0, c1} and {c2, c3}
```

Merge results from parallel blocks with bitwise OR:

```python
from flux_check import coalesce
total = coalesce([0b01, 0b10])  # → 0b11
```

## Sediment Layers

Edge-case corrections accumulate as immutable layers — new corrections overlay old ones without mutating them:

```python
from flux_check.sediment import SedimentStack, ConstraintCorrection

stack = SedimentStack()
stack.add_layer(
    input_context={"crisis": "cold start false alarm"},
    corrections=[ConstraintCorrection(
        constraint_name="coolant_temp_c",
        override_pass=True,
        reason="Engine cold start can briefly dip below -40°C",
    )],
)

result = stack.check_with_sediment(
    base_error_mask=0b001,
    base_severity=1,
    constraint_names=["coolant_temp_c", "rpm", "speed"],
    values={"coolant_temp_c": -50, "rpm": 3000, "speed": 50},
)
print(result.passed)  # True — sediment corrected the false alarm
```

## Presets

| Preset | Constraints | Domain |
|--------|-------------|--------|
| automotive | 8 | CAN bus sensors |
| aviation | 8 | ADS-B / flight data |
| medical | 8 | Vital signs (FHIR) |
| energy | 8 | SCADA grid data |
| iot | 8 | MQTT environmental |
| financial | 8 | FIX protocol |

Each preset has 8 constraints with bounds derived from industry standards.

## Performance

Benchmarks should be run on your own hardware with `flux-check bench`. The hot-path single check is zero-allocation. Batch mode uses numpy vectorization — throughput scales with your hardware's SIMD width.

## Where to Go Next

| If you want... | Go to |
|----------------|-------|
| The unified library (all modules in one import) | [flux-lib](../flux-lib-py) |
| Thermodynamic analysis of constraints | [flux-lib](../flux-lib-py) — `ThermoEngine` |
| Hyperbolic geometry for model routing | [flux-hyperbolic](../flux-hyperbolic-py) |
| Genetic expression engine | [flux-genome](../flux-genome-py) |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
