"""
FLUX Exact Constraint Engine — Zero False Negatives.

INVARIANT: A value outside bounds is ALWAYS detected. No exceptions.
NaN always violates all constraints. No opt-in required.

Adapted from flux_constraint_exact.py for the flux-check CLI package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Union

import numpy as np

Number = Union[int, float]

# ── Severity ────────────────────────────────────────────────

class Severity(IntEnum):
    PASS = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3

_SEVERITY_TABLE = [
    Severity.PASS, Severity.CAUTION, Severity.CAUTION,
    Severity.WARNING, Severity.WARNING,
    Severity.CRITICAL, Severity.CRITICAL, Severity.CRITICAL, Severity.CRITICAL,
]


def severity(mask: int) -> Severity:
    n = bin(mask).count("1")
    return _SEVERITY_TABLE[n] if n < len(_SEVERITY_TABLE) else Severity.CRITICAL

def passed(mask: int) -> bool:
    return mask == 0


# ── Data classes ────────────────────────────────────────────

@dataclass
class ExactConstraintDef:
    lo: float
    hi: float
    name: str

    def __post_init__(self):
        self.lo = float(self.lo)
        self.hi = float(self.hi)
        if self.lo > self.hi:
            raise ValueError(f"Constraint '{self.name}': lo ({self.lo}) > hi ({self.hi})")


@dataclass
class ExactDetail:
    name: str
    lo: float
    hi: float
    value: float
    passed: bool
    lo_violated: bool
    hi_violated: bool


@dataclass
class ExactResult:
    error_mask: int = 0
    severity: Severity = Severity.PASS
    violated_lo: int = 0
    violated_hi: int = 0
    violated_count: int = 0
    details: List[ExactDetail] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error_mask == 0

    def to_dict(self) -> dict:
        return {
            "error_mask": self.error_mask,
            "severity": int(self.severity),
            "severity_name": self.severity.name,
            "violated_lo": self.violated_lo,
            "violated_hi": self.violated_hi,
            "violated_count": self.violated_count,
            "passed": self.passed,
            "details": [
                {"name": d.name, "lo": d.lo, "hi": d.hi, "value": d.value, "passed": d.passed}
                for d in self.details
            ],
        }


# ── FluxExact engine ───────────────────────────────────────

class FluxExact:
    """
    FLUX Exact Constraint Engine — Zero False Negatives.

    check_mask()  → int error_mask    (zero-alloc hot path)
    check()       → ExactResult       (backward compat)
    check_batch() → np.ndarray uint8  (numpy vectorized)
    check_detail() → dict             (allocates, full info)
    """

    __slots__ = ("_lo", "_hi", "_names", "n", "constraints")

    def __init__(self, constraints: List[Dict]):
        if not constraints:
            raise ValueError("FluxExact requires non-empty constraints list")
        if len(constraints) > 8:
            raise ValueError("Maximum 8 constraints (error_mask is uint8)")

        self._lo = tuple(float(c["lo"]) for c in constraints)
        self._hi = tuple(float(c["hi"]) for c in constraints)
        self._names = tuple(c.get("name", f"C{i}") for i, c in enumerate(constraints))
        self.n = len(constraints)

        for i in range(self.n):
            if self._lo[i] > self._hi[i]:
                raise ValueError(
                    f"Constraint '{self._names[i]}': lo ({self._lo[i]}) > hi ({self._hi[i]})"
                )

        self.constraints = [
            ExactConstraintDef(lo=self._lo[i], hi=self._hi[i], name=self._names[i])
            for i in range(self.n)
        ]

    def check_mask(self, value: Number) -> int:
        """Check value. Returns error_mask (0 = all pass). Zero allocations."""
        v = float(value)
        if v != v:  # NaN
            return (1 << self.n) - 1
        mask = 0
        for i in range(self.n):
            if v < self._lo[i] or v > self._hi[i]:
                mask |= (1 << i)
        return mask

    def check(self, value: Number) -> ExactResult:
        """Check value. Returns ExactResult with .passed, .severity, .details."""
        v = float(value)
        is_nan = v != v
        mask = 0
        lo_mask = 0
        hi_mask = 0
        details = []

        for i in range(self.n):
            if is_nan:
                lo_f = hi_f = True
            else:
                lo_f = v < self._lo[i]
                hi_f = v > self._hi[i]
            p = not lo_f and not hi_f
            if not p:
                mask |= (1 << i)
            if lo_f:
                lo_mask |= (1 << i)
            if hi_f:
                hi_mask |= (1 << i)
            details.append(ExactDetail(
                name=self._names[i], lo=self._lo[i], hi=self._hi[i], value=v,
                passed=p, lo_violated=lo_f, hi_violated=hi_f,
            ))

        vc = bin(mask).count("1")
        return ExactResult(
            error_mask=mask,
            severity=_SEVERITY_TABLE[vc] if vc < len(_SEVERITY_TABLE) else Severity.CRITICAL,
            violated_lo=lo_mask,
            violated_hi=hi_mask,
            violated_count=vc,
            details=details,
        )

    def check_batch_numpy(self, values) -> np.ndarray:
        """Vectorized batch check. Returns np.ndarray of uint8 error_masks."""
        vals = np.asarray(values, dtype=np.float64)
        flat = vals.ravel()
        masks = np.zeros(len(flat), dtype=np.uint8)
        nan_mask = np.isnan(flat)
        if self.n <= 8:
            masks[nan_mask] = np.uint8((1 << self.n) - 1)
        valid = ~nan_mask
        for i in range(self.n):
            violated = valid & ((flat < self._lo[i]) | (flat > self._hi[i]))
            masks[violated] |= np.uint8(1 << i)
        return masks.reshape(vals.shape)

    def check_detail(self, value: Number) -> dict:
        """Full result as dict. Allocates — not for hot path."""
        v = float(value)
        is_nan = v != v
        mask = 0
        lo_mask = 0
        hi_mask = 0
        details = []
        for i in range(self.n):
            if is_nan:
                lo_f = hi_f = True
            else:
                lo_f = v < self._lo[i]
                hi_f = v > self._hi[i]
            p = not lo_f and not hi_f
            if not p:
                mask |= (1 << i)
            if lo_f:
                lo_mask |= (1 << i)
            if hi_f:
                hi_mask |= (1 << i)
            details.append({
                "name": self._names[i], "lo": self._lo[i], "hi": self._hi[i],
                "value": v, "passed": p, "lo_violated": lo_f, "hi_violated": hi_f,
            })
        vc = bin(mask).count("1")
        return {
            "error_mask": mask,
            "severity": int(_SEVERITY_TABLE[vc] if vc < len(_SEVERITY_TABLE) else Severity.CRITICAL),
            "violated_lo": lo_mask, "violated_hi": hi_mask,
            "violated_count": vc, "passed": mask == 0, "details": details,
        }

    def benchmark(self, iterations: int = 1_000_000) -> float:
        """Returns checks/sec."""
        t0 = time.perf_counter()
        for i in range(iterations):
            self.check_mask((i % 1000) - 500)
        return iterations / (time.perf_counter() - t0)

    def benchmark_detail(self, iterations: int = 1_000_000) -> Dict:
        rate = self.benchmark(iterations)
        return {
            "rate": rate, "rate_M": rate / 1e6,
            "total_ms": iterations / rate * 1000,
            "iterations": iterations, "constraints": self.n,
        }

    @classmethod
    def from_preset(cls, name: str, presets: Dict[str, List[Dict]] = None) -> "FluxExact":
        from flux_check.presets import PRESETS
        src = presets or PRESETS
        if name not in src:
            raise ValueError(f"Unknown preset: {name}. Available: {', '.join(src.keys())}")
        data = src[name]
        constraints = data["constraints"] if isinstance(data, dict) else data
        return cls(constraints)


# ── Convenience functions ───────────────────────────────────

def check_exact(values: List[float], bounds: List[Dict]) -> int:
    """
    One-shot exact check. Returns combined error mask.

    Args:
        values: list of values, one per constraint
        bounds: list of {"lo": ..., "hi": ..., "name": ...} dicts

    Returns:
        int error_mask (0 = all pass)
    """
    if len(values) != len(bounds):
        raise ValueError(f"Values length ({len(values)}) != bounds length ({len(bounds)})")
    fc = FluxExact(bounds)
    combined = 0
    for i, v in enumerate(values):
        result = fc.check(v)
        if not result.details[i].passed:
            combined |= (1 << i)
    return combined


def check_batch(values_array, bounds: List[Dict]) -> np.ndarray:
    """
    Batch check: array of values against constraints.

    Args:
        values_array: array-like of values
        bounds: list of {"lo": ..., "hi": ...} dicts

    Returns:
        np.ndarray of uint8 error_masks
    """
    fc = FluxExact(bounds)
    return fc.check_batch_numpy(values_array)
