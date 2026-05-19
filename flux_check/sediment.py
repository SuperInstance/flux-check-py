"""
FLUX Sediment — Accumulated Correctness as Computational Sediment.

Models constraint correctness as geological sediment: layers of edge-case
corrections that accumulate over time, each layer immutable, new layers
superseding specific corrections from older ones.

Core theorem: A constraint system with N sediment layers has strictly
higher correctness than the same system with fewer layers, converging
monotonically toward complete coverage.

Adapted from flux_sediment.py for the flux-check CLI package.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── ConstraintCorrection ────────────────────────────────────

@dataclass(frozen=True)
class ConstraintCorrection:
    """A single correction to a constraint definition."""
    constraint_name: str
    old_lo: Optional[float] = None
    old_hi: Optional[float] = None
    new_lo: Optional[float] = None
    new_hi: Optional[float] = None
    override_pass: Optional[bool] = None
    reason: str = ""

    def apply_to(self, lo: float, hi: float, passed: bool) -> Tuple[float, float, bool]:
        out_lo = self.new_lo if self.new_lo is not None else lo
        out_hi = self.new_hi if self.new_hi is not None else hi
        out_passed = self.override_pass if self.override_pass is not None else passed
        return out_lo, out_hi, out_passed

    def to_dict(self) -> dict:
        return {
            "constraint_name": self.constraint_name,
            "old_lo": self.old_lo, "old_hi": self.old_hi,
            "new_lo": self.new_lo, "new_hi": self.new_hi,
            "override_pass": self.override_pass,
            "reason": self.reason,
        }


# ── SedimentLayer ───────────────────────────────────────────

@dataclass
class SedimentLayer:
    """An immutable layer of edge-case corrections."""
    layer_id: int
    input_context: Dict[str, Any]
    corrections: List[ConstraintCorrection]
    timestamp: float = field(default_factory=time.time)
    provenance: str = ""
    model: str = ""
    superseded: bool = False
    superseded_by: Optional[int] = None
    catch_count: int = 0

    def content_hash(self) -> str:
        blob = json.dumps(
            {"id": self.layer_id, "corrections": [c.to_dict() for c in self.corrections]},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_tile(self) -> dict:
        return {
            "tile_type": "sediment_layer",
            "layer_id": self.layer_id,
            "content_hash": self.content_hash(),
            "input_context": self.input_context,
            "corrections": [c.to_dict() for c in self.corrections],
            "timestamp": self.timestamp,
            "provenance": self.provenance,
            "model": self.model,
            "superseded": self.superseded,
            "superseded_by": self.superseded_by,
            "catch_count": self.catch_count,
        }

    @classmethod
    def from_tile(cls, tile: dict) -> "SedimentLayer":
        corrections = [ConstraintCorrection(**c) for c in tile.get("corrections", [])]
        return cls(
            layer_id=tile["layer_id"],
            input_context=tile.get("input_context", {}),
            corrections=corrections,
            timestamp=tile.get("timestamp", 0.0),
            provenance=tile.get("provenance", ""),
            model=tile.get("model", ""),
            superseded=tile.get("superseded", False),
            superseded_by=tile.get("superseded_by"),
            catch_count=tile.get("catch_count", 0),
        )


# ── SedimentResult ──────────────────────────────────────────

@dataclass
class SedimentResult:
    """Result of running a check through the sediment stack."""
    base_error_mask: int
    base_severity: int
    final_error_mask: int
    final_severity: int
    layers_applied: List[int]
    corrections_applied: int
    passed: bool

    def to_dict(self) -> dict:
        return {
            "base_error_mask": self.base_error_mask,
            "base_severity": self.base_severity,
            "final_error_mask": self.final_error_mask,
            "final_severity": self.final_severity,
            "layers_applied": self.layers_applied,
            "corrections_applied": self.corrections_applied,
            "passed": self.passed,
        }


# ── SedimentStack ───────────────────────────────────────────

class SedimentStack:
    """
    Stack of sediment layers (oldest at bottom, newest at top).
    Layers are NEVER deleted, only superseded.
    """

    def __init__(self):
        self._layers: List[SedimentLayer] = []
        self._next_id: int = 0

    @property
    def depth(self) -> int:
        return len(self._layers)

    @property
    def active_layers(self) -> List[SedimentLayer]:
        return [l for l in self._layers if not l.superseded]

    def add_layer(
        self,
        input_context: Dict[str, Any],
        corrections: List[ConstraintCorrection],
        provenance: str = "",
        model: str = "",
    ) -> SedimentLayer:
        layer = SedimentLayer(
            layer_id=self._next_id,
            input_context=input_context,
            corrections=corrections,
            provenance=provenance,
            model=model,
        )
        self._layers.append(layer)
        self._next_id += 1
        return layer

    def supersede_layer(self, old_id: int, new_id: int) -> bool:
        for layer in self._layers:
            if layer.layer_id == old_id and not layer.superseded:
                layer.superseded = True
                layer.superseded_by = new_id
                return True
        return False

    def check_with_sediment(
        self,
        base_error_mask: int,
        base_severity: int,
        constraint_names: List[str],
        values: Dict[str, float],
        constraint_defs: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> SedimentResult:
        current_mask = base_error_mask
        current_severity = base_severity
        layers_applied: List[int] = []
        corrections_applied = 0

        accumulated_bounds: Dict[str, Tuple[float, float]] = {}
        if constraint_defs:
            accumulated_bounds = dict(constraint_defs)

        for layer in self._layers:
            if layer.superseded:
                continue

            for correction in layer.corrections:
                if correction.constraint_name not in constraint_names:
                    continue

                bit_idx = constraint_names.index(correction.constraint_name)
                bit = 1 << bit_idx
                is_violated = bool(current_mask & bit)

                if correction.override_pass is not None:
                    if correction.override_pass and is_violated:
                        current_mask &= ~bit
                        corrections_applied += 1
                        layers_applied.append(layer.layer_id)
                    elif not correction.override_pass and not is_violated:
                        current_mask |= bit
                        corrections_applied += 1
                        layers_applied.append(layer.layer_id)
                elif constraint_defs and correction.constraint_name in constraint_defs:
                    orig_lo, orig_hi = accumulated_bounds.get(
                        correction.constraint_name,
                        constraint_defs[correction.constraint_name],
                    )
                    new_lo, new_hi, _ = correction.apply_to(orig_lo, orig_hi, not is_violated)
                    val = values.get(correction.constraint_name, float("nan"))
                    now_violated = val < new_lo or val > new_hi or val != val
                    if now_violated:
                        current_mask |= bit
                    else:
                        current_mask &= ~bit
                    accumulated_bounds[correction.constraint_name] = (new_lo, new_hi)
                    corrections_applied += 1
                    layers_applied.append(layer.layer_id)

        return SedimentResult(
            base_error_mask=base_error_mask,
            base_severity=base_severity,
            final_error_mask=current_mask,
            final_severity=current_severity,
            layers_applied=layers_applied,
            corrections_applied=corrections_applied,
            passed=current_mask == 0,
        )

    def coverage_stats(self) -> Dict[str, Any]:
        active = self.active_layers
        total_corrections = sum(len(l.corrections) for l in active)
        total_catches = sum(l.catch_count for l in self._layers)
        return {
            "total_layers": len(self._layers),
            "active_layers": len(active),
            "superseded_layers": len(self._layers) - len(active),
            "total_corrections": total_corrections,
            "total_catches": total_catches,
            "depth": self.depth,
        }
