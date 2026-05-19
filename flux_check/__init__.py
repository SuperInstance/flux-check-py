"""flux-check — FLUX constraint engine: zero false negatives."""

from flux_check.core import FluxExact, check_exact, check_batch
from flux_check.presets import PRESETS, list_presets

__all__ = ["FluxExact", "check_exact", "check_batch", "PRESETS", "list_presets"]
__version__ = "0.1.0"
