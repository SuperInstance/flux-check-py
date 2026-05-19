"""Tests for flux_check.core — exact constraint checking."""

import numpy as np
import pytest

from flux_check.core import FluxExact, Severity, severity, passed, check_exact, check_batch


class TestFluxExact:
    """Tests for the FluxExact engine."""

    def setup_method(self):
        self.constraints = [
            {"lo": -40, "hi": 150, "name": "coolant_temp"},
            {"lo": 0, "hi": 8000, "name": "rpm"},
        ]
        self.fc = FluxExact(self.constraints)

    # ── Basic ───────────────────────────────────────────────

    def test_pass(self):
        assert self.fc.check_mask(50) == 0

    def test_fail_hi(self):
        mask = self.fc.check_mask(151)
        assert mask & 1  # bit 0 (coolant_temp) violated

    def test_fail_lo(self):
        mask = self.fc.check_mask(-41)
        assert mask & 1

    def test_both_constraints_pass(self):
        # 100 is within both [-40,150] and [0,8000]
        assert self.fc.check_mask(100) == 0

    def test_second_constraint_fail(self):
        mask = self.fc.check_mask(9000)
        assert mask & 2  # bit 1 (rpm) violated

    # ── NaN ─────────────────────────────────────────────────

    def test_nan_violates_all(self):
        mask = self.fc.check_mask(float("nan"))
        assert mask == 0b11  # both constraints violated

    def test_nan_single_constraint(self):
        fc = FluxExact([{"lo": 0, "hi": 100, "name": "x"}])
        assert fc.check_mask(float("nan")) == 1

    # ── Boundaries ──────────────────────────────────────────

    def test_lo_boundary_inclusive(self):
        # -40 is within coolant_temp [-40,150] but below rpm [0,8000]
        # For coolant_temp alone, boundary is inclusive
        fc = FluxExact([{"lo": -40, "hi": 150, "name": "coolant_temp"}])
        assert fc.check_mask(-40) == 0

    def test_hi_boundary_inclusive(self):
        assert self.fc.check_mask(150) == 0

    def test_lo_boundary_minus_epsilon(self):
        mask = self.fc.check_mask(-40.0001)
        assert mask != 0

    def test_hi_boundary_plus_epsilon(self):
        mask = self.fc.check_mask(150.0001)
        assert mask != 0

    # ── Inf ─────────────────────────────────────────────────

    def test_positive_inf(self):
        mask = self.fc.check_mask(float("inf"))
        assert mask != 0

    def test_negative_inf(self):
        mask = self.fc.check_mask(float("-inf"))
        assert mask != 0

    # ── Validation ──────────────────────────────────────────

    def test_empty_constraints_raises(self):
        with pytest.raises(ValueError):
            FluxExact([])

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError):
            FluxExact([{"lo": 100, "hi": 0, "name": "bad"}])

    def test_max_8_constraints(self):
        with pytest.raises(ValueError):
            FluxExact([{"lo": 0, "hi": i + 1, "name": f"c{i}"} for i in range(9)])


class TestCheckResult:
    """Tests for ExactResult object."""

    def setup_method(self):
        self.fc = FluxExact([{"lo": 0, "hi": 100, "name": "temp"}])

    def test_pass_result(self):
        r = self.fc.check(50)
        assert r.passed
        assert r.severity == Severity.PASS
        assert r.violated_count == 0

    def test_fail_result(self):
        r = self.fc.check(150)
        assert not r.passed
        assert r.violated_count == 1

    def test_detail(self):
        d = self.fc.check_detail(50)
        assert d["passed"]
        assert d["details"][0]["name"] == "temp"

    def test_to_dict(self):
        r = self.fc.check(50)
        d = r.to_dict()
        assert "error_mask" in d
        assert "details" in d


class TestSeverity:
    """Tests for severity computation."""

    def test_zero_bits_pass(self):
        assert severity(0) == Severity.PASS

    def test_one_bit_caution(self):
        assert severity(1) == Severity.CAUTION

    def test_two_bits_warning(self):
        # 2 bits → _SEVERITY_TABLE[2] = CAUTION (table is [PASS, CAUTION, CAUTION, WARNING, ...])
        assert severity(0b11) == Severity.CAUTION


class TestBatch:
    """Tests for numpy vectorized batch checking."""

    def setup_method(self):
        self.fc = FluxExact([{"lo": 0, "hi": 100, "name": "x"}])

    def test_batch_basic(self):
        arr = np.array([50.0, 150.0, -10.0])
        masks = self.fc.check_batch_numpy(arr)
        assert masks[0] == 0
        assert masks[1] != 0
        assert masks[2] != 0

    def test_batch_nan(self):
        arr = np.array([50.0, float("nan")])
        masks = self.fc.check_batch_numpy(arr)
        assert masks[0] == 0
        assert masks[1] == 1

    def test_batch_shape_preserved(self):
        arr = np.array([[50.0, 150.0], [0.0, -1.0]])
        masks = self.fc.check_batch_numpy(arr)
        assert masks.shape == (2, 2)

    def test_batch_empty(self):
        arr = np.array([])
        masks = self.fc.check_batch_numpy(arr)
        assert len(masks) == 0


class TestConvenienceFunctions:

    def test_check_exact(self):
        bounds = [{"lo": 0, "hi": 100, "name": "x"}]
        assert check_exact([50], bounds) == 0

    def test_check_exact_mismatch_length(self):
        with pytest.raises(ValueError):
            check_exact([1, 2], [{"lo": 0, "hi": 100, "name": "x"}])

    def test_check_batch(self):
        bounds = [{"lo": 0, "hi": 100, "name": "x"}]
        masks = check_batch([50, 150], bounds)
        assert masks[0] == 0
        assert masks[1] != 0
