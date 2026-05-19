"""Tests for flux_check.sediment — sediment layers and corrections."""

import pytest

from flux_check.sediment import (
    ConstraintCorrection, SedimentLayer, SedimentStack,
)


class TestConstraintCorrection:

    def test_apply_bounds(self):
        c = ConstraintCorrection("temp", new_lo=-50, new_hi=200)
        lo, hi, passed = c.apply_to(-40, 150, True)
        assert lo == -50
        assert hi == 200

    def test_override_pass(self):
        c = ConstraintCorrection("temp", override_pass=True)
        _, _, p = c.apply_to(0, 100, False)
        assert p is True

    def test_no_change(self):
        c = ConstraintCorrection("temp")
        lo, hi, p = c.apply_to(0, 100, True)
        assert lo == 0
        assert hi == 100
        assert p is True

    def test_to_dict(self):
        c = ConstraintCorrection("temp", new_lo=-50, reason="cold start")
        d = c.to_dict()
        assert d["constraint_name"] == "temp"
        assert d["new_lo"] == -50
        assert d["reason"] == "cold start"


class TestSedimentLayer:

    def test_create(self):
        layer = SedimentLayer(
            layer_id=0,
            input_context={"crisis": "test"},
            corrections=[ConstraintCorrection("x", override_pass=True)],
        )
        assert layer.layer_id == 0
        assert not layer.superseded

    def test_content_hash_deterministic(self):
        layer = SedimentLayer(
            layer_id=0,
            input_context={},
            corrections=[ConstraintCorrection("x")],
        )
        assert layer.content_hash() == layer.content_hash()

    def test_tile_round_trip(self):
        layer = SedimentLayer(
            layer_id=1,
            input_context={"test": True},
            corrections=[ConstraintCorrection("x", new_lo=-10)],
            provenance="test",
        )
        tile = layer.to_tile()
        restored = SedimentLayer.from_tile(tile)
        assert restored.layer_id == 1
        assert len(restored.corrections) == 1
        assert restored.provenance == "test"


class TestSedimentStack:

    def test_add_layer(self):
        stack = SedimentStack()
        layer = stack.add_layer(
            input_context={"test": True},
            corrections=[ConstraintCorrection("x", override_pass=True)],
        )
        assert stack.depth == 1
        assert layer.layer_id == 0

    def test_active_layers(self):
        stack = SedimentStack()
        stack.add_layer({}, [ConstraintCorrection("x")])
        stack.add_layer({}, [ConstraintCorrection("y")])
        assert len(stack.active_layers) == 2

    def test_supersede(self):
        stack = SedimentStack()
        l0 = stack.add_layer({}, [ConstraintCorrection("x")])
        l1 = stack.add_layer({}, [ConstraintCorrection("y")])
        assert stack.supersede_layer(l0.layer_id, l1.layer_id)
        assert len(stack.active_layers) == 1

    def test_check_with_sediment_override_pass(self):
        """Override can turn a FAIL into PASS."""
        stack = SedimentStack()
        stack.add_layer(
            input_context={"crisis": "cold start"},
            corrections=[ConstraintCorrection(
                "coolant_temp", override_pass=True, reason="cold start OK"
            )],
        )
        result = stack.check_with_sediment(
            base_error_mask=0b01,  # coolant_temp violated
            base_severity=1,
            constraint_names=["coolant_temp"],
            values={"coolant_temp": -50},
        )
        assert result.passed
        assert result.base_error_mask != result.final_error_mask

    def test_check_with_sediment_override_fail(self):
        """Override can turn a PASS into FAIL."""
        stack = SedimentStack()
        stack.add_layer(
            input_context={"crisis": "sensor fault"},
            corrections=[ConstraintCorrection(
                "rpm", override_pass=False, reason="sensor unreliable"
            )],
        )
        result = stack.check_with_sediment(
            base_error_mask=0b00,
            base_severity=0,
            constraint_names=["rpm"],
            values={"rpm": 3000},
        )
        assert not result.passed

    def test_monotonic_coverage(self):
        """More layers = more corrections available."""
        stack = SedimentStack()
        stack.add_layer({}, [ConstraintCorrection("x")])
        stats1 = stack.coverage_stats()
        stack.add_layer({}, [ConstraintCorrection("y"), ConstraintCorrection("z")])
        stats2 = stack.coverage_stats()
        assert stats2["total_corrections"] > stats1["total_corrections"]

    def test_coverage_stats(self):
        stack = SedimentStack()
        stack.add_layer({}, [ConstraintCorrection("x")])
        stats = stack.coverage_stats()
        assert stats["total_layers"] == 1
        assert stats["active_layers"] == 1
        assert stats["total_corrections"] == 1
