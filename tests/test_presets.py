"""Tests for flux_check.presets — preset validation."""

import pytest

from flux_check.presets import PRESETS, list_presets, get_preset, get_preset_description


class TestPresets:

    def test_all_presets_exist(self):
        assert len(PRESETS) >= 6

    def test_list_presets(self):
        names = list_presets()
        assert "automotive" in names
        assert "aviation" in names
        assert "medical" in names
        assert "energy" in names
        assert "iot" in names
        assert "financial" in names

    def test_get_preset(self):
        constraints = get_preset("automotive")
        assert len(constraints) == 8
        assert all("lo" in c and "hi" in c and "name" in c for c in constraints)

    def test_get_preset_invalid(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_all_presets_valid_constraints(self):
        """Every preset has valid constraints with lo <= hi."""
        for name in list_presets():
            constraints = get_preset(name)
            assert len(constraints) > 0, f"Preset {name} has no constraints"
            assert len(constraints) <= 8, f"Preset {name} has > 8 constraints"
            for c in constraints:
                assert "lo" in c, f"Preset {name}: missing 'lo' in {c}"
                assert "hi" in c, f"Preset {name}: missing 'hi' in {c}"
                assert "name" in c, f"Preset {name}: missing 'name' in {c}"
                assert c["lo"] <= c["hi"], f"Preset {name}: lo > hi for {c['name']}"

    def test_all_names_unique_per_preset(self):
        """Constraint names within a preset are unique."""
        for name in list_presets():
            constraints = get_preset(name)
            names = [c["name"] for c in constraints]
            assert len(names) == len(set(names)), f"Preset {name} has duplicate names"

    def test_description_exists(self):
        for name in list_presets():
            desc = get_preset_description(name)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_preset_with_flux_exact(self):
        """Every preset can be loaded into FluxExact."""
        from flux_check.core import FluxExact
        for name in list_presets():
            fc = FluxExact.from_preset(name)
            assert fc.n == len(get_preset(name))
