"""Tests for flux_check.fracture — BFS fracture and coalesce."""

import numpy as np
import pytest

from flux_check.fracture import (
    DependencyGraph, Block, FractureResult,
    Fracturer, Coalescer,
)


class TestDependencyGraph:

    def test_from_masks(self):
        masks = [np.array([0, 1]), np.array([1, 2])]
        g = DependencyGraph.from_masks(masks, ["c0", "c1"], ["d0", "d1", "d2"])
        assert g.n_constraints == 2
        assert g.n_dimensions == 3
        assert g.involves(0, 0)
        assert g.involves(0, 1)
        assert not g.involves(0, 2)

    def test_from_adjacency(self):
        adj = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        g = DependencyGraph.from_adjacency(adj)
        assert g.n_constraints == 2
        assert g.n_dimensions == 2

    def test_constraint_dims(self):
        masks = [np.array([0, 2])]
        g = DependencyGraph.from_masks(masks)
        dims = g.constraint_dims(0)
        assert set(dims) == {0, 2}


class TestFracturer:

    def test_single_block(self):
        """All constraints share a dimension → one block."""
        masks = [np.array([0]), np.array([0])]
        g = DependencyGraph.from_masks(masks, ["c0", "c1"])
        f = Fracturer()
        result = f.fracture(g)
        assert result.n_blocks == 1
        assert result.largest_block_size == 2

    def test_two_independent_blocks(self):
        """Two pairs sharing nothing → two blocks."""
        masks = [
            np.array([0, 1]),
            np.array([0]),
            np.array([2]),
            np.array([2]),
        ]
        g = DependencyGraph.from_masks(masks, ["c0", "c1", "c2", "c3"])
        f = Fracturer()
        result = f.fracture(g)
        assert result.n_blocks == 2
        assert result.speedup_potential == 2.0

    def test_all_independent(self):
        """Each constraint has its own dimension → N blocks."""
        masks = [np.array([i]) for i in range(4)]
        g = DependencyGraph.from_masks(masks)
        f = Fracturer()
        result = f.fracture(g)
        assert result.n_blocks == 4

    def test_fracture_from_bounds(self):
        constraints = [
            {"lo": 0, "hi": 10, "dims": [0, 1]},
            {"lo": 0, "hi": 20, "dims": [1, 2]},
            {"lo": 0, "hi": 30, "dims": [3]},
        ]
        f = Fracturer()
        result = f.fracture_from_bounds(constraints)
        assert result.n_blocks == 2  # {0,1} and {2}

    def test_summary(self):
        masks = [np.array([0]), np.array([1])]
        g = DependencyGraph.from_masks(masks)
        result = Fracturer().fracture(g)
        s = result.summary()
        assert "n_blocks" in s
        assert "speedup_potential" in s


class TestCoalescer:

    def test_coalesce_masks(self):
        c = Coalescer()
        result = c.coalesce_masks([0b01, 0b10])
        assert result == 0b11

    def test_coalesce_empty(self):
        c = Coalescer()
        assert c.coalesce_masks([]) == 0

    def test_coalesce_arrays(self):
        c = Coalescer()
        a = np.array([0b01, 0b00], dtype=np.uint8)
        b = np.array([0b10, 0b01], dtype=np.uint8)
        result = c.coalesce_arrays([a, b])
        np.testing.assert_array_equal(result, [0b11, 0b01])

    def test_verify_correct(self):
        c = Coalescer()
        # Block 0 has constraint 0, block mask 0b01 → bit 0 set
        # Block 1 has constraint 1, block mask 0b01 → bit 0 set (local)
        ok, msg = c.verify_coalescence(
            block_masks=[0b01, 0b01],
            block_constraint_indices=[[0], [1]],
            monolithic_mask=0b11,
        )
        assert ok

    def test_verify_incorrect(self):
        c = Coalescer()
        ok, msg = c.verify_coalescence(
            block_masks=[0b01],
            block_constraint_indices=[[0]],
            monolithic_mask=0b11,  # bit 1 missing
        )
        assert not ok
