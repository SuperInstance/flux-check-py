"""
FLUX Fracture-Coalesce — Disjoint Linear Algebra for Constraint Systems.

Fractures constraint systems into independent blocks via BFS on the
constraint-dimension dependency graph, then coalesces results provably
correct via bitwise OR.

THEOREM: If fracture correctly identifies connected components, coalescence
via bitwise OR preserves zero false negatives.

Adapted from flux_fracture.py for the flux-check CLI package.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ── DependencyGraph ─────────────────────────────────────────

@dataclass
class DependencyGraph:
    """Bipartite graph: constraints (rows) × dimensions (columns)."""
    adjacency: NDArray[np.uint8]
    n_constraints: int
    n_dimensions: int
    constraint_names: List[str] = field(default_factory=list)
    dimension_names: List[str] = field(default_factory=list)

    @classmethod
    def from_masks(cls,
                   masks: Sequence[NDArray[np.integer]],
                   constraint_names: Sequence[str] = (),
                   dimension_names: Sequence[str] = ()) -> "DependencyGraph":
        n_c = len(masks)
        n_d = max((m.max() for m in masks), default=0) + 1 if masks else 0
        adj = np.zeros((n_c, n_d), dtype=np.uint8)
        for i, m in enumerate(masks):
            adj[i, m] = 1
        return cls(
            adjacency=adj, n_constraints=n_c, n_dimensions=n_d,
            constraint_names=list(constraint_names) or [f"c{i}" for i in range(n_c)],
            dimension_names=list(dimension_names) or [f"d{j}" for j in range(n_d)],
        )

    @classmethod
    def from_adjacency(cls, adj: NDArray[np.uint8],
                       constraint_names: Sequence[str] = (),
                       dimension_names: Sequence[str] = ()) -> "DependencyGraph":
        n_c, n_d = adj.shape
        return cls(
            adjacency=adj.astype(np.uint8), n_constraints=n_c, n_dimensions=n_d,
            constraint_names=list(constraint_names) or [f"c{i}" for i in range(n_c)],
            dimension_names=list(dimension_names) or [f"d{j}" for j in range(n_d)],
        )

    def involves(self, constraint_idx: int, dimension_idx: int) -> bool:
        return bool(self.adjacency[constraint_idx, dimension_idx])

    def constraint_dims(self, constraint_idx: int) -> NDArray[np.intp]:
        return np.flatnonzero(self.adjacency[constraint_idx])

    def dim_constraints(self, dimension_idx: int) -> NDArray[np.intp]:
        return np.flatnonzero(self.adjacency[:, dimension_idx])


# ── Block & FractureResult ──────────────────────────────────

@dataclass
class Block:
    """One independent block of the fractured system."""
    constraint_indices: List[int]
    dimension_indices: List[int]
    size: int = 0

    def __post_init__(self):
        self.size = len(self.constraint_indices)


@dataclass
class FractureResult:
    """Result of fracturing a constraint system into independent blocks."""
    blocks: List[Block]
    graph: DependencyGraph
    n_blocks: int = 0
    largest_block_size: int = 0
    speedup_potential: float = 1.0

    def __post_init__(self):
        self.n_blocks = len(self.blocks)
        self.largest_block_size = max((b.size for b in self.blocks), default=0)
        n_c = self.graph.n_constraints
        self.speedup_potential = n_c / self.largest_block_size if self.largest_block_size > 0 else 1.0

    def summary(self) -> Dict:
        return {
            "n_blocks": self.n_blocks,
            "largest_block_size": self.largest_block_size,
            "speedup_potential": round(self.speedup_potential, 2),
            "block_sizes": [b.size for b in self.blocks],
            "n_constraints": self.graph.n_constraints,
            "n_dimensions": self.graph.n_dimensions,
        }


# ── Fracturer ───────────────────────────────────────────────

class Fracturer:
    """
    Fractures a constraint system by finding connected components
    of the constraint-dimension bipartite dependency graph via BFS.
    """

    def fracture(self, graph: DependencyGraph) -> FractureResult:
        visited_c = np.zeros(graph.n_constraints, dtype=bool)
        visited_d = np.zeros(graph.n_dimensions, dtype=bool)
        blocks: List[Block] = []

        for seed_c in range(graph.n_constraints):
            if visited_c[seed_c]:
                continue
            comp_c: Set[int] = set()
            comp_d: Set[int] = set()
            queue: deque = deque()
            queue.append(("c", seed_c))
            while queue:
                node_type, idx = queue.popleft()
                if node_type == "c":
                    if visited_c[idx]:
                        continue
                    visited_c[idx] = True
                    comp_c.add(idx)
                    for d in np.flatnonzero(graph.adjacency[idx]):
                        if not visited_d[d]:
                            queue.append(("d", d))
                else:
                    if visited_d[idx]:
                        continue
                    visited_d[idx] = True
                    comp_d.add(idx)
                    for c in np.flatnonzero(graph.adjacency[:, idx]):
                        if not visited_c[c]:
                            queue.append(("c", c))

            blocks.append(Block(
                constraint_indices=sorted(comp_c),
                dimension_indices=sorted(comp_d),
            ))

        for d in range(graph.n_dimensions):
            if not visited_d[d]:
                blocks.append(Block(constraint_indices=[], dimension_indices=[d]))

        return FractureResult(blocks=blocks, graph=graph)

    def fracture_from_bounds(self, constraints: List[Dict]) -> FractureResult:
        masks = []
        for i, c in enumerate(constraints):
            if "dims" in c:
                masks.append(np.array(c["dims"], dtype=np.intp))
            else:
                masks.append(np.array([i], dtype=np.intp))
        graph = DependencyGraph.from_masks(masks)
        return self.fracture(graph)


# ── Coalescer ───────────────────────────────────────────────

class Coalescer:
    """
    Coalesces block-level error masks into a unified error mask via bitwise OR.

    CORRECTNESS: Since blocks are independent (no shared dimensions),
    E = E_1 | E_2 | ... | E_k captures ALL violations with zero false negatives.
    """

    def coalesce_masks(self, block_masks: List[int], n_total_constraints: int = 0) -> int:
        result = 0
        for m in block_masks:
            result |= m
        return result

    def coalesce_arrays(self, block_arrays: List[NDArray[np.uint8]]) -> NDArray[np.uint8]:
        if not block_arrays:
            return np.array([], dtype=np.uint8)
        result = np.zeros_like(block_arrays[0])
        for arr in block_arrays:
            result |= arr
        return result

    def verify_coalescence(self,
                           block_masks: List[int],
                           block_constraint_indices: List[List[int]],
                           monolithic_mask: int) -> Tuple[bool, str]:
        coalesced = self.coalesce_masks(block_masks)
        reconstructed = 0
        for mask, indices in zip(block_masks, block_constraint_indices):
            for bit_pos, c_idx in enumerate(indices):
                if mask & (1 << bit_pos):
                    reconstructed |= (1 << c_idx)
        if reconstructed == monolithic_mask:
            return True, f"PERFECT MATCH: coalesced={monolithic_mask:#x}"
        else:
            false_negatives = monolithic_mask & ~reconstructed
            false_positives = reconstructed & ~monolithic_mask
            return False, (
                f"MISMATCH: reconstructed={reconstructed:#x} vs monolithic={monolithic_mask:#x}. "
                f"false_neg={false_negatives:#x} false_pos={false_positives:#x}"
            )
