#!/usr/bin/env python3
"""Tree shape builders: full binary tree and caterpillar tree.

Provides configurable tree topology generation and property assignment (g, p, distribution)
for different tree shapes, decoupled from CLI entry points.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TreeStructure:
    """Output of tree builder: topology and metadata."""

    parents: List[int]
    """Parent of each non-root node (length = node_counts - 1)."""

    leaves: List[int]
    """Leaf node indices in DFS left-to-right order."""

    node_counts: int
    """Total number of nodes."""

    depth_map: List[int]
    """Depth of each node (root is depth 0)."""

    max_depth: int
    """Maximum depth in the tree."""


class FullBinaryTreeBuilder:
    """Generate a full S-ary tree of depth K."""

    def __init__(self, s: int, k: int) -> None:
        if s < 2:
            raise ValueError("S must be >= 2")
        if k < 1:
            raise ValueError("K must be >= 1")
        self.s = s
        self.k = k

    def build(self) -> TreeStructure:
        """Build full S-ary tree with depth K.

        Returns:
            TreeStructure with parents, leaves (DFS left-to-right order), node_counts, depth_map.
        """
        s, k = self.s, self.k

        # Compute total node count: sum of S^i for i=0..K
        node_counts = (s ** (k + 1) - 1) // (s - 1)

        # Build parent array
        parents: List[int] = []
        for node in range(1, node_counts):
            parent = (node - 1) // s
            parents.append(parent)

        # Compute depth for each node
        depth_map = [0] * node_counts
        for node in range(1, node_counts):
            par = parents[node - 1]
            depth_map[node] = depth_map[par] + 1

        # Collect leaves in DFS left-to-right order
        first_leaf = (s**k - 1) // (s - 1)  # First node at depth K
        leaves = list(range(first_leaf, node_counts))

        return TreeStructure(
            parents=parents,
            leaves=leaves,
            node_counts=node_counts,
            depth_map=depth_map,
            max_depth=k,
        )


class CaterpillarTreeBuilder:
    """Generate a caterpillar tree of depth K.

    Structure: A "backbone" that goes left K times, with a right leaf at each level.
        - Root at depth 0 has left child at depth 1 and right leaf at depth 1.
        - Each backbone node at depth d < K has a left child that continues the backbone
            and a right leaf at depth d+1.
        - The deepest backbone node at depth K is a leaf.

    Example (K=3):
        Node 0 (depth 0)
        ├─ Node 1 (depth 1, interior)
        │  ├─ Node 2 (depth 2, interior)
        │  │  ├─ Node 3 (depth 3, leaf)
        │  │  └─ Node 4 (depth 3, leaf)
        │  └─ Node 5 (depth 2, leaf)
        └─ Node 6 (depth 1, leaf)

    Total leaves: 3 + 1 = 4 (nodes 3, 4, 5, 6)
    Total nodes: 2*K + 1
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("K must be >= 1")
        self.k = k

    def build(self) -> TreeStructure:
        """Build caterpillar tree of depth K using iterative approach.

        Returns:
            TreeStructure with parents, leaves, node_counts, depth_map.
        """
        k = self.k
        node_counts = 2 * k + 1
        parents: List[int] = []
        depth_map: List[int] = [0]  # Root at depth 0

        # Build a true backbone: each left child becomes the next interior node.
        current_backbone_node = 0
        current_node_id = 1
        for backbone_depth in range(k):
            next_backbone_node = current_node_id
            parents.append(current_backbone_node)
            depth_map.append(backbone_depth + 1)
            current_node_id += 1

            parents.append(current_backbone_node)
            depth_map.append(backbone_depth + 1)
            current_node_id += 1

            current_backbone_node = next_backbone_node

        # Collect leaves: nodes with no children
        children_count = [0] * node_counts
        for p in parents:
            children_count[p] += 1

        leaves = [i for i in range(node_counts) if children_count[i] == 0 and i != 0]

        return TreeStructure(
            parents=parents,
            leaves=leaves,
            node_counts=node_counts,
            depth_map=depth_map,
            max_depth=k,
        )


def assign_g_values_full_binary(
    tree: TreeStructure, ratio: float, rng: random.Random
) -> List[int]:
    """Assign g values for full binary tree.

    Strategy:
    - Pick first ceil(ratio * |leaves|) leaves in order.
    - Mark these leaves and all their ancestors as g=1.
    - Mark all other non-root nodes as g=0.

    Args:
        tree: TreeStructure from FullBinaryTreeBuilder.
        ratio: Fraction of leaves to mark as g=1 (in [0,1]).
        rng: Random number generator (unused for deterministic prefix selection).

    Returns:
        g array (length = node_counts - 1, indexed as g[node-1]).
    """
    node_counts = tree.node_counts
    num_special_leaves = max(1, int((len(tree.leaves) * ratio) + 0.5))  # ceil(ratio * |leaves|)
    special_leaves_set = set(tree.leaves[:num_special_leaves])

    g = [0] * (node_counts - 1)

    # Mark special leaves and their ancestors iteratively
    visited = set()
    stack = list(special_leaves_set)

    while stack:
        node = stack.pop()
        if node == 0 or node in visited:
            continue
        visited.add(node)
        if node >= 1:
            g[node - 1] = 1
            # Find parent and add to stack
            if node - 1 < len(tree.parents):
                parent = tree.parents[node - 1]
                if parent not in visited:
                    stack.append(parent)

    return g


def assign_g_values_caterpillar(tree: TreeStructure, r: int) -> List[int]:
    """Assign g values for caterpillar tree.

    Strategy:
    - Nodes with depth <= R are g=0.
    - All other nodes are g=1.

    Args:
        tree: TreeStructure from CaterpillarTreeBuilder.
        r: First R layers (from root depth=0) have g=0.

    Returns:
        g array (length = node_counts - 1).
    """
    node_counts = tree.node_counts
    g = [0] * (node_counts - 1)

    # Nodes with depth > R are g=1
    for node in range(1, node_counts):
        if tree.depth_map[node] > r:
            g[node - 1] = 1

    return g


def assign_g_values_mix_caterpillar(
    tree: TreeStructure, ratio: float, rng: random.Random
) -> List[int]:
    """Assign g values for mixcaterpillar tree.

    Strategy:
    - Each non-root node independently samples g=1 with probability `ratio`.
    - Otherwise g=0.

    Args:
        tree: TreeStructure from CaterpillarTreeBuilder.
        ratio: Probability of g=1 for each non-root node, in [0, 1].
        rng: Random number generator.

    Returns:
        g array (length = node_counts - 1).
    """
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("ratio must be in [0, 1]")

    node_counts = tree.node_counts
    g = [0] * (node_counts - 1)

    for node in range(1, node_counts):
        if rng.random() < ratio:
            g[node - 1] = 1

    return g


def assign_p_values_full_binary(tree: TreeStructure, rng: random.Random) -> List[float]:
    """Assign p values for full binary tree leaves.

    All leaves sample from [0.2, 0.8] uniformly.

    Args:
        tree: TreeStructure with leaves list.
        rng: Random number generator.

    Returns:
        p array (length = node_counts - 1, indexed as p[node-1] for non-root nodes).
    """
    node_counts = tree.node_counts
    p = [0.0] * (node_counts - 1)

    for leaf in tree.leaves:
        if leaf >= 1:
            p[leaf - 1] = round(rng.uniform(0.2, 0.8), 6)

    return p


def assign_p_values_caterpillar(tree: TreeStructure, rng: random.Random) -> List[float]:
    """Assign p values for caterpillar tree leaves.

    Strategy:
    - Leaves are sampled from depth-dependent intervals.
    - Deeper leaves sample from higher [p_min, p_max] values.
    - Linear interpolation: at depth d, sample from [0.2 + (d/max_d)*0.6, ...]

    Args:
        tree: TreeStructure with leaves and depth_map.
        rng: Random number generator.

    Returns:
        p array (length = node_counts - 1).
    """
    node_counts = tree.node_counts
    p = [0.0] * (node_counts - 1)

    max_depth = tree.max_depth

    for leaf in tree.leaves:
        if leaf >= 1:
            depth = tree.depth_map[leaf]
            # Interpolate sampling interval: as depth increases from 0 to max_depth,
            # min increases from 0.2 to 0.8, max stays at 0.8
            # Or: min=0.2, max goes from 0.8 to something higher
            # Better: as depth increases, interval shifts right: [0.2 + k*d, 0.8 + k*d]
            # Constrain to [0, 1]: safer to keep interval in [0.2, 0.8] but shift by depth ratio
            if max_depth > 0:
                depth_ratio = depth / max_depth
                p_min = 0.2 + depth_ratio * 0.6  # From 0.2 to 0.8 as depth goes 0 to max
                p_max = 0.8
            else:
                p_min, p_max = 0.2, 0.8

            p[leaf - 1] = round(rng.uniform(p_min, min(p_max, 1.0)), 6)

    return p


def assign_distribution(
    tree: TreeStructure, special_leaf_idx: int, rng: random.Random
) -> dict[int, str]:
    """Assign distribution TimeVariant to a special leaf.

    Args:
        tree: TreeStructure with leaves list.
        special_leaf_idx: Index into leaves list (0-indexed).
        rng: Random number generator (unused).

    Returns:
        Dictionary mapping leaf node ID to "TIMEVARIANT" string.
    """
    if special_leaf_idx < 0 or special_leaf_idx >= len(tree.leaves):
        raise ValueError(f"special_leaf_idx {special_leaf_idx} out of range [0, {len(tree.leaves)})")

    special_leaf = tree.leaves[special_leaf_idx]
    return {special_leaf: "TIMEVARIANT"}


def select_random_leaf(leaves: List[int], rng: random.Random) -> int:
    """Randomly select a leaf node ID from the leaves list."""
    if not leaves:
        raise ValueError("No leaves available")
    return rng.choice(leaves)
