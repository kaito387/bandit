"""JSON-driven multistage bandit simulator (Python + Numba).

Input: JSON array of environments.
Output per environment:
1) CSV time-series for all selected algorithms.
2) JSON summary with structural metrics and final stats.
3) PNG plot for avgRegret[t] vs t.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
from numba import njit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12
NUM_AVERAGE_RUNS = 20

DYNAMIC_TABLE_COLUMNS = [
    "algo_name",
    "env_name",
    "avgCost",
    "bestPathRate",
    "shareRate",
    "finalAvgRegret",
    "finalAccumRegret",
]

ALGO_CODE_TO_ID = {
    "PS": 1,
    "E3": 2,
    "EE3": 3,
    "R": 4,
    "Q": 5,
    "E3Q": 6,
}

ALGO_ID_TO_NAME = {
    1: "PS",
    2: "E3",
    3: "EE3",
    4: "R",
    5: "Q",
    6: "E3Q",
}

DIST_BERNOULLI = 0
DIST_TIME_VARIANT = 1

DIST_NAME_TO_CODE = {
    "BERNOULLI": DIST_BERNOULLI,
    "TIMEVARIANT": DIST_TIME_VARIANT,
}


@dataclass
class EnvironmentInput:
    env_name: str
    algo: List[str]
    seed: int
    node_counts: int
    rounds: int
    parents: List[int]
    g: List[int]
    p: List[float]
    distribution: Dict[int, int]
    distribution_raw: Dict[str, str] = None  # Original string format from JSON


@dataclass
class PreparedEnvironment:
    env_name: str
    algo_ids: np.ndarray
    seed: int
    n: int
    rounds: int
    parent: np.ndarray
    child_start: np.ndarray
    child_count: np.ndarray
    child_list: np.ndarray
    local_index_in_parent: np.ndarray
    is_share: np.ndarray
    leaf_prob: np.ndarray
    leaf_distribution: np.ndarray
    is_leaf: np.ndarray
    is_safe: np.ndarray
    depth_value: int
    d: np.ndarray
    max_branching: int
    leaf_count: np.ndarray
    subtree_leaf_start: np.ndarray
    subtree_leaf_count: np.ndarray
    subtree_leaves: np.ndarray
    risk: np.ndarray
    r0: int
    need_explore: np.ndarray
    leaves: np.ndarray
    best_leaf: int
    best_leaf_p: float
    # Original configuration from JSON for reference
    raw_parents: List[int]
    raw_g: List[int]
    raw_p: List[float]
    raw_distribution: Dict[str, str]
    raw_algo: List[str]


def _build_tree_arrays(n: int, parents: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = np.full(n, -1, dtype=np.int64)
    parent[0] = -1
    for node in range(1, n):
        parent[node] = int(parents[node - 1])

    child_count = np.zeros(n, dtype=np.int64)
    for node in range(1, n):
        child_count[parent[node]] += 1

    child_start = np.zeros(n, dtype=np.int64)
    offset = 0
    for node in range(n):
        child_start[node] = offset
        offset += child_count[node]

    child_list = np.empty(offset, dtype=np.int64)
    used = np.zeros(n, dtype=np.int64)
    local_index_in_parent = np.zeros(n, dtype=np.int64)
    for node in range(1, n):
        pnode = parent[node]
        slot = child_start[pnode] + used[pnode]
        child_list[slot] = node
        local_index_in_parent[node] = used[pnode]
        used[pnode] += 1

    return parent, child_start, child_count, child_list, local_index_in_parent


def _compute_depth(parent: np.ndarray) -> Tuple[np.ndarray, int]:
    n = parent.shape[0]
    depth = np.zeros(n, dtype=np.int64)
    max_depth = 0
    for node in range(1, n):
        d = 1
        cur = int(parent[node])
        while cur > 0:
            d += 1
            cur = int(parent[cur])
        depth[node] = d
        if d > max_depth:
            max_depth = d
    return depth, int(max_depth)


def _postorder(parent: np.ndarray) -> np.ndarray:
    n = parent.shape[0]
    children = [[] for _ in range(n)]
    for node in range(1, n):
        children[int(parent[node])].append(node)
    order: List[int] = []

    def dfs(u: int) -> None:
        for v in children[u]:
            dfs(v)
        order.append(u)

    dfs(0)
    return np.array(order, dtype=np.int64)


def _validate_env_obj(obj: dict, index: int) -> EnvironmentInput:
    required = ["env_name", "algo", "seed", "node_counts", "rounds", "parents", "g", "p"]
    for key in required:
        if key not in obj:
            raise ValueError(f"env[{index}] missing field: {key}")

    env_name = str(obj["env_name"])
    algo = list(obj["algo"])
    seed = int(obj["seed"])
    node_counts = int(obj["node_counts"])
    rounds = int(obj["rounds"])
    parents = [int(x) for x in obj["parents"]]
    g = [int(x) for x in obj["g"]]
    p = [float(x) for x in obj["p"]]
    distribution_raw = obj.get("distribution", {})

    if node_counts <= 0:
        raise ValueError(f"env[{index}] node_counts must be positive")
    if rounds <= 0:
        raise ValueError(f"env[{index}] rounds must be positive")

    expected = node_counts - 1
    if len(parents) != expected or len(g) != expected or len(p) != expected:
        raise ValueError(
            f"env[{index}] expected parents/g/p length {expected}, got {len(parents)}/{len(g)}/{len(p)}"
        )

    for node in range(1, node_counts):
        par = parents[node - 1]
        if par < 0 or par >= node_counts:
            raise ValueError(f"env[{index}] invalid parent index for node {node}: {par}")
        if par == node:
            raise ValueError(f"env[{index}] self-parent not allowed at node {node}")

    for node in range(1, node_counts):
        seen = set([node])
        cur = parents[node - 1]
        while cur != 0:
            if cur in seen:
                raise ValueError(f"env[{index}] cycle detected on node {node}")
            seen.add(cur)
            cur = parents[cur - 1]

    for val in g:
        if val not in (0, 1):
            raise ValueError(f"env[{index}] g values must be 0 or 1")

    for val in p:
        if val < 0.0 or val > 1.0:
            raise ValueError(f"env[{index}] p values must be in [0,1]")

    if not isinstance(distribution_raw, dict):
        raise ValueError(f"env[{index}] distribution must be an object")

    distribution: Dict[int, int] = {}
    for key, value in distribution_raw.items():
        try:
            node = int(key)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"env[{index}] distribution key must be node index, got {key}") from exc

        if node <= 0 or node >= node_counts:
            raise ValueError(f"env[{index}] distribution node out of range: {node}")

        if not isinstance(value, str):
            raise ValueError(f"env[{index}] distribution value for node {node} must be string")

        norm = value.strip().upper()
        if norm not in DIST_NAME_TO_CODE:
            allowed = ", ".join(sorted(DIST_NAME_TO_CODE.keys()))
            raise ValueError(f"env[{index}] unknown distribution '{value}' for node {node}; allowed: {allowed}")

        distribution[node] = DIST_NAME_TO_CODE[norm]

    if not algo:
        raise ValueError(f"env[{index}] algo must not be empty")
    for code in algo:
        if code not in ALGO_CODE_TO_ID:
            raise ValueError(f"env[{index}] unknown algo code: {code}")

    return EnvironmentInput(
        env_name=env_name,
        algo=algo,
        seed=seed,
        node_counts=node_counts,
        rounds=rounds,
        parents=parents,
        g=g,
        p=p,
        distribution=distribution,
        distribution_raw={key: str(value) for key, value in distribution_raw.items()},
    )


def _prepare_environment(raw: EnvironmentInput) -> PreparedEnvironment:
    n = raw.node_counts
    parent, child_start, child_count, child_list, local_index_in_parent = _build_tree_arrays(n, raw.parents)

    is_share = np.zeros(n, dtype=np.int64)
    is_share[0] = 1
    for node in range(1, n):
        is_share[node] = int(raw.g[node - 1])

    leaf_prob = np.zeros(n, dtype=np.float64)
    leaf_distribution = np.full(n, DIST_BERNOULLI, dtype=np.int64)
    for node in range(1, n):
        leaf_prob[node] = float(raw.p[node - 1])

    is_leaf = (child_count == 0).astype(np.int64)
    depth_arr, depth_value = _compute_depth(parent)
    d = child_count.copy()
    max_branching = int(np.max(d)) if n > 0 else 0

    order = _postorder(parent)
    leaf_count = np.zeros(n, dtype=np.int64)
    is_safe = np.zeros(n, dtype=np.int64)
    risk = np.zeros(n, dtype=np.int64)
    need_explore = np.zeros(n, dtype=np.int64)

    for u in order:
        if is_leaf[u] == 1:
            leaf_count[u] = 1
            is_safe[u] = 1
            risk[u] = 0
            need_explore[u] = 0
            continue

        total_leaf = 0
        safe = 1
        max_risk_child = 0
        explore_needed = 0
        start = child_start[u]
        cnt = child_count[u]
        for i in range(cnt):
            v = child_list[start + i]
            total_leaf += leaf_count[v]
            if not (is_safe[v] == 1 and is_share[v] == 1):
                safe = 0
            if risk[v] > max_risk_child:
                max_risk_child = int(risk[v])
            if is_leaf[v] == 0:
                explore_needed = 1

        leaf_count[u] = total_leaf
        is_safe[u] = safe
        risk[u] = (1 - safe) + max_risk_child
        need_explore[u] = explore_needed

    # Precompute flattened leaf lists for each subtree to support stable on-demand W[v]/W[u].
    children: List[List[int]] = [[] for _ in range(n)]
    for node in range(1, n):
        children[int(parent[node])].append(node)

    subtree_leaf_nodes: List[List[int]] = [[] for _ in range(n)]

    def collect_subtree_leaves(u: int) -> List[int]:
        if is_leaf[u] == 1:
            subtree_leaf_nodes[u] = [u]
            return subtree_leaf_nodes[u]

        merged: List[int] = []
        for v in children[u]:
            merged.extend(collect_subtree_leaves(v))
        subtree_leaf_nodes[u] = merged
        return merged

    collect_subtree_leaves(0)

    subtree_leaf_start = np.zeros(n, dtype=np.int64)
    subtree_leaf_count = np.zeros(n, dtype=np.int64)
    total_entries = sum(len(x) for x in subtree_leaf_nodes)
    subtree_leaves = np.empty(total_entries, dtype=np.int64)
    offset = 0
    for u in range(n):
        leaves_u = subtree_leaf_nodes[u]
        cnt_u = len(leaves_u)
        subtree_leaf_start[u] = offset
        subtree_leaf_count[u] = cnt_u
        for i, leaf in enumerate(leaves_u):
            subtree_leaves[offset + i] = int(leaf)
        offset += cnt_u

    leaves = np.where(is_leaf == 1)[0].astype(np.int64)
    if leaves.shape[0] == 0:
        raise ValueError(f"env {raw.env_name} has no leaves")

    for leaf in leaves:
        if leaf != 0:
            continue
        raise ValueError(f"env {raw.env_name} root cannot be leaf (need at least one edge)")

    for node, dist_code in raw.distribution.items():
        if is_leaf[node] != 1:
            raise ValueError(f"env {raw.env_name} distribution can only be set on leaves, got node {node}")
        leaf_distribution[node] = int(dist_code)

    best_leaf = int(leaves[0])
    best_p = float(leaf_prob[best_leaf])
    for leaf in leaves:
        pval = float(leaf_prob[leaf])
        if pval < best_p - 1e-15 or (abs(pval - best_p) <= 1e-15 and int(leaf) < best_leaf):
            best_leaf = int(leaf)
            best_p = pval

    algo_ids: List[int] = []
    seen = set()
    for code in raw.algo:
        aid = ALGO_CODE_TO_ID[code]
        if aid not in seen:
            seen.add(aid)
            algo_ids.append(aid)

    return PreparedEnvironment(
        env_name=raw.env_name,
        algo_ids=np.array(algo_ids, dtype=np.int64),
        seed=raw.seed,
        n=n,
        rounds=raw.rounds,
        parent=parent,
        child_start=child_start,
        child_count=child_count,
        child_list=child_list,
        local_index_in_parent=local_index_in_parent,
        is_share=is_share,
        leaf_prob=leaf_prob,
        leaf_distribution=leaf_distribution,
        is_leaf=is_leaf,
        is_safe=is_safe,
        depth_value=depth_value,
        d=d,
        max_branching=max_branching,
        leaf_count=leaf_count,
        subtree_leaf_start=subtree_leaf_start,
        subtree_leaf_count=subtree_leaf_count,
        subtree_leaves=subtree_leaves,
        risk=risk,
        r0=int(risk[0]),
        need_explore=need_explore,
        leaves=leaves,
        best_leaf=best_leaf,
        best_leaf_p=best_p,
        raw_parents=raw.parents,
        raw_g=raw.g,
        raw_p=raw.p,
        raw_distribution=raw.distribution_raw or {},
        raw_algo=raw.algo,
    )


@njit(cache=True)
def _softmax_child(theta: np.ndarray, children: np.ndarray, start: int, count: int, eta: float, out_probs: np.ndarray) -> None:
    max_v = eta * theta[children[start]]
    for i in range(1, count):
        v = eta * theta[children[start + i]]
        if v > max_v:
            max_v = v
    s = 0.0
    for i in range(count):
        x = math.exp(eta * theta[children[start + i]] - max_v)
        out_probs[i] = x
        s += x
    if s <= EPS:
        inv = 1.0 / count
        for i in range(count):
            out_probs[i] = inv
    else:
        for i in range(count):
            out_probs[i] /= s


@njit(cache=True)
def _sample_discrete(probs: np.ndarray, count: int) -> int:
    r = np.random.random()
    c = 0.0
    for i in range(count - 1):
        c += probs[i]
        if r < c:
            return i
    return count - 1

@njit(cache=True)
def _sample_bernoulli(p: float) -> float:
    return 1.0 if np.random.random() < p else 0.0

@njit(cache=True)
def _sample_timevariant(p: float, t_ratio: float) -> float:
    return 1.0 if t_ratio < p else 0.0


@njit(cache=True)
def _sample_leaf_cost(dist_code: int, p: float, t: int, rounds: int) -> float:
    if dist_code == DIST_TIME_VARIANT:
        t_ratio = float(t + 1) / float(max(rounds, 1))
        return _sample_timevariant(p, t_ratio)
    return _sample_bernoulli(p)


@njit(cache=True)
def _logsumexp_subtree(
    node: int,
    eta: float,
    theta: np.ndarray,
    subtree_leaf_start: np.ndarray,
    subtree_leaf_count: np.ndarray,
    subtree_leaves: np.ndarray,
) -> float:
    start = subtree_leaf_start[node]
    cnt = subtree_leaf_count[node]

    first_leaf = subtree_leaves[start]
    max_theta = eta * float(theta[int(first_leaf)])
    for i in range(1, cnt):
        leaf = subtree_leaves[start + i]
        val = eta * float(theta[int(leaf)])
        if val > max_theta:
            max_theta = val

    s = 0.0
    for i in range(cnt):
        leaf = subtree_leaves[start + i]
        s += math.exp(eta * float(theta[int(leaf)]) - max_theta)
    if s <= EPS:
        s = EPS

    return max_theta + math.log(s)


@njit(cache=True)
def _stable_ps_safe_probs(
    node: int,
    start: int,
    cnt: int,
    child_list: np.ndarray,
    eta_ps: float,
    theta: np.ndarray,
    subtree_leaf_start: np.ndarray,
    subtree_leaf_count: np.ndarray,
    subtree_leaves: np.ndarray,
    out_probs: np.ndarray,
) -> None:
    log_w_u = _logsumexp_subtree(node, eta_ps, theta, subtree_leaf_start, subtree_leaf_count, subtree_leaves)

    total = 0.0
    for i in range(cnt):
        child = child_list[start + i]
        log_w_v = _logsumexp_subtree(child, eta_ps, theta, subtree_leaf_start, subtree_leaf_count, subtree_leaves)
        ratio = math.exp(log_w_v - log_w_u)
        if ratio < 0.0:
            ratio = 0.0
        out_probs[i] = ratio
        total += ratio

    if total <= EPS:
        raise ValueError("underflowed")
    else:
        for i in range(cnt):
            out_probs[i] /= total


@njit(cache=True)
def _compute_child_probabilities(
    algo_id: int,
    node: int,
    start: int,
    cnt: int,
    theta: np.ndarray,
    q: np.ndarray,
    child_list: np.ndarray,
    is_safe: np.ndarray,
    need_explore: np.ndarray,
    eta_ps: float,
    eta_e3: float,
    eta_ee3: float,
    eps_ps: float,
    eps_ee3: float,
    subtree_leaf_start: np.ndarray,
    subtree_leaf_count: np.ndarray,
    subtree_leaves: np.ndarray,
    out_probs: np.ndarray,
) -> None:
    if algo_id == 4:  # Random
        inv = 1.0 / cnt
        for i in range(cnt):
            out_probs[i] = inv
        return

    if algo_id == 5:  # Q-learning
        chosen_slot = 0
        best_q = q[child_list[start]]
        for i in range(1, cnt):
            cand = child_list[start + i]
            if q[cand] < best_q - 1e-15:
                best_q = q[cand]
                chosen_slot = i
        for i in range(cnt):
            out_probs[i] = 0.0
        out_probs[chosen_slot] = 1.0
        return

    if algo_id == 1:  # PS
        if is_safe[node] == 1:
            _stable_ps_safe_probs(
                node,
                start,
                cnt,
                child_list,
                eta_ps,
                theta,
                subtree_leaf_start,
                subtree_leaf_count,
                subtree_leaves,
                out_probs,
            )
            return

        real_eps = eps_ps if need_explore[node] == 1 else 0.0
        _softmax_child(theta, child_list, start, cnt, eta_ps, out_probs)
        for i in range(cnt):
            out_probs[i] = real_eps / cnt + (1.0 - real_eps) * out_probs[i]
        return

    if algo_id == 2 or algo_id == 6:  # E3 or E3Q
        _softmax_child(theta, child_list, start, cnt, eta_e3, out_probs)
        return

    _softmax_child(theta, child_list, start, cnt, eta_ee3, out_probs)
    for i in range(cnt):
        out_probs[i] = eps_ee3 / cnt + (1.0 - eps_ee3) * out_probs[i]


@njit(cache=True)
def _fill_leaf_distribution(
    algo_id: int,
    theta: np.ndarray,
    q: np.ndarray,
    child_start: np.ndarray,
    child_count: np.ndarray,
    child_list: np.ndarray,
    is_leaf: np.ndarray,
    is_safe: np.ndarray,
    need_explore: np.ndarray,
    eta_ps: float,
    eta_e3: float,
    eta_ee3: float,
    eps_ps: float,
    eps_ee3: float,
    subtree_leaf_start: np.ndarray,
    subtree_leaf_count: np.ndarray,
    subtree_leaves: np.ndarray,
    temp_probs: np.ndarray,
    node_stack: np.ndarray,
    prob_stack: np.ndarray,
    out_probs: np.ndarray,
) -> None:
    for i in range(out_probs.shape[0]):
        out_probs[i] = 0.0

    top = 0
    node_stack[0] = 0
    prob_stack[0] = 1.0

    while top >= 0:
        node = int(node_stack[top])
        prob = float(prob_stack[top])
        top -= 1

        if is_leaf[node] == 1:
            out_probs[node] = prob
            continue

        start = child_start[node]
        cnt = child_count[node]
        _compute_child_probabilities(
            algo_id,
            node,
            start,
            cnt,
            theta,
            q,
            child_list,
            is_safe,
            need_explore,
            eta_ps,
            eta_e3,
            eta_ee3,
            eps_ps,
            eps_ee3,
            subtree_leaf_start,
            subtree_leaf_count,
            subtree_leaves,
            temp_probs,
        )

        for i in range(cnt - 1, -1, -1):
            top += 1
            child = child_list[start + i]
            node_stack[top] = child
            prob_stack[top] = prob * temp_probs[i]


@njit(cache=True)
def _run_algo_numba(
    algo_id: int,
    seed: int,
    rounds: int,
    n: int,
    parent: np.ndarray,
    child_start: np.ndarray,
    child_count: np.ndarray,
    child_list: np.ndarray,
    is_leaf: np.ndarray,
    is_safe: np.ndarray,
    is_share: np.ndarray,
    leaf_prob: np.ndarray,
    leaf_distribution: np.ndarray,
    leaf_count: np.ndarray,
    subtree_leaf_start: np.ndarray,
    subtree_leaf_count: np.ndarray,
    subtree_leaves: np.ndarray,
    need_explore: np.ndarray,
    depth_value: int,
    max_branching: int,
    r0: int,
    best_leaf: int,
    best_leaf_p: float,
    track_leaf_probs: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    theta = np.zeros(n, dtype=np.float64)
    q = np.ones(n, dtype=np.float64)
    count_seen = np.zeros(n, dtype=np.int64)

    eps_ee3 = max_branching * (rounds ** (-1.0 / (depth_value + 1.0))) if depth_value >= 0 else 0.0
    eps_ps = max_branching * (rounds ** (-1.0 / (r0 + 1.0))) if r0 >= 0 else 0.0
    # NOTE in paper, ps = di * ..., here we use max_branching instead for simplicity

    eta_ps = rounds ** (-float(r0) / (r0 + 1.0)) if r0 > 0 else 1.0
    eta_e3 = math.sqrt(math.log(max(max_branching, 2)) / (rounds * max(max_branching, 1)))
    eta_ee3 = rounds ** (-float(depth_value) / (depth_value + 1.0)) if depth_value > 0 else 1.0

    cost = np.zeros(rounds, dtype=np.float64)
    leaf_used = np.zeros(rounds, dtype=np.int64)
    regret = np.zeros(rounds, dtype=np.float64)
    accum = np.zeros(rounds, dtype=np.float64)
    avg = np.zeros(rounds, dtype=np.float64)
    leaf_probs = np.zeros((rounds, n), dtype=np.float64) if track_leaf_probs else np.zeros((1, 1), dtype=np.float64)

    path_nodes = np.empty(n, dtype=np.int64)
    prob_to_node = np.ones(n, dtype=np.float64)
    local_prob = np.ones(n, dtype=np.float64)
    temp_probs = np.empty(max(max_branching, 1), dtype=np.float64)
    node_stack = np.empty(n, dtype=np.int64)
    prob_stack = np.empty(n, dtype=np.float64)

    cum_reg = 0.0

    for t in range(rounds):
        if (t + 1) % max(1, rounds // 5) == 0:
            print(f"Round {t + 1}/{rounds}...")
        if track_leaf_probs:
            _fill_leaf_distribution(
                algo_id,
                theta,
                q,
                child_start,
                child_count,
                child_list,
                is_leaf,
                is_safe,
                need_explore,
                eta_ps,
                eta_e3,
                eta_ee3,
                eps_ps,
                eps_ee3,
                subtree_leaf_start,
                subtree_leaf_count,
                subtree_leaves,
                temp_probs,
                node_stack,
                prob_stack,
                leaf_probs[t],
            )

        node = 0
        depth = 0
        prob_to_node[0] = 1.0
        local_prob[0] = 1.0

        while is_leaf[node] == 0:
            path_nodes[depth] = node
            depth += 1

            start = child_start[node]
            cnt = child_count[node]
            _compute_child_probabilities(
                algo_id,
                node,
                start,
                cnt,
                theta,
                q,
                child_list,
                is_safe,
                need_explore,
                eta_ps,
                eta_e3,
                eta_ee3,
                eps_ps,
                eps_ee3,
                subtree_leaf_start,
                subtree_leaf_count,
                subtree_leaves,
                temp_probs,
            )

            if algo_id == 4:  # Random
                chosen_slot = int(np.random.randint(0, cnt))
            else:
                chosen_slot = _sample_discrete(temp_probs, cnt)
            chosen_prob = max(temp_probs[chosen_slot], EPS)

            child = child_list[start + chosen_slot]
            prob_to_node[child] = prob_to_node[node] * chosen_prob
            local_prob[child] = chosen_prob
            node = child

        leaf = node
        p_leaf = leaf_prob[leaf]
        c = _sample_leaf_cost(int(leaf_distribution[leaf]), p_leaf, t, rounds)

        cost[t] = c
        leaf_used[t] = leaf
        regret_t = c - best_leaf_p
        regret[t] = regret_t
        cum_reg += regret_t
        accum[t] = cum_reg
        avg[t] = cum_reg / float(t + 1)

        if algo_id == 1: # PS
            cur = leaf
            while cur != -1:
                theta[cur] -= c / max(prob_to_node[cur], EPS)
                cur = parent[cur]

        elif algo_id == 2: # E3
            cur = leaf
            while cur != -1:
                theta[cur] -= c / max(local_prob[cur], EPS)
                cur = parent[cur]

        elif algo_id == 3 or algo_id == 6: # EE3 or E3Q
            cur = leaf
            while cur != -1:
                theta[cur] -= c / max(prob_to_node[cur], EPS)
                cur = parent[cur]

        elif algo_id == 5: # Q-learning
            cur = leaf
            while cur != -1:
                count_seen[cur] += 1
                q[cur] = q[cur] + (c - q[cur]) / count_seen[cur]
                cur = parent[cur]

    return cost, leaf_used, regret, accum, avg, leaf_probs


def _require_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for this workflow. Install it with: pip install wandb"
        ) from exc
    return wandb


def _log_avg_regret_plot_streaming(rows_generator_list: List[Tuple], env_name: str) -> None:
    """Memory-efficient version using generators for large datasets."""
    wandb = _require_wandb()
    algos = sorted({row[1] for row in rows_generator_list})
    plt.figure(figsize=(10, 6))
    
    for algo in algos:
        for env_name_g, algo_name, mean_cost, mean_regret, mean_accum, mean_avg, mean_best_path_rate, mean_share_rate in rows_generator_list:
            if algo_name != algo:
                continue
            
            rounds = len(mean_avg)
            max_t = rounds
            skip_until = int(max_t * 0.01)
            
            ts = np.arange(skip_until + 1, rounds + 1, dtype=np.int64)
            ys = mean_avg[skip_until:]
            
            if len(ts) > 0 and len(ys) > 0:
                plt.plot(ts, ys, label=algo, linewidth=1.5)
    
    plt.title(f"avgRegret[t] vs t ({env_name})")
    plt.xlabel("t")
    plt.ylabel("avgRegret[t]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    wandb.log({"charts/avg_regret": wandb.Image(plt.gcf())})
    plt.close()


def _log_dynamic_table_streaming(rows_generator_list: List[Tuple]) -> None:
    """Log one final summary row per algorithm."""
    wandb = _require_wandb()
    table = wandb.Table(columns=DYNAMIC_TABLE_COLUMNS)

    for algo_name, env_name, avg_cost, best_path_rate, share_rate, final_avg_regret, final_accum_regret in rows_generator_list:
        table.add_data(
            str(algo_name),
            str(env_name),
            float(avg_cost),
            float(best_path_rate),
            float(share_rate),
            float(final_avg_regret),
            float(final_accum_regret),
        )
    
    wandb.log({"tables/dynamic_metrics": table})

def _log_leaf_probabilities(
    leaf_probs: np.ndarray,
    leaves: np.ndarray,
    env_name: str,
    algo_name: str,
) -> None:
    wandb = _require_wandb()
    plt.figure(figsize=(11, 6))
    ts = np.arange(1, leaf_probs.shape[0] + 1, dtype=np.int64)
    for leaf in leaves:
        plt.plot(ts, leaf_probs[:, int(leaf)], label=f"leaf {int(leaf)}", linewidth=1.4, alpha=0.9)

    plt.title(f"Leaf selection probability over time ({env_name}, {algo_name})")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    wandb.log({f"charts/leaf_prob_{algo_name}": wandb.Image(plt.gcf())})
    plt.close()




def _log_summary_bars(env_name: str, summary_algorithms: Dict[str, Dict[str, float]]) -> None:
    wandb = _require_wandb()
    metrics = [
        ("avgCost", "avgCost", "avgCost"),
        ("bestPathRate", "bestPathRate", "bestPathRate"),
        ("shareRate", "shareRate", "shareRate"),
        ("finalAvgRegret", "finalAvgRegret", "finalAvgRegret"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.ravel()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    algorithms = list(summary_algorithms.keys())

    for idx, (metric, title, ylabel) in enumerate(metrics):
        values = np.asarray([float(summary_algorithms[algo][metric]) for algo in algorithms], dtype=np.float64)
        ax = axes_flat[idx]
        x = np.arange(len(algorithms))
        bars = ax.bar(x, values, color=colors[idx % len(colors)], width=0.68)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        y_min, y_max = ax.get_ylim()
        span = max(y_max - y_min, 1e-12)
        offset = span * 0.02
        for bar, value in zip(bars, values):
            x_txt = bar.get_x() + bar.get_width() / 2.0
            y_txt = bar.get_height()
            ax.text(x_txt, y_txt + offset, f"{value:.4g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Algorithm Summary Comparison ({env_name})", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    wandb.log({"charts/summary_bars": wandb.Image(fig)})
    plt.close(fig)


def _init_wandb_run(args: argparse.Namespace, env: PreparedEnvironment, job_type: str) -> Any:
    wandb = _require_wandb()
    testcase_path = getattr(args, "testcase_path", None)
    sweep_group = getattr(args, "sweep_group", None)
    run_name = getattr(args, "wandb_name", env.env_name)

    config = {
        "env_name": env.env_name,
        "seed": int(env.seed),
        "rounds": int(env.rounds),
        "algorithms": [ALGO_ID_TO_NAME[int(x)] for x in env.algo_ids],
        "num_average_runs": int(NUM_AVERAGE_RUNS),
        "testcase_path": str(testcase_path) if testcase_path else None,
        "sweep_group": str(sweep_group) if sweep_group else None,
    }
    return wandb.init(
        group=sweep_group or args.wandb_group,
        job_type=job_type,
        name=run_name,
        mode=args.wandb_mode,
        config=config,
        reinit="finish_previous",
    )


def simulate_single_env(env: PreparedEnvironment, args: argparse.Namespace, job_type: str = "avg_regret") -> None:
    run = _init_wandb_run(args, env, job_type=job_type)
    try:
        _simulate_one_env(env)
        if getattr(args, "leaf_prob", False):
            run_env_leaf_prob(env)
    finally:
        run.finish()


def _log_summary_payload(summary: Dict[str, Any]) -> None:
    wandb = _require_wandb()

    wandb.config.update({"simulation_parameters": summary["simulation_parameters"]}, allow_val_change=True)

    wandb.summary["env_name"] = summary["env_name"]
    wandb.summary["runs"] = int(summary["runs"])
    wandb.summary["bestPath"] = int(summary["tree_metrics"]["bestPath"])
    wandb.summary["bestPathP"] = float(summary["tree_metrics"]["bestPathP"])
    wandb.summary["tree_metrics"] = summary["tree_metrics"]

    for algo_name, metrics in summary["algorithms_summary"].items():
        for key, value in metrics.items():
            wandb.summary[f"algorithms_summary/{key}"] = value


def _simulate_one_env(env: PreparedEnvironment) -> None:
    summary_algorithms: Dict[str, Dict[str, float]] = {}
    eps_ee3 = env.max_branching * (env.rounds ** (-1.0 / (env.depth_value + 1.0)))
    eps_ps = env.max_branching * (env.rounds ** (-1.0 / (env.r0 + 1.0)))
    eta_ps = env.rounds ** (-float(env.r0) / (env.r0 + 1.0)) if env.r0 > 0 else 1.0
    eta_e3 = math.sqrt(math.log(max(int(env.max_branching), 2)) / (env.rounds * max(int(env.max_branching), 1)))
    eta_ee3 = env.rounds ** (-float(env.depth_value) / (env.depth_value + 1.0))

    rows_generator_list = []
    final_rows = []

    for idx, algo_id in enumerate(env.algo_ids):
        algo_name = ALGO_ID_TO_NAME[int(algo_id)]
        base_seed = env.seed + 10007 * (idx + 1)
        cost_runs: List[np.ndarray] = []
        regret_runs: List[np.ndarray] = []
        accum_runs: List[np.ndarray] = []
        avg_runs: List[np.ndarray] = []
        best_path_hits = np.zeros(env.rounds, dtype=np.float64)
        share_hits = np.zeros(env.rounds, dtype=np.float64)
        avg_costs: List[float] = []

        for rep in range(NUM_AVERAGE_RUNS):
            run_seed = base_seed + 1000003 * rep
            print(f"Running {algo_name} on {env.env_name}, run {rep + 1}/{NUM_AVERAGE_RUNS} with seed {run_seed}")
            cost, leaf_used, regret, accum, avg, _ = _run_algo_numba(
                int(algo_id),
                int(run_seed),
                int(env.rounds),
                int(env.n),
                env.parent,
                env.child_start,
                env.child_count,
                env.child_list,
                env.is_leaf,
                env.is_safe,
                env.is_share,
                env.leaf_prob,
                env.leaf_distribution,
                env.leaf_count,
                env.subtree_leaf_start,
                env.subtree_leaf_count,
                env.subtree_leaves,
                env.need_explore,
                int(env.depth_value),
                int(max(env.max_branching, 1)),
                int(env.r0),
                int(env.best_leaf),
                float(env.best_leaf_p),
                False,
            )

            cost_runs.append(cost)
            regret_runs.append(regret)
            accum_runs.append(accum)
            avg_runs.append(avg)
            best_path_hits += (leaf_used == env.best_leaf).astype(np.float64)
            share_hits += (env.is_share[leaf_used] == 1).astype(np.float64)
            avg_costs.append(float(np.mean(cost)))

        mean_cost = np.mean(np.stack(cost_runs, axis=0), axis=0)                # mean across runs, shape (rounds,)
        mean_regret = np.mean(np.stack(regret_runs, axis=0), axis=0)
        mean_accum = np.mean(np.stack(accum_runs, axis=0), axis=0)
        mean_avg = np.mean(np.stack(avg_runs, axis=0), axis=0)
        mean_best_path_rate = best_path_hits / float(NUM_AVERAGE_RUNS)
        mean_share_rate = share_hits / float(NUM_AVERAGE_RUNS)

        avg_cost = float(np.mean(avg_costs))                                    # average cost across all runs (single scalar)
        best_path_rate = float(np.mean(mean_best_path_rate))
        share_rate = float(np.mean(mean_share_rate))

        summary_algorithms[algo_name] = {
            "avgCost": avg_cost,
            "bestPathRate": float(best_path_rate),
            "shareRate": float(share_rate),
            "finalAccumRegret": float(mean_accum[-1]),
            "finalAvgRegret": float(mean_avg[-1]),
            "runs": int(NUM_AVERAGE_RUNS),
        }

        rows_generator_list.append((env.env_name, algo_name, mean_cost, mean_regret, mean_accum, mean_avg, mean_best_path_rate, mean_share_rate))
        final_rows.append(
            (
                algo_name,
                env.env_name,
                float(avg_cost),
                float(best_path_rate),
                float(share_rate),
                float(mean_avg[-1]),
                float(mean_accum[-1]),
            )
        )

    _log_dynamic_table_streaming(final_rows)
    _log_avg_regret_plot_streaming(rows_generator_list, env.env_name)

    summary = {
        "env_name": env.env_name,
        "seed": env.seed,
        "node_counts": env.n,
        "rounds": env.rounds,
        "runs": int(NUM_AVERAGE_RUNS),
        "simulation_parameters": {
            "eps_ps": float(eps_ps),
            "eps_ee3": float(eps_ee3),
            "eta_ps": float(eta_ps),
            "eta_e3": float(eta_e3),
            "eta_ee3": float(eta_ee3),
            "num_average_runs": int(NUM_AVERAGE_RUNS),
        },
        "algorithms": [ALGO_ID_TO_NAME[int(x)] for x in env.algo_ids],
        "tree_metrics": {
            "K": int(env.depth_value),
            "S": int(env.max_branching),
            "bestPath": int(env.best_leaf),
            "bestPathP": float(env.best_leaf_p),
            "R0": int(env.r0),
            "p": env.leaf_prob.tolist(),
            "d": env.d.tolist(),
            "isLeaf": env.is_leaf.tolist(),
            "isSafe": env.is_safe.tolist(),
            "leafCount": env.leaf_count.tolist(),
            "R": env.risk.tolist(),
            "needExplore": env.need_explore.tolist(),
        },
        "algorithms_summary": summary_algorithms,
    }

    _log_summary_payload(summary)
    _log_summary_bars(env.env_name, summary_algorithms)
    print(f"[{env.env_name}] logged avg-regret chart, dynamic table, summary, and summary-bars to WandB")


def run_env_leaf_prob(env: PreparedEnvironment) -> None:

    rows_generator_list = []

    for idx, algo_id in enumerate(env.algo_ids):
        algo_name = ALGO_ID_TO_NAME[int(algo_id)]
        base_seed = env.seed + 10007 * (idx + 1)
        cost, leaf_used, regret, accum, avg, leaf_probs = _run_algo_numba(
            int(algo_id),
            int(base_seed),
            int(env.rounds),
            int(env.n),
            env.parent,
            env.child_start,
            env.child_count,
            env.child_list,
            env.is_leaf,
            env.is_safe,
            env.is_share,
            env.leaf_prob,
            env.leaf_distribution,
            env.leaf_count,
            env.subtree_leaf_start,
            env.subtree_leaf_count,
            env.subtree_leaves,
            env.need_explore,
            int(env.depth_value),
            int(max(env.max_branching, 1)),
            int(env.r0),
            int(env.best_leaf),
            float(env.best_leaf_p),
            True,
        )

        best_path_hits = (leaf_used == env.best_leaf).astype(np.float64)
        share_hits = (env.is_share[leaf_used] == 1).astype(np.float64)

        rows_generator_list.append((env.env_name, algo_name, cost, regret, accum, avg, best_path_hits, share_hits, 1))

        _log_leaf_probabilities(leaf_probs, env.leaves, env.env_name, algo_name)


def _load_environments(input_file: Path) -> List[PreparedEnvironment]:
    raw = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("input JSON must be an array of environments")

    envs = []
    for idx, obj in enumerate(raw):
        if not isinstance(obj, dict):
            raise ValueError(f"env[{idx}] must be an object")
        envs.append(_prepare_environment(_validate_env_obj(obj, idx)))
    return envs


def load_environments(input_file: Path) -> List[PreparedEnvironment]:
    return _load_environments(input_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multistage bandit simulator (JSON + Numba)")
    parser.add_argument("--leaf-prob", action="store_true", help="whether to track and log leaf selection probabilities (can increase runtime and memory)")
    parser.add_argument("--csv-output", action="store_true", help="whether to write dynamic metrics to CSV file (in addition to WandB)")
    parser.add_argument("--input", required=True, help="path to JSON file (array of environments)")
    parser.add_argument("--output-dir", default="results_json", help="unused; kept for backward compatibility")
    parser.add_argument("--wandb-group", default="simulation-result", help="optional WandB group")
    parser.add_argument("--wandb-name", default=None, help="optional WandB run name")
    parser.add_argument("--sweep-group", default=None, help="optional WandB group override used by sweeps")
    parser.add_argument("--testcase-path", default=None, help="path of generated testcase for logging")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)

    envs = _load_environments(input_file)
    for env in envs:
        simulate_single_env(env, args, job_type="avg_regret")


if __name__ == "__main__":
    main()
