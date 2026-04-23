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
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
from numba import njit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12
NUM_AVERAGE_RUNS = 20

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
    risk: np.ndarray
    r0: int
    need_explore: np.ndarray
    leaves: np.ndarray
    best_leaf: int
    best_leaf_p: float


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
        risk=risk,
        r0=int(risk[0]),
        need_explore=need_explore,
        leaves=leaves,
        best_leaf=best_leaf,
        best_leaf_p=best_p,
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
    return 1.0 if p < t_ratio else 0.0


@njit(cache=True)
def _sample_leaf_cost(dist_code: int, p: float, t: int, rounds: int) -> float:
    if dist_code == DIST_TIME_VARIANT:
        t_ratio = float(t + 1) / float(max(rounds, 1))
        return _sample_timevariant(p, t_ratio)
    return _sample_bernoulli(p)


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
    need_explore: np.ndarray,
    depth_value: int,
    max_branching: int,
    r0: int,
    best_leaf: int,
    best_leaf_p: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    theta = np.zeros(n, dtype=np.float64)
    w = leaf_count.astype(np.float64)
    q = np.ones(n, dtype=np.float64)
    count_seen = np.zeros(n, dtype=np.int64)

    epsilon = max_branching * (rounds ** (-1.0 / (depth_value + 1.0))) if depth_value >= 0 else 0.0
    if epsilon < 0.0:
        epsilon = 0.0
    if epsilon > 1.0:
        epsilon = 1.0

    eta_ps = rounds ** (-float(r0) / (r0 + 1.0)) if r0 > 0 else 1.0
    eta_e3 = math.sqrt(math.log(max(max_branching, 2)) / (rounds * max(max_branching, 1)))
    eta_ee3 = rounds ** (-float(depth_value) / (depth_value + 1.0)) if depth_value > 0 else 1.0

    cost = np.zeros(rounds, dtype=np.float64)
    leaf_used = np.zeros(rounds, dtype=np.int64)
    regret = np.zeros(rounds, dtype=np.float64)
    accum = np.zeros(rounds, dtype=np.float64)
    avg = np.zeros(rounds, dtype=np.float64)

    path_nodes = np.empty(n, dtype=np.int64)
    prob_to_node = np.ones(n, dtype=np.float64)
    local_prob = np.ones(n, dtype=np.float64)
    temp_probs = np.empty(max(max_branching, 1), dtype=np.float64)

    cum_reg = 0.0

    for t in range(rounds):
        node = 0
        depth = 0
        prob_to_node[0] = 1.0
        local_prob[0] = 1.0

        while is_leaf[node] == 0:
            path_nodes[depth] = node
            depth += 1

            start = child_start[node]
            cnt = child_count[node]
            chosen_slot = 0
            chosen_prob = 1.0 / cnt

            if algo_id == 4:
                for i in range(cnt):
                    temp_probs[i] = 1.0 / cnt
                chosen_slot = int(np.random.randint(0, cnt))
                chosen_prob = temp_probs[chosen_slot]

            elif algo_id == 5:
                chosen_slot = 0
                best_q = q[child_list[start]]
                for i in range(1, cnt):
                    cand = child_list[start + i]
                    if q[cand] < best_q - 1e-15:
                        best_q = q[cand]
                        chosen_slot = i
                chosen_prob = 1.0

            elif algo_id == 1:
                if is_safe[node] == 1:
                    total_w = w[node]
                    if total_w <= EPS:
                        for i in range(cnt):
                            temp_probs[i] = 1.0 / cnt
                    else:
                        for i in range(cnt):
                            child = child_list[start + i]
                            temp_probs[i] = max(w[child], 0.0) / total_w
                    chosen_slot = _sample_discrete(temp_probs, cnt)
                    chosen_prob = max(temp_probs[chosen_slot], EPS)
                else:
                    real_eps = epsilon if need_explore[node] == 1 else 0.0
                    _softmax_child(theta, child_list, start, cnt, eta_ps, temp_probs)
                    for i in range(cnt):
                        temp_probs[i] = real_eps / cnt + (1.0 - real_eps) * temp_probs[i]
                    chosen_slot = _sample_discrete(temp_probs, cnt)
                    chosen_prob = max(temp_probs[chosen_slot], EPS)

            elif algo_id == 2 or algo_id == 6:
                _softmax_child(theta, child_list, start, cnt, eta_e3, temp_probs)
                chosen_slot = _sample_discrete(temp_probs, cnt)
                chosen_prob = max(temp_probs[chosen_slot], EPS)

            else:
                _softmax_child(theta, child_list, start, cnt, eta_ee3, temp_probs)
                for i in range(cnt):
                    temp_probs[i] = epsilon / cnt + (1.0 - epsilon) * temp_probs[i]
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

        if algo_id == 1:
            pi_leaf = max(prob_to_node[leaf], EPS)
            delta = math.exp(theta[leaf] - c / pi_leaf) - math.exp(theta[leaf])
            cur = leaf
            while cur != -1:
                w[cur] += delta
                theta[cur] -= c / max(prob_to_node[cur], EPS)
                cur = parent[cur]

        elif algo_id == 2:
            cur = leaf
            while cur != -1:
                theta[cur] -= c / max(local_prob[cur], EPS)
                cur = parent[cur]

        elif algo_id == 3 or algo_id == 6:
            cur = leaf
            while cur != -1:
                theta[cur] -= c / max(prob_to_node[cur], EPS)
                cur = parent[cur]

        elif algo_id == 5:
            cur = leaf
            while cur != -1:
                count_seen[cur] += 1
                q[cur] = q[cur] + (c - q[cur]) / count_seen[cur]
                cur = parent[cur]

    return cost, leaf_used, regret, accum, avg


def _plot_avg_regret(csv_rows: List[Dict[str, int | float | str]], env_name: str, output_path: Path) -> None:
    algos = sorted({row["algo"] for row in csv_rows})
    plt.figure(figsize=(10, 6))
    for algo in algos:
        filtered = [row for row in csv_rows if row["algo"] == algo]
        if not filtered:
            continue

        max_t = max(int(row["t"]) for row in filtered)
        skip_until = int(max_t * 0.01)  # skip first 1% of data to reduce noise in the plot
        plot_rows = [row for row in filtered if int(row["t"]) > skip_until]
        if not plot_rows:
            plot_rows = filtered

        ts = np.array([int(row["t"]) for row in plot_rows], dtype=np.int64)
        ys = np.array([float(row["avgRegret"]) for row in plot_rows], dtype=np.float64)
        plt.plot(ts, ys, label=algo, linewidth=1.5)

    plt.title(f"avgRegret[t] vs t ({env_name})")
    plt.xlabel("t")
    plt.ylabel("avgRegret[t]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_csv(rows: List[Dict[str, int | float | str]], output_file: Path) -> None:
    headers = ["env_name", "algo", "t", "cost", "regret", "accumRegret", "avgRegret", "bestPathRate", "shareRate", "runs"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(
            f"{row['env_name']},{row['algo']},{int(row['t'])},"
            f"{row['cost']:.10f},{row['regret']:.10f},{row['accumRegret']:.10f},{row['avgRegret']:.10f},"
            f"{row['bestPathRate']:.10f},{row['shareRate']:.10f},{int(row['runs'])}"
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _simulate_one_env(env: PreparedEnvironment, output_dir: Path) -> None:
    rows: List[Dict[str, int | float | str]] = []
    summary_algorithms: Dict[str, Dict[str, float]] = {}

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
            cost, leaf_used, regret, accum, avg = _run_algo_numba(
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
                env.need_explore,
                int(env.depth_value),
                int(max(env.max_branching, 1)),
                int(env.r0),
                int(env.best_leaf),
                float(env.best_leaf_p),
            )

            cost_runs.append(cost)
            regret_runs.append(regret)
            accum_runs.append(accum)
            avg_runs.append(avg)
            best_path_hits += (leaf_used == env.best_leaf).astype(np.float64)
            share_hits += (env.is_share[leaf_used] == 1).astype(np.float64)
            avg_costs.append(float(np.mean(cost)))

        mean_cost = np.mean(np.stack(cost_runs, axis=0), axis=0)
        mean_regret = np.mean(np.stack(regret_runs, axis=0), axis=0)
        mean_accum = np.mean(np.stack(accum_runs, axis=0), axis=0)
        mean_avg = np.mean(np.stack(avg_runs, axis=0), axis=0)
        mean_best_path_rate = best_path_hits / float(NUM_AVERAGE_RUNS)
        mean_share_rate = share_hits / float(NUM_AVERAGE_RUNS)

        avg_cost = float(np.mean(avg_costs))
        best_path_rate = float(np.mean(mean_best_path_rate))
        share_rate = float(np.mean(mean_share_rate))

        summary_algorithms[algo_name] = {
            "avgCost": avg_cost,
            "bestPath": int(env.best_leaf),
            "bestPathRate": float(best_path_rate),
            "shareRate": float(share_rate),
            "finalAccumRegret": float(mean_accum[-1]),
            "finalAvgRegret": float(mean_avg[-1]),
            "runs": int(NUM_AVERAGE_RUNS),
        }

        for t in range(env.rounds):
            rows.append(
                {
                    "env_name": env.env_name,
                    "algo": algo_name,
                    "t": t + 1,
                    "cost": float(mean_cost[t]),
                    "regret": float(mean_regret[t]),
                    "accumRegret": float(mean_accum[t]),
                    "avgRegret": float(mean_avg[t]),
                    "bestPathRate": float(mean_best_path_rate[t]),
                    "shareRate": float(mean_share_rate[t]),
                    "runs": int(NUM_AVERAGE_RUNS),
                }
            )

    env_dir = output_dir / env.env_name
    env_dir.mkdir(parents=True, exist_ok=True)

    csv_path = env_dir / "dynamic_metrics.csv"
    json_path = env_dir / "summary.json"
    plot_path = env_dir / "avg_regret.png"

    _write_csv(rows, csv_path)
    _plot_avg_regret(rows, env.env_name, plot_path)

    summary = {
        "env_name": env.env_name,
        "seed": env.seed,
        "node_counts": env.n,
        "rounds": env.rounds,
        "runs": int(NUM_AVERAGE_RUNS),
        "algorithms": [ALGO_ID_TO_NAME[int(x)] for x in env.algo_ids],
        "tree_metrics": {
            "isLeaf": env.is_leaf.tolist(),
            "isSafe": env.is_safe.tolist(),
            "K": int(env.depth_value),
            "d": env.d.tolist(),
            "S": int(env.max_branching),
            "leafCount": env.leaf_count.tolist(),
            "R": env.risk.tolist(),
            "R0": int(env.r0),
            "needExplore": env.need_explore.tolist(),
            "bestPath": int(env.best_leaf),
            "bestPathP": float(env.best_leaf_p),
        },
        "algorithms_summary": summary_algorithms,
        "files": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_path),
        },
    }

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{env.env_name}] wrote {csv_path}")
    print(f"[{env.env_name}] wrote {json_path}")
    print(f"[{env.env_name}] wrote {plot_path}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multistage bandit simulator (JSON + Numba)")
    parser.add_argument("--input", required=True, help="path to JSON file (array of environments)")
    parser.add_argument("--output-dir", default="results_json", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = _load_environments(input_file)
    for env in envs:
        _simulate_one_env(env, output_dir)


if __name__ == "__main__":
    main()
