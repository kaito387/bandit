"""
Reproduce Section 7.1 of "Distributed No-Regret Learning for Multi-Stage Systems"
Algorithms: epsilon-EXP3 (Algorithm 2) and Standard EXP3
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit
import time


# ============================================================
# Section 1: Tree helpers
# ============================================================

def build_leaf_probs(K, S, pmin):
    """Create leaf probability array. K^S leaves, linearly spaced pmin to 1.0."""
    num_leaves = K ** S
    # NOTE 这样生成概率理论上是很能体现教育的，但是论文未指定生成方式
    probs = np.linspace(pmin, 1.0, num_leaves)
    return probs


def optimal_cost_at(t, leaf_probs, change_time):
    """Optimal fixed-leaf expected cost over t rounds."""
    costs = leaf_probs * t
    # Last leaf (p=1.0) changes to p=0 at change_time
    costs[-1] = 1.0 * min(t, change_time) + 0.0 * max(0, t - change_time)
    return float(np.min(costs))


# ============================================================
# Section 2: Numba utilities
# ============================================================

@njit(cache=True)
def _softmax(eta, w, K_node):
    """Numerically stable softmax."""
    max_w = w[0]
    for i in range(1, K_node):
        if w[i] > max_w:
            max_w = w[i]
    s = 0.0
    out = np.empty(K_node)
    for i in range(K_node):
        # NOTE if eta * (w[i] - max_w) < -700, exp will underflow to 0
        out[i] = np.exp(eta * (w[i] - max_w))
        s += out[i]
    for i in range(K_node):
        out[i] /= s
    return out


@njit(cache=True)
def _sample_from(probs, K_node):
    """Sample an index from a discrete distribution."""
    # TODO 考虑为随机函数设定种子保证可复现性
    r = np.random.random()
    cum = 0.0
    for c in range(K_node - 1):
        cum += probs[c]
        if r < cum:
            return c
    return K_node - 1


# ============================================================
# Section 3: epsilon-EXP3 (Algorithm 2)
# ============================================================

@njit(cache=True)
def run_eps_exp3(K, S, T, leaf_probs_init, change_time, eta, pi_arr, sample_times):
    """
    One trial of epsilon-EXP3.

    Tree: S levels of internal nodes (level 0 = root). Level l has K^l nodes.
    Node (l, i) has children at level l+1, indices [i*K, i*K+K-1].
    Leaves at level S: K^S leaves.
    Weights: w[l][i*K + c] for child c of node i at level l.

    pi_arr: array of probabilities for uniform selection at each level
    sample_times: array of times at which to record cumulative cost
    """
    # Flat weight storage: offsets[l] marks start of level l weights
    total_w = 0
    offsets = np.empty(S, dtype=np.int64)
    for l in range(S):
        offsets[l] = total_w
        total_w += (K ** l) * K
    w = np.zeros(total_w)

    num_samples = len(sample_times)
    cum_costs = np.zeros(num_samples)
    sample_idx = 0
    cum_cost = 0.0

    for t in range(1, T + 1):
        # --- Forward pass: root to leaf ---
        path_nodes = np.empty(S, dtype=np.int64)
        path_children = np.empty(S, dtype=np.int64)
        path_modes = np.empty(S, dtype=np.int64)  # 0=uniform, 1=EXP3
        path_q = np.empty(S + 1)  # q[l] = prob that node at level l receives job
        path_q[0] = 1.0

        node_idx = 0
        for l in range(S):
            path_nodes[l] = node_idx
            w_start = offsets[l] + node_idx * K

            # Softmax over weights
            w_node = w[w_start:w_start + K]
            sm = _softmax(eta, w_node, K)

            pi_l = pi_arr[l]

            # Mixed probability p[v, c, t]
            p_mix = np.empty(K)
            for c in range(K):
                p_mix[c] = pi_l / K + (1.0 - pi_l) * sm[c]

            # Decide mode
            if np.random.random() < pi_l:
                # Uniform selection mode (education)
                chosen = np.random.randint(0, K)
                mode = 0
            else:
                # EXP3 mode (exploration/exploitation)
                chosen = _sample_from(sm, K)
                mode = 1

            path_children[l] = chosen
            path_modes[l] = mode
            path_q[l + 1] = path_q[l] * p_mix[chosen]
            node_idx = node_idx * K + chosen

        # --- Leaf cost ---
        leaf_idx = node_idx
        p_leaf = leaf_probs_init[leaf_idx]
        if t >= change_time and leaf_idx == len(leaf_probs_init) - 1:
            p_leaf = 0.0
        cost = 1.0 if np.random.random() < p_leaf else 0.0
        cum_cost += cost

        # --- Backward pass: update weights along the path ---
        for l in range(S - 1, -1, -1):
            nidx = path_nodes[l]
            chosen_c = path_children[l]
            mode = path_modes[l]
            q_v = path_q[l]

            w_start = offsets[l] + nidx * K
            w_node = w[w_start:w_start + K]
            sm = _softmax(eta, w_node, K)

            if mode == 0:
                # Uniform mode: g[chosen] = y * |C_v| / q[v,t]
                g = cost * K / q_v
            else:
                # EXP3 mode: g[chosen] = y * sum(exp) / (q[v,t] * exp(chosen))
                # = y / (q[v,t] * sm[chosen])
                sm_c = sm[chosen_c]
                if sm_c > 1e-30:
                    g = cost / (q_v * sm_c)
                else:
                    # NOTE numberical stability
                    g = 0

            # Only update the chosen child's weight
            w[w_start + chosen_c] -= g

        # --- Record sample ---
        if sample_idx < num_samples and t == sample_times[sample_idx]:
            cum_costs[sample_idx] = cum_cost
            sample_idx += 1

    return cum_costs


# ============================================================
# Section 4: Standard EXP3
# ============================================================

@njit(cache=True)
def run_exp3(K, S, T, leaf_probs_init, change_time, eta, sample_times):
    """
    One trial of standard EXP3 (book version, Stoltz 2005: no explicit exploration).
    Each node runs EXP3 independently.
    Uses pure softmax: p[c] = exp(-eta*L[c]) / Z
    """
    total_w = 0
    offsets = np.empty(S, dtype=np.int64)
    for l in range(S):
        offsets[l] = total_w
        total_w += (K ** l) * K
    # Cumulative importance-weighted loss
    L = np.zeros(total_w)

    num_samples = len(sample_times)
    cum_costs = np.zeros(num_samples)
    sample_idx = 0
    cum_cost = 0.0

    for t in range(1, T + 1):
        # --- Forward pass ---
        path_nodes = np.empty(S, dtype=np.int64)
        path_children = np.empty(S, dtype=np.int64)
        path_probs = np.empty(S)

        node_idx = 0
        for l in range(S):
            path_nodes[l] = node_idx
            w_start = offsets[l] + node_idx * K

            # Compute softmax over negative loss
            neg_L = np.empty(K)
            for c in range(K):
                neg_L[c] = -L[w_start + c]

            p = _softmax(eta, neg_L, K)

            chosen = _sample_from(p, K)
            path_children[l] = chosen
            path_probs[l] = p[chosen]
            node_idx = node_idx * K + chosen

        # --- Leaf cost ---
        leaf_idx = node_idx
        p_leaf = leaf_probs_init[leaf_idx]
        if t >= change_time and leaf_idx == len(leaf_probs_init) - 1:
            p_leaf = 0.0
        cost = 1.0 if np.random.random() < p_leaf else 0.0
        cum_cost += cost

        # --- Backward pass: update chosen child at each level ---
        for l in range(S - 1, -1, -1):
            nidx = path_nodes[l]
            chosen_c = path_children[l]
            p_c = path_probs[l]
            w_start = offsets[l] + nidx * K
            # Importance-weighted loss
            if p_c > 1e-30:
                L[w_start + chosen_c] += cost / p_c
            # (unchosen children get 0 update)

        # --- Record ---
        if sample_idx < num_samples and t == sample_times[sample_idx]:
            cum_costs[sample_idx] = cum_cost
            sample_idx += 1

    return cum_costs


@njit(cache=True)
def run_exp3_qv(K, S, T, leaf_probs_init, change_time, eta, sample_times):
    """
    One trial of EXP3 with path-context correction and scalar eta.
    Same forward policy as standard EXP3, but backward update uses q_v
    (probability that node v receives the job) in denominator.
    eta follows the standard EXP3 choice.
    """
    total_w = 0
    offsets = np.empty(S, dtype=np.int64)
    for l in range(S):
        offsets[l] = total_w
        total_w += (K ** l) * K
    L = np.zeros(total_w)

    num_samples = len(sample_times)
    cum_costs = np.zeros(num_samples)
    sample_idx = 0
    cum_cost = 0.0

    for t in range(1, T + 1):
        # --- Forward pass ---
        path_nodes = np.empty(S, dtype=np.int64)
        path_children = np.empty(S, dtype=np.int64)
        path_q = np.empty(S + 1)
        path_q[0] = 1.0

        node_idx = 0
        for l in range(S):
            path_nodes[l] = node_idx
            w_start = offsets[l] + node_idx * K

            neg_L = np.empty(K)
            for c in range(K):
                neg_L[c] = -L[w_start + c]

            p = _softmax(eta, neg_L, K)

            chosen = _sample_from(p, K)
            path_children[l] = chosen
            path_q[l + 1] = path_q[l] * p[chosen]
            node_idx = node_idx * K + chosen

        # --- Leaf cost ---
        leaf_idx = node_idx
        p_leaf = leaf_probs_init[leaf_idx]
        if t >= change_time and leaf_idx == len(leaf_probs_init) - 1:
            p_leaf = 0.0
        cost = 1.0 if np.random.random() < p_leaf else 0.0
        cum_cost += cost

        # --- Backward pass: update chosen child at each level ---
        for l in range(S - 1, -1, -1):
            nidx = path_nodes[l]
            chosen_c = path_children[l]
            q_v = path_q[l]
            w_start = offsets[l] + nidx * K
            if q_v > 1e-30:
                L[w_start + chosen_c] += cost / q_v

        # --- Record ---
        if sample_idx < num_samples and t == sample_times[sample_idx]:
            cum_costs[sample_idx] = cum_cost
            sample_idx += 1

    return cum_costs


# ============================================================
# Section 5: Experiment runner
# ============================================================

def run_experiment(K, S, pmin, T, num_runs=20, num_samples=200, change_time=None):
    """Run one (K, S, pmin) config. Returns regrets of three algorithms."""
    leaf_probs = build_leaf_probs(K, S, pmin)
    if change_time is None:
        change_time = T // 100
    num_leaves = K ** S

    # Sample times (evenly spaced)
    sample_times = np.linspace(T // num_samples, T, num_samples).astype(np.int64)

    # epsilon-EXP3 params (Lemma 4)
    eta_eps = T ** (-float(S) / (S + 1))
    # EXP3-style eta: sqrt(2 log K / (T K))
    eta_exp3 = np.sqrt(2.0 * np.log(K) / (T * K))
    pi_arr = np.empty(S)
    for l in range(S):
        if l == S - 1:
            # Children are leaves -> no education needed
            pi_arr[l] = 0.0
        else:
            pi_arr[l] = T ** (-1.0 / (S + 1))

    # Storage
    eps_costs_all = np.zeros((num_runs, num_samples))
    exp3_qv_costs_all = np.zeros((num_runs, num_samples))

    print(f"  Config: K={K}, S={S}, pmin={pmin}, T={T}, eta_eps={eta_eps:.6f}, "
            f"pi={pi_arr[0]:.6f}, "
            f"eta_qv={eta_exp3:.6f}")

    for run in range(num_runs):
        t0 = time.time()
        eps_costs_all[run] = run_eps_exp3(
            K, S, T, leaf_probs, change_time, eta_eps, pi_arr, sample_times)
        t1 = time.time()
        exp3_qv_costs_all[run] = run_exp3_qv(
            K, S, T, leaf_probs, change_time, eta_exp3, sample_times)
        t2 = time.time()
        if run < 3 or (run + 1) % 5 == 0:
            print(
                f"    Run {run+1}/{num_runs}: eps-EXP3 {t1-t0:.1f}s, "
                f"EXP3-qv {t2-t1:.1f}s"
            )

    # Optimal expected cost at each sample time
    opt_at_sample = np.array([optimal_cost_at(int(st), leaf_probs, change_time)
                              for st in sample_times])

    # Time-average regret: (cum_cost - opt) / t
    eps_regret = (eps_costs_all - opt_at_sample[None, :]) / sample_times[None, :].astype(float)
    exp3_qv_regret = (exp3_qv_costs_all - opt_at_sample[None, :]) / sample_times[None, :].astype(float)

    return sample_times, eps_regret, exp3_qv_regret


# ============================================================
# Section 6: Plotting
# ============================================================

def plot_results(configs, T):
    """Plot Fig. 3: 6 subplots of time-average regret."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for idx, (K, S, pmin) in enumerate(configs):
        print(f"\n=== Experiment {idx+1}/6: K={K}, S={S}, pmin={pmin} ===")
        sample_times, eps_regret, exp3_qv_regret = run_experiment(K, S, pmin, T)

        ax = axes_flat[idx]
        x = sample_times / 1e6

        eps_mean = eps_regret.mean(axis=0)
        eps_std = eps_regret.std(axis=0)
        exp3_qv_mean = exp3_qv_regret.mean(axis=0)
        exp3_qv_std = exp3_qv_regret.std(axis=0)

        ax.plot(x, eps_mean, label='ε-EXP3', color='blue')
        ax.fill_between(x, eps_mean - eps_std, eps_mean + eps_std,
                        alpha=0.2, color='blue')

        ax.plot(x, exp3_qv_mean, label='EXP3-qv', color='orange')
        ax.fill_between(x, exp3_qv_mean - exp3_qv_std, exp3_qv_mean + exp3_qv_std,
                alpha=0.2, color='orange')

        # Asymptotic trend c / T^{1/(S+1)}, calibrated at midpoint
        mid = len(sample_times) // 2
        if eps_mean[mid] > 0:
            c_cal = eps_mean[mid] * sample_times[mid] ** (1.0 / (S + 1))
            trend = c_cal / sample_times ** (1.0 / (S + 1))
            ax.plot(x, trend, '--', label=f'O(1/T^{{1/{S+1}}})', color='green')

        ax.set_title(f'K={K}, S={S}, pmin={pmin}')
        ax.set_xlabel('T (×10⁶)')
        ax.set_ylabel('Time-avg regret')
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('regret.png', dpi=150)
    print("\nSaved regret.png")


def plot_change_time_comparison(K, S, pmin, T, change_times=None):
    """Plot subplots comparing different change_time values."""
    if change_times is None:
        change_times = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    change_time_samples = [int(ct * T) for ct in change_times]

    n = len(change_times)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    
    for idx, (ct_label, ct_value) in enumerate(zip(change_times, change_time_samples)):
        print(f"\n=== Experiment {idx+1}/{len(change_times)}: K={K}, S={S}, change_time={ct_label}T ===")
        sample_times, eps_regret, exp3_qv_regret = run_experiment(
            K, S, pmin, T, change_time=ct_value
        )
        
        ax = axes_flat[idx]
        x = sample_times / 1e6
        
        eps_mean = eps_regret.mean(axis=0)
        eps_std = eps_regret.std(axis=0)
        exp3_qv_mean = exp3_qv_regret.mean(axis=0)
        exp3_qv_std = exp3_qv_regret.std(axis=0)
        
        ax.plot(x, eps_mean, label='ε-EXP3', color='blue')
        ax.fill_between(x, eps_mean - eps_std, eps_mean + eps_std,
                        alpha=0.2, color='blue')
        
        ax.plot(x, exp3_qv_mean, label='EXP3-qv', color='orange')
        ax.fill_between(x, exp3_qv_mean - exp3_qv_std, exp3_qv_mean + exp3_qv_std,
                        alpha=0.2, color='orange')
        
        # Asymptotic trends: O(1/T^(1/(S+1))) and O(1/sqrt(T))
        mid = len(sample_times) // 2
        if eps_mean[mid] > 0:
            # O(1/T^(1/(S+1))) calibrated from eps-EXP3
            c_cal = eps_mean[mid] * sample_times[mid] ** (1.0 / (S + 1))
            trend = c_cal / sample_times ** (1.0 / (S + 1))
            ax.plot(x, trend, '--', label=f'O(1/T^{{1/{S+1}}})', color='green', linewidth=2)
        
        if exp3_qv_mean[mid] > 0:
            # O(1/sqrt(T)) calibrated from EXP3-qv
            c_sqrt = exp3_qv_mean[mid] * np.sqrt(sample_times[mid])
            trend_sqrt = c_sqrt / np.sqrt(sample_times)
            ax.plot(x, trend_sqrt, ':', label='O(1/√T)', color='purple', linewidth=2)
        
        ax.set_title(f'K={K}, S={S}, change_time={ct_label}T')
        ax.set_xlabel('T (×10⁶)')
        ax.set_ylabel('Time-avg regret')
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    # Hide any unused axes when grid is larger than number of experiments.
    for i in range(len(change_times), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    out_file = f'change_time_comparison_K{K}_S{S}.png'
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved {out_file}")


# ============================================================
# Section 7: Main
# ============================================================

if __name__ == '__main__':
    # Run requested configs with selected change_time ratios.
    pmin = 0.2
    T = 10_000_000  # Full experiment
    configs = [(2, 4), (3, 3), (4, 2)]
    target_change_times = [0.01, 0.03]

    # JIT warm-up
    print("Warming up Numba JIT...")
    dummy_lp = np.array([0.5, 1.0])
    dummy_st = np.array([50, 100], dtype=np.int64)
    run_eps_exp3(2, 2, 100, dummy_lp, 5, 0.1, np.array([0.0]), dummy_st)
    run_exp3_qv(2, 2, 100, dummy_lp, 5, 0.1, dummy_st)
    print("JIT ready.\n")

    for K, S in configs:
        plot_change_time_comparison(K, S, pmin, T, change_times=target_change_times)
