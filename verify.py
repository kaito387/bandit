"""Check regret values from a quick run to verify behavior matches paper."""
import numpy as np
from simulate import run_eps_exp3, run_exp3, build_leaf_probs, optimal_cost_at

# K=2, S=2, pmin=0.2 - simplest case
K, S, pmin = 2, 2, 0.2
T = 500_000
leaf_probs = build_leaf_probs(K, S, pmin)
change_time = T // 100

print(f"Leaf probs: {leaf_probs}")
print(f"Change time: {change_time}")
print(f"Optimal cost at T: {optimal_cost_at(T, leaf_probs, change_time):.1f}")
print(f"  changed leaf cost: {change_time:.0f}")
print(f"  pmin leaf cost: {pmin*T:.0f}")

eta = T ** (-float(S) / (S + 1))
pi_arr = np.array([T ** (-1.0 / (S + 1)), 0.0])
gamma = min(1.0, np.sqrt(K * np.log(K) / T))

sample_times = np.array([100_000, 200_000, 300_000, 400_000, 500_000], dtype=np.int64)

# Average over 10 runs
eps_regrets = []
exp3_regrets = []
for _ in range(10):
    c1 = run_eps_exp3(K, S, T, leaf_probs, change_time, eta, pi_arr, sample_times)
    c2 = run_exp3(K, S, T, leaf_probs, change_time, gamma, sample_times)
    for i, st in enumerate(sample_times):
        opt = optimal_cost_at(int(st), leaf_probs, change_time)
        eps_regrets.append((c1[i] - opt) / st)
        exp3_regrets.append((c2[i] - opt) / st)

eps_regrets = np.array(eps_regrets).reshape(10, 5)
exp3_regrets = np.array(exp3_regrets).reshape(10, 5)

print("\nTime-avg regret at T checkpoints (mean of 10 runs):")
print("  T(k)    eps-EXP3    EXP3")
for i, st in enumerate(sample_times):
    print(f"  {st//1000:>5}    {eps_regrets[:,i].mean():.4f}      {exp3_regrets[:,i].mean():.4f}")

print(f"\nExpected: eps-EXP3 regret should decrease, EXP3 should converge to ~pmin={pmin}")
