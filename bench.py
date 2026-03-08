import numpy as np
import time
from simulate import run_eps_exp3, run_exp3

# Benchmark: T=100k, K=2 S=2 and K=4 S=4
for K, S in [(2,2), (4,4)]:
    T = 100_000
    lp = np.linspace(0.2, 1.0, K**S)
    ct = T // 100
    eta = T ** (-float(S)/(S+1))
    pi_arr = np.empty(S)
    for l in range(S):
        pi_arr[l] = 0.0 if l == S-1 else T ** (-1.0/(S+1))
    gamma = min(1.0, np.sqrt(K*np.log(K)/T))
    st = np.array([T], dtype=np.int64)

    t0 = time.time()
    run_eps_exp3(K, S, T, lp, ct, eta, pi_arr, st)
    t1 = time.time()
    run_exp3(K, S, T, lp, ct, gamma, st)
    t2 = time.time()
    print(f'K={K} S={S}: eps-EXP3 {t1-t0:.2f}s, EXP3 {t2-t1:.2f}s per 100k rounds')
    est_1M = (t1-t0+t2-t1)*10*20
    print(f'  Estimated for T=1M, 20 runs: {est_1M:.0f}s = {est_1M/60:.1f}min')
