import numpy as np
from simulate import run_eps_exp3, run_exp3

K, S, T = 2, 2, 5000
lp = np.linspace(0.2, 1.0, 4)
ct = T // 100
eta = T ** (-2.0/3)
pi_arr = np.array([T ** (-1.0/3), 0.0])
st = np.array([2500, 5000], dtype=np.int64)

print('Compiling eps-EXP3...')
c1 = run_eps_exp3(K, S, T, lp, ct, eta, pi_arr, st)
print('eps-EXP3 costs:', c1)

gamma = min(1.0, np.sqrt(K*np.log(K)/T))
print('Compiling EXP3...')
c2 = run_exp3(K, S, T, lp, ct, gamma, st)
print('EXP3 costs:', c2)

for i in range(2):
    opt = min(lp[:-1].min()*st[i], 1.0*min(st[i],ct))
    print(f't={st[i]}: eps-regret={(c1[i]-opt)/st[i]:.4f}, exp3-regret={(c2[i]-opt)/st[i]:.4f}')
print('OK')
