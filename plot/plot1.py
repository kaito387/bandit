import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(28, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
gs.update(left=0.07, right=0.97, top=0.8, bottom=0.18, wspace=0.32)

caterpillar = fig.add_subplot(gs[0])
fullTree = fig.add_subplot(gs[1])
# caterpillar.set_title('Caterpillar Tree', fontsize=22)
# fullTree.set_title('Full Binary Tree', fontsize=22)

# data
# CatE3 = [0.150119219, 0.150112199, 0.150109718, 0.150121252, 0.150118143, 0.150115457, 0.150119748]
CatEE3 = [0.004598144, 0.033462369, 0.10466662, 0.168055937, 0.192523281, 0.202656258]
CatQ = [0.12899184, 0.183003012, 0.158993078, 0.192006737, 0.188391788, 0.181999828]
CatR = [0.26250542, 0.281254896, 0.278124683, 0.26718494, 0.254526742, 0.2425696]
# CatE3Q = [-0.01553483, 0.003438587, 0.009018691, 0.02146055, 0.055050475, 0.063519293, 0.113545414]
# CatPS80 = [0.022799639, 0.062797125, 0.098599717, 0.141113131, 0.160538907, 0.170487087, 0.18434850]
# CatPS90 = [0.01368844, 0.035838578, 0.065120545, 0.097429824, 0.108705539, 0.128915176, 0.143537645]
CatPS95 = [-0.006319094, 0.010378191, 0.025578112, 0.04195812, 0.039720196, 0.061832489]
CatPS98 = [-0.027593834, -0.018679072, -0.016227721, -0.00856054, -0.008901202, 0.007697747]
CatPS100 = [-0.03458542, -0.037648174, -0.037721329, -0.03102171, -0.036324242, -0.035707599]

Cat_x_pos = np.arange(2, 8)

caterpillar.plot(Cat_x_pos, CatQ, marker='^', markersize=12, linestyle='-', linewidth=2.5, label='NaiveMix', color='#426A2A')
caterpillar.plot(Cat_x_pos, CatR, marker='*', markersize=12, linestyle='--', linewidth=2.5, label='Random', color='#B4AA37')
caterpillar.plot(Cat_x_pos, CatEE3, marker='s', markersize=12, linestyle='-.', linewidth=2.5, label='$\epsilon$-EXP3', color='#B59A56')
caterpillar.plot(Cat_x_pos, CatPS95, marker='v', markersize=12, linestyle=':', linewidth=2.5, label='ASH-0.95', color='#B94F67')
caterpillar.plot(Cat_x_pos, CatPS98, marker='X', markersize=12, linestyle='-', linewidth=2.5, label='ASH-0.98', color="#FF528D")
caterpillar.plot(Cat_x_pos, CatPS100, marker='o', markersize=12, linestyle='--', linewidth=2.5, label='ASH-1.00', color='#B53EF5')

caterpillar.set_ylabel('Average Regret', fontsize=20)
caterpillar.set_xlabel('Tree Depth K', fontsize=20)
caterpillar.set_xticks([2, 3, 4, 5, 6, 7])
caterpillar.tick_params(axis='x', labelsize=20)
caterpillar.tick_params(axis='y', labelsize=20)
caterpillar.set_yticks([-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
caterpillar.grid(True, linestyle='--', alpha=0.8)

Full_x_pos = np.arange(2, 8)

# data
# FullE3 = [0.151169348, 0.151863024, 0.152603144, 0.153375388, 0.15416932, 0.154952852]
FullEE3 = [0.011948888, 0.069097756, 0.186034004, 0.286203396, 0.342833304, 0.376423476]
FullQ = [0.201975492, 0.21169808, 0.265982704, 0.242522492, 0.266213072, 0.24627294]
FullR = [0.26250362, 0.356240052, 0.40311344, 0.426534916, 0.438289228, 0.44418432]
# FullE3Q = [0.004245324, 0.035613616, 0.067521328, 0.094534604, 0.114673316, 0.142829136]
# FullPS80 = [0.058876176, 0.116496936, 0.221243064, 0.30003544, 0.350201908, 0.37846838]
# FullPS90 = [0.03439048, 0.10178524, 0.225596608, 0.305598132, 0.354164284, 0.380085044]
FullPS95 = [0.010350132, 0.044245316, 0.172165164, 0.297170764, 0.356346836, 0.381102992]
FullPS98 = [-0.018274264, -0.005191204, 0.07807984, 0.194288144, 0.316973168, 0.363764488]
FullPS100 = [-0.037828252, -0.032446688, -0.031799352, -0.030688088, -0.028122336, -0.01877198]


fullTree.plot(Full_x_pos, FullQ, marker='^', markersize=12, linestyle='-', linewidth=2.5, label='NaiveMix', color='#426A2A')
fullTree.plot(Full_x_pos, FullR, marker='*', markersize=12, linestyle='--', linewidth=2.5, label='Random', color='#B4AA37')
fullTree.plot(Full_x_pos, FullEE3, marker='s', markersize=12, linestyle='-.', linewidth=2.5, label='$\epsilon$-EXP3', color='#B59A56')
fullTree.plot(Full_x_pos, FullPS95, marker='v', markersize=12, linestyle=':', linewidth=2.5, label='ASH-0.95', color='#B94F67')
fullTree.plot(Full_x_pos, FullPS98, marker='X', markersize=12, linestyle='-', linewidth=2.5, label='ASH-0.98', color="#FF528D")
fullTree.plot(Full_x_pos, FullPS100, marker='o', markersize=12, linestyle='--', linewidth=2.5, label='ASH-1.00', color='#B53EF5')

fullTree.set_ylabel('Final Average Regret', fontsize=20)
fullTree.set_xlabel('Tree Depth K', fontsize=20)
fullTree.set_xticks([2, 3, 4, 5, 6, 7])
fullTree.tick_params(axis='y', labelsize=20)
fullTree.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
fullTree.grid(True, linestyle='--', alpha=0.8)

handles, labels = caterpillar.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    fontsize=18,
    ncol=6,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.92),
    frameon=False
)

plt.savefig('sim_algo_vs_depth.pdf', dpi=300, bbox_inches='tight')
plt.show()