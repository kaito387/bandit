import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 1, width_ratios=[1])
gs.update(left=0.13, right=0.97, top=0.8, bottom=0.18, wspace=0.32)

caterpillar = fig.add_subplot(gs[0])

caterpillar.set_title('Caterpillar Tree (K=5)', fontsize=22)


# data
CatASH = [0.163792054, 0.158114343, 0.153870755, 0.141113131, 0.113653415, 0.097429824, 0.077277839, 0.068872218, 0.050523277, 0.04195812, 0.026516053, 0.007244392, -0.00856054, -0.00856054, -0.012341437, -0.01929519, -0.03102171]
CatEE3 = [0.168055937] * len(CatASH)
CatQ = [0.192006737] * len(CatASH)
CatR = [0.26718494] * len(CatASH)

Cat_x_pos = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1]

caterpillar.plot(Cat_x_pos, CatQ, marker='^', markersize=12, linestyle='-', linewidth=2.5, label='NaiveMix', color='#426A2A')
caterpillar.plot(Cat_x_pos, CatR, marker='*', markersize=12, linestyle='--', linewidth=2.5, label='Random', color='#B4AA37')
caterpillar.plot(Cat_x_pos, CatEE3, marker='s', markersize=12, linestyle='-.', linewidth=2.5, label='$\epsilon$-EXP3', color='#B59A56')
caterpillar.plot(Cat_x_pos, CatASH, marker='v', markersize=12, linestyle=':', linewidth=2.5, label='ASH', color='#B94F67')

caterpillar.set_ylabel('Final Average Regret', fontsize=20)
caterpillar.set_xlabel('Share-Admissible Node Ratio', fontsize=20)
caterpillar.set_xticks([0.6, 0.7, 0.8, 0.9, 1])
caterpillar.tick_params(axis='x', labelsize=20)
caterpillar.tick_params(axis='y', labelsize=20)
caterpillar.set_yticks([-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
caterpillar.grid(True, linestyle='--', alpha=0.8)

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

plt.savefig('abstract.pdf', dpi=300, bbox_inches='tight')
plt.show()