import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 1, width_ratios=[1])
gs.update(left=0.13, right=0.97, top=0.8, bottom=0.18, wspace=0.32)

caterpillar = fig.add_subplot(gs[0])

caterpillar.set_title('Caterpillar Tree', fontsize=22)

# data
CatK2 = [0.003020643, 0.005927195, 0.010998763, 0.021994197, 0.028135508, 0.038308791, 0.033647871, 0.022799639, 0.016026707, 0.01368844, 0.0007617, 0.0007617, -0.004629919, -0.006319094, -0.014784434, -0.023803804, -0.027593834, -0.027593834, -0.028431304, -0.032903607, -0.03458542, ]
CatK3 = [0.026252004, 0.033790489, 0.048838937, 0.063886993, 0.070030157, 0.084925802, 0.080564723, 0.062797125, 0.041098398, 0.035838578, 0.023143539, 0.028171293, 0.018001734, 0.010378191, -0.000995944, -0.013612261, -0.018679072, -0.018679072, -0.022447114, -0.030026432, -0.037648174, ]
CatK4 = [0.088981856, 0.088981856, 0.097369378, 0.107860566, 0.109042253, 0.115410651, 0.110028322, 0.098599717, 0.074329301, 0.065120545, 0.049816959, 0.047134002, 0.030667288, 0.025578112, 0.011742954, -0.005996515, -0.016227721, -0.016227721, -0.020021415, -0.027588979, -0.037721329, ]
CatK5 = [0.158819293, 0.159530423, 0.161195234, 0.162133427, 0.163792054, 0.158114343, 0.153870755, 0.141113131, 0.113653415, 0.097429824, 0.077277839, 0.068872218, 0.050523277, 0.04195812, 0.026516053, 0.007244392, -0.00856054, -0.00856054, -0.012341437, -0.01929519, -0.03102171, ]
CatK6 = [0.189053626, 0.188710189, 0.18730867, 0.185909133, 0.184638001, 0.181423695, 0.170521044, 0.160538907, 0.139907988, 0.108705539, 0.088289537, 0.078896889, 0.052801371, 0.039720196, 0.023511495, 0.004224289, -0.008901202, -0.008901202, -0.012637346, -0.024653481, -0.036324242, ]
CatK7 = [0.20119642, 0.200605612, 0.199639595, 0.197687663, 0.195540996, 0.192959685, 0.18146679, 0.170487087, 0.148201192, 0.128915176, 0.103530259, 0.094240012, 0.067067889, 0.061832489, 0.039614742, 0.020718735, 0.007697747, 0.006707668, 0.002942648, -0.014956904, -0.035707599, ]
# CatK8 = [0.205346764, 0.205067061, 0.204138428, 0.203426552, 0.202032266, 0.201072192, 0.198910991, 0.184348507, 0.165506907, 0.143537645, 0.118485841, 0.113641753, 0.082026367, 0.07724646, 0.055504622, 0.030891718, 0.017982115, 0.012195161, 0.003638647, -0.014143903, -0.035165198, ]

Cat_x_pos = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1]

caterpillar.plot(Cat_x_pos, CatK2, marker='o', markersize=12, linestyle='-', linewidth=2.5, label='K=2', color='#69C0FF')
caterpillar.plot(Cat_x_pos, CatK3, marker='s', markersize=12, linestyle='--', linewidth=2.5, label='K=3', color='#95DE64')
caterpillar.plot(Cat_x_pos, CatK4, marker='^', markersize=12, linestyle='-.', linewidth=2.5, label='K=4', color='#FFEB6B')
caterpillar.plot(Cat_x_pos, CatK5, marker='v', markersize=12, linestyle=':', linewidth=2.5, label='K=5', color='#FFBC5C')
caterpillar.plot(Cat_x_pos, CatK6, marker='X', markersize=12, linestyle='-', linewidth=2.5, label='K=6', color="#B53EF5")
caterpillar.plot(Cat_x_pos, CatK7, marker='D', markersize=12, linestyle='--', linewidth=2.5, label='K=7', color='#FF6B6B')
# caterpillar.plot(Cat_x_pos, CatK8, marker='*', markersize=12, linestyle='-.', linewidth=2.5, label='K=8', color='#FF929D')

caterpillar.set_ylabel('Final Average Regret', fontsize=20)
caterpillar.set_xlabel('Share-Admissible Node Ratio', fontsize=20)
caterpillar.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
caterpillar.tick_params(axis='x', labelsize=20)
caterpillar.tick_params(axis='y', labelsize=20)
caterpillar.set_yticks([-0.05, 0, 0.05, 0.1, 0.15, 0.2])
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

plt.savefig('sim_ps_ratio_depth_caterpillar_only.pdf', dpi=300, bbox_inches='tight')
plt.show()
