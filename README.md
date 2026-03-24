# Multi-Stage Bandit 仿真复现

复现论文 **"Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback"** (I-Hong Hou, MOBIHOC '24) 中 Section 7.1 的实验。

## 项目结构

| 文件 | 说明 |
|------|------|
| `simulate.py` | **主程序**。实现 ε-EXP3 和标准 EXP3 算法，运行 6 组实验并生成 Fig.3 |
| `fig3_regret.png` | 实验输出图（对应论文 Fig. 3） |
| `multistage-noted.pdf` | 原始论文 |

## simulate.py 详细说明

### 模块结构

```
Section 1: Tree helpers
  - build_leaf_probs(K, S, pmin)    # 构建 K^S 个叶节点的 Bernoulli 参数
  - optimal_cost_at(t, ...)         # 计算最优固定策略的期望代价

Section 2: Numba utilities
  - _softmax(eta, w, K)             # 数值稳定的 softmax (max-subtraction trick)
  - _sample_from(probs, K)          # 从离散分布采样

Section 3: ε-EXP3 算法 (Algorithm 2)
  - run_eps_exp3(K, S, T, ...)      # 单次试验，返回各采样点的累积代价

Section 4: 标准 EXP3 算法
  - run_exp3(K, S, T, ...)          # 单次试验，每个节点独立运行 EXP3

Section 5: 实验运行器
  - run_experiment(K, S, pmin, T)   # 执行 20 次独立运行，返回 regret 统计

Section 6: 绘图
  - plot_results(configs, T)        # 生成 2×3 子图 (Fig. 3)

Section 7: Main
  - 6 组参数配置，JIT 预热，调用 plot_results
```

### 算法说明

**ε-EXP3**（论文 Algorithm 2）：
- 树结构：深度 S+1 的 K 叉树，根节点收到 job 后逐层转发到叶节点
- 双模式选择：以概率 π_v 进入 **uniform 模式**（教育子节点），否则进入 **EXP3 模式**（探索/利用）
- 参数：η = T^{-S/(S+1)}，π_v = T^{-1/(S+1)}（若子节点非叶），π_v = 0（若子节点全为叶）
- 权重更新：importance-weighted loss，仅更新被选中的子节点

**标准 EXP3**（基线）：
- 每个节点独立运行 EXP3，以端到端代价作为 loss 信号
- 参数：γ = min(1, √(K ln K / T))

### 实验配置 (Section 7.1)

| 子图 | K | S | p_min |
|------|---|---|-------|
| (a)  | 2 | 2 | 0.2   |
| (b)  | 2 | 3 | 0.4   |
| (c)  | 2 | 4 | 0.6   |
| (d)  | 4 | 2 | 0.2   |
| (e)  | 4 | 3 | 0.4   |
| (f)  | 4 | 4 | 0.6   |

- T = 10^7 轮，每组 20 次独立运行
- 叶节点代价：K^S 个叶节点，Bernoulli 参数从 p_min 线性分布到 1.0
- 动态变化：在 t = T/100 时，p=1.0 的叶节点变为 p=0
- 度量：time-average regret = (累积代价 − 最优固定策略代价) / t

## 依赖

- Python 3.10+
- NumPy
- Numba（JIT 编译加速核心循环）
- Matplotlib（绘图）

```bash
pip install numpy numba matplotlib
```

## 运行

```bash
# 完整实验 (约 8 分钟)
python3 simulate.py
```

## 主要结果

- **ε-EXP3**：time-average regret 持续下降趋向 0，确认 no-regret 性质，收敛速率与理论界 O(1/T^{1/(S+1)}) 吻合
- **标准 EXP3**：regret 收敛到 p_min 附近，在多阶段系统中不是 no-regret 策略——因为缺少"教育"机制，子节点无法充分学习
