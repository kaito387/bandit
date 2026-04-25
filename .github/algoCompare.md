首先，问题的背景是 multistage-bandit。具体来说，给出一个有 $n$ 个节点的树，所有叶节点都是一个老虎机。我们会玩 $T$ 轮老虎机，每一轮，根节点收到一个请求，它根据自己的 policy 将请求分配给自己的一个孩子，这个孩子也会根据自己的 policy 将请求分配给它的一个孩子，直到这个请求被传递到一个叶子。而叶节点是个老虎机，会产出一个 $c_i$ （$[0, 1]$ 内的实数）表示第 $i$ 轮的 `cost`。

让我们把环境当作一个算法题。

对于每个实验环境，读入一个 `json` 文件，这个 `json` 文件是一个环境设置。

环境设置有这么几个东西：
1. `env_name`，一个 `str`, 表示环境名称
2. `algo`，一个 str 数组，表示本次环境需要运行的算法。
    可选元素如下
    1. `"PS"`: stand for algo 1: Partial Sshare
    2. `"E3"`: stand for algo 2: EXP3
    3. `"EE3"`: stand for algo 3: eps-EXP3
    4. `"R"`: stand for algo 4: random
    5. `"Q"`: stand for algo 5: greedy
    6. `"E3Q"`: stand for algo 6: EXP3-QV
3. `seed`，本环境的随机数种子，保证可复现性
4. `node_counts`，一个 `integer`,表示节点总数
5. `rounds`：即老虎机轮数。
6. `parents`，一个长度为 n - 1 的数组，表示1～n-1号节点的父亲（0号为根,环境均为0-based）
7. $g$：一个长度为 n - 1 的数组，表示 1~n-1 号节点是否 share
8. $p$：一个长度为 n - 1 的数组，表示 1~n-1 号节点的 p。p 的含义只有在该节点是叶子时才有效，意思是这个叶节点的 cost 会采样自一个以 $p$ 为参数的伯努利分布 $c_i\sim\operatorname{Bernoulli}(p)$.

根据这些定义，我们通过对树进行一次遍历，给出

1. `isLeaf[]`：表示一个节点是否是叶子
2. isSafe[]：一个节点 $u$ 是 Safe 的（`isSafe[u] = true`），当且仅当 $\forall v \in C_u, \text{isSafe}_v \wedge \text{isShare}_v$。这里 $C_u$ 表示 $u$ 的所有 children.
3. K：树的深度
4. `d[]`：d[u] 表示节点 $u$ 的孩子数量。
5. S：树中最大分支树（每个节点有的最大孩子数量）即 $S = \max_u d[u]$
6. leafCount[]：`leafCount[u]` 表示节点 $u$ 的子树内，叶子的数量。显然有 `leafCount[u] = isLeaf[u] + sum leafCount[v]`
7. $R[]$：危险系数，递归定义
    `R[u] = [isSafe[u] == 0] + max(R[v])`，`R0 = R[root]`。即当前节点是 Risky 就会使 `R[u] += 1`。
8. needExplore[]: needExplore[u] = 1 if isLeaf[v] = 0 for any child v

然后，我们给出 5 种算法。

## 算法 1：Partial Share

```plain
theta[u] = 0
W[u] = sum exp(theta[l]) for all leaves l in subtree[u]
epsilon = S * power(T, -1 / (K + 1))
eta = power(T, -R / (R + 1))

for t = 1..T do
	u = root, Pi[u] = 1
	while !isLeaf[u]:
		if isSafe[u]:
			p[v] = W[v] / W[u]
		else:
			let real_eps = needExplore[u] ? epsilon : 0
			p[v] = real_eps / d[u] + (1 - real_eps) * soft_max(theta[v])
			
        choose child v with probability p[v]
        Pi[v] = Pi[u] * p[v]
        // foward job to v
        u = v
	END WHILE
	// u is Leaf
	observe cost c[t] from u
	while u is not -1	// fa[root] = -1
		theta[u] -= c[t] / Pi[u]
		u = fa[u]
	ENDWHILE
```

## 算法 2：EXP3

```plain
theta[u] = 0 forall u
eta = sqrt(log(S) / TS)

for t = 1..T do
	u = root
	while !isLeaf[u]:
		p[v] = softmax(theta[v])
        choose child v with probability p[v]
        // foward job to v
        u = v
	END WHILE
	// u is Leaf
	observe cost c[t] from u
	while u is not -1	// fa[root] = -1
		theta[u] -= c[t] / p[v]	// note that here EXP3 updates by local p probability
		u = fa[u]
	ENDWHILE
```

## 算法 3：eps-EXP3

```plain
theta[u] = 0 forall u
epsilon = S * power(T, -1 / (K + 1))
eta = power(T, -K / (K + 1))

for t = 1..T do
	u = root, Pi[u] = 1
	while !isLeaf[u]:
        p[v] = epsilon / d[u] + (1 - epsilon) * softmax(theta[v])
        choose v with probability p[v]
        // foward job to v
        Pi[v] = Pi[u] * p[v]
        u = v
	END WHILE
	// u is Leaf
	observe cost c[t] from u
	while u is not -1	// fa[root] = -1
		theta[u] -= c[t] / Pi[u]
		u = fa[u]
	ENDWHIEL
```

## 算法 4：random

```plain
for t = 1..T do
	u = root
	while !isLeaf[u]:
		p[v] = 1 / d[u]
        choose child v with probability p[v]
        // foward job to v
        u = v
	END WHILE
	observe cost c[t] from u
```

## 算法 5：greedy

```plain
Q[u] = 1 forall u
N[u] = 0 for all u

for t = 1..T do
	u = root
	while !isLeaf[u]:
        choose child v as arg min Q[v]
        // foward job to v
        u = v
	END WHILE
	observe cost c[t] from u
	while u is not -1	// fa[root] = -1
		N[u] = N[u] + 1
		Q[u] = Q[u] + (c[t] - Q[u]) / N[u]
		u = fa[u]
	ENDWHILE
```

## 算法 6：EXP3-QV

```plain
theta[u] = 0 forall u
eta = sqrt(log(S) / TS)

for t = 1..T do
	u = root, Pi[u] = 1
	while !isLeaf[u]:
		p[v] = softmax(theta[v])
        choose child v with probability p[v]
        // foward job to v
		Pi[v] = Pi[u] * p[v]
        u = v
	END WHILE
	// u is Leaf
	observe cost c[t] from u
	while u is not -1	// fa[root] = -1
		theta[u] -= c[t] / Pi[u]
		u = fa[u]
	ENDWHILE
```

---

显然对于不同算法，每一轮的 c[t] 是不同的。

定义 $\mathcal L$ 为叶子节点集合，每一轮最后的 $c[t]$ 来源自 $l[t]$（即第 $t$ 轮的第 1 个 while 结束后令 `l[t] = u`）。

需要观察的记录的指标为：

- $\text{avgCost}=\sum_t^T c[t] / T$
- $\text{bestPath} = \arg\min_{l\in\mathcal  L} p_l$
- $\text{bestPathRate} = \#\{l[t] = \text{bestPath}\mid t\in [T]\} / T$
- $\text{shareRate} = \#\{[\text{isShare}[l[t]] =1] \mid t\in[T]\}$

时间相关的指标：

- $\text{regret}[t] = \text c[t] - \min_{l \in \mathcal L} p_l$
- $\text{accumRegret}[t] = \sum_{\tau=1}^t \text{regret}[\tau]=\operatorname{accumulate}(\text{regret}[])$
- $\text{avgRegret}[t] = \text{accumRegret[t]} / t$

因为算法是随机算法，同一个环境，需要跑 20 轮取平均得到真实指标。

结果输出：

1. CSV 文件, 包含不同算法时间相关的动态指标
2. JSON 文件，包含标量指标和最终统计
3. python 绘图，包含不同算法，每个算法一个折线，绘制 $\text{avgRegret}[t]$ vs $t$ 的图。注意在绘图的时候，前 1% 的数据可能会有较大波动，考虑把前 1% 的数据丢掉，或者在绘图的时候把 $t$ 的范围限制在 $[0.01T, T]$ 上。