分支数记为 $K$，最优路径长度记为 $D$。

---

### 1. 设定与记号

在时刻 $t$，算法采样一条从根到叶的路径 $A_t$。定义指示变量

$$
A_{ti}=\mathbf 1\{i\in A_t\},\qquad
P_{ti}=\mathbb P(i\in A_t\mid \mathcal H_{t-1}).
$$

若 $j\in S(i)$（$j$ 是 $i$ 的子节点），定义

$$
\pi_t(j\mid i)=\mathbb P(j\in A_t\mid i\in A_t,\mathcal H_{t-1}),
$$

于是

$$
P_{tj}=P_{ti}\,\pi_t(j\mid i).
$$

定义“从节点 $i$ 出发继续按算法策略走到叶子”的条件期望奖励：

$$
\bar x_{ti}=
\begin{cases}
x_{ti}, & i\text{ 是叶子},\\
\sum_{j\in S(i)}\pi_t(j\mid i)\,\bar x_{tj}, & i\text{ 是内部节点}.
\end{cases}
$$

记叶子节点提供的 reward 为 $X_t\in[0,1]$。

---

### 2. 无偏估计

考虑两种等价估计：

$$
\hat X_{ti}^{(g)}=\frac{A_{ti}}{P_{ti}}X_t,
\qquad
\hat X_{ti}^{(qv)}=1-\frac{A_{ti}}{P_{ti}}(1-X_t).
$$

#### 2.1 Lemma 1（无偏性）

$$
\mathbb E_{t-1}[\hat X_{ti}^{(g)}]=\bar x_{ti},
\qquad
\mathbb E_{t-1}[\hat X_{ti}^{(qv)}]=\bar x_{ti}.
$$

证明如下：

对第一式

$$
\mathbb E_{t-1}\!\left[\frac{A_{ti}}{P_{ti}}X_t\right]
=\frac{1}{P_{ti}}\mathbb E_{t-1}[A_{ti}X_t]
=\mathbb E_{t-1}[X_t\mid i\in A_t]=\bar x_{ti}.
$$

第二式由第一式直接得到

$$
\mathbb E_{t-1}\!\left[1-\frac{A_{ti}}{P_{ti}}(1-X_t)\right]
=1-\bigl(1-\bar x_{ti}\bigr)
=\bar x_{ti}.
$$

$\square$

---

### 3. regret 裂项

设最优叶子为 $i^*$，其路径为

$$
p_0^*=r,\ p_1^*,\dots,p_D^*=i^*.
$$

定义 weak regret

$$
R_n:=\sum_{t=1}^nx_{t i^*}-\mathbb E\!\left[\sum_{t=1}^n X_t\right] = \mathbb E\!\left[\sum_{t=1}^n \bigl(x_{t i^*} - X_t\bigr)\right].
$$

又因为根节点 $t$ 满足 $\bar x_{tr}=\mathbb E_{t-1}[X_t]$，由塔式法则 $\mathbb E[X_t]=\mathbb E[\bar x_{tr}]$，因此

$$
\begin{aligned}
R_n
&=\sum_{t=1}^n\mathbb E\bigl[\bar x_{t i^*}-\bar x_{tr}\bigr]\\
&=\sum_{k=1}^D\sum_{t=1}^n
\mathbb E\bigl[\bar x_{t p_k^*}-\bar x_{t p_{k-1}^*}\bigr].
\end{aligned}
$$

所以只需 bound 每一层 $(p_{k-1}^*,p_k^*)$ 的项。

---

### 4. 单层 bound

固定某一层，记父节点 $u:=p_{k-1}^*$，目标子节点（最优路径上的那个）为 $u^+:=p_k^*$。

设 $S(u)=\{1,\dots,K\}$。在时刻 $t$，节点 $u$ 采用分布 $\pi_t(\cdot\mid u)$ 选一个子节点。

定义 qv 估计量

$$
\hat X_{t j}:=1-\frac{A_{tj}}{P_{tj}}(1-X_t),\qquad j\in S(u).
$$

定义累计估计回报

$$
\hat S_{t,j}:=\sum_{s=1}^t \hat X_{s j},\qquad \hat S_{0,j}=0,
$$

并定义 policy 的分布 $\pi_t$

$$
w_{t,j}:=\exp(\eta \hat S_{t-1,j}),
\qquad
\pi_t(j\mid u)=\frac{w_{t,j}}{\sum_{m=1}^K w_{t,m}}.
$$

#### 4.1 势函数与下界

定义势函数

$$
W_t:=\sum_{j=1}^K \exp(\eta \hat S_{t,j}).
$$

由于 $W_0=K$，且对任意固定动作 $j$

有 $W_n\ge \exp(\eta \hat S_{n,j})$，故

$$
\log\frac{W_n}{W_0}
\ge
\eta \hat S_{n,j}-\log K.
$$

#### 4.2 势函数与上界

注意

$$
\frac{W_t}{W_{t-1}}
=\sum_{j=1}^K \pi_t(j\mid u)\exp(\eta\hat X_{t j}).
$$


显然 $\eta\hat X_{t j}\le 1$（如果设置的 $0 \le \eta \le 1$），利用初等不等式，$x\le 1$ 时，$e^x\le 1+x+x^2$，则

$$
\exp(\eta\hat X_{t j})
\le
1+\eta\hat X_{t j}+\eta^2\hat X_{t j}^2.
$$

因此

$$
\frac{W_t}{W_{t-1}}
\le
1+\eta\sum_j\pi_t(j\mid u)\hat X_{t j}
+\eta^2\sum_j\pi_t(j\mid u)\hat X_{t j}^2.
$$

两边取对数，并对右端使用初等不等式 $\log(1+y)\le y$，有

$$
\log \frac{W_t}{W_{t-1}} \le \eta\sum_j\pi_t(j\mid u)\hat X_{t j}
+\eta^2\sum_j\pi_t(j\mid u)\hat X_{t j}^2.
$$


对 $t$（$t=1,\dots,n$）求和得到：

$$
\log\frac{W_n}{W_0}
\le
\eta\sum_{t=1}^n\sum_j\pi_t(j\mid u)\hat X_{t j}
+\eta^2\sum_{t=1}^n\sum_j\pi_t(j\mid u)\hat X_{t j}^2.
$$

#### 4.3 合并上下界

把 4.1 和 4.2 合并，对任意固定 $j\in S(u)$：

$$
\begin{aligned}
\eta \hat S_{n j} - \log K
&\le
\eta\sum_{t=1}^n\sum_m\pi_t(m\mid u)\hat X_{t m} +\eta^2\sum_{t=1}^n\sum_m\pi_t(m\mid u)\hat X_{t m}^2\\
\sum_{t=1}^n \hat X_{tj} - \sum_{t=1}^n\sum_m\pi_t(m\mid u)\hat X_{t m}
&\le \frac{\log K}{\eta} +
\eta\sum_{t=1}^n\sum_m\pi_t(m\mid u)\hat X_{t m}^2
\end{aligned}
$$

对上面的式子取 $j=u^+$。并对 $\mathcal H_{t-1}$ 取条件期望，利用 Lemma 1：

$$
\mathbb E_{t-1}[\hat X_{t u^+}]=\bar x_{t u^+},
\qquad
\mathbb E_{t-1}\!\left[\sum_m\pi_t(m\mid u)\hat X_{t m}\right]
=\sum_m\pi_t(m\mid u)\bar x_{t m}=\bar x_{t u}.
$$

于是不等式左边变为 $\bar x_{t u^+}-\bar x_{t u}$。

#### 4.4 二阶矩项的上界

