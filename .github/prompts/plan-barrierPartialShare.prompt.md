## Plan: Rust 化 Barrier Partial-Share 仿真（按优化版规则）

把当前 Python 的 multistage bandit 仿真重构成 Rust 实现，直接承载优化后的 Barrier Partial-Share 算法。核心目标是统一到节点级状态 `isShare` 与 `isSafe`，并在不区分“叶子 shared 集合”和“gate”的前提下，完成采样、更新、回传和实验驱动迁移。

**Core Rules（本次采用）**
1. 每个节点都有 `isShare`（布尔）。不再单独维护“Shared 叶集合 + gate”。
2. 对节点 `u`，若 `subtree(u)` 中除 `u` 本身外的所有节点都满足 `isShare=true`，则 `u` 是 Safe，否则是 Risky。（等价于孩子 isSafe 和 isShare 均为 true）
3. 对所有节点都维护 `W[i]` 和 `theta[i,j]`；采样时根据 `isSafe` 选择使用 `W` 分支或 `epsilon + softmax(theta)` 分支。
4. 初始化改为 `W[u] = leafCount(subtree(u))`（不再是常数 1）。
5. 回传改为一直 propagate 到 root（不再遇到 unshare 截断）。
6. 参数设置采用一些固定值如 0.01。

**Steps**
1. 搭建 Rust 工程骨架，作为 Python 代码的替代实现，建立清晰模块边界。*depends on current repository shape*
   - 新建 Rust 工程入口与基础配置。
   - 把“树参数、实验参数、输出统计”拆分为独立模块。
   - 预留实验运行入口，保证后续可批量试验。

2. 抽象环境与节点状态。*depends on 1*
   - 定义树、节点、父子关系、叶计数缓存 `leafCount(subtree(u))`。
   - 为所有节点存储 `isShare`、`isSafe`、`W[i]`。
   - 为所有内部节点存储 `theta[i,j]`（按 child 索引）。

3. 实现预处理与初始化。*depends on 2*
   - 自底向上计算每个节点子树是否“全共享”（排除节点自身），据此标记 `isSafe`。
   - 初始化所有节点 `W[u] = leafCount(subtree(u))`。
   - 初始化所有 `theta[i,j] = 0`。
   - 将预处理结果固化为回合可复用状态。

4. 重写每轮采样逻辑（按 `isSafe` 分流）。*depends on 3*
   - 若节点 Safe：按 `p_t(j|i) = W[j] / W[i]` 采样子节点。
   - 若节点 Risky：按 `epsilon_i` 决定 Uniform/EXP3 分支。
   - EXP3 分支使用 `softmax(eta * theta[i,*])`。
   - 记录路径、每层 `p_t`、以及 `Pi_t` 以支持后续更新。

5. 实现更新规则（按优化版统一）。*depends on 4*
   - 路径上所有 Risky 节点按
     `theta[u,v] -= c_t / Pi_t(v)` 更新。
   - 叶节点到根执行统一回传：
     - 先对叶相关状态求 `Delta_t`（由新旧值差得到）。
     - 将 `Delta_t` 沿祖先链一直传播到 root。
   - 因为现在所有节点都有 `W`，回传时同步更新沿途 `W`（具体映射规则按实现中的一致公式）。

6. 迁移实验驱动到 Rust，并保持输出可对照。*depends on 1-5*
   - 复用现有实验配置组织方式，支持多组 `K, S, p_min, T`。
   - 输出累计 cost / regret 等统计，方便对照旧实现与论文趋势。
   - 图形可先后置，优先确保数据导出可用。

7. 文档与配置收敛。*depends on 6*
   - 更新 README，说明 `isShare`/`isSafe` 语义、初始化和更新规则。
   - 标注 Python 旧实现定位，避免入口歧义。
   - 视需要收紧 Python 依赖，避免误导为主执行路径。

**Relevant files**
- `/home/lht/dev/research/bandit/simulate.py` — 当前行为参考基线。
- `/home/lht/dev/research/bandit/main.py` — 现占位入口。
- `/home/lht/dev/research/bandit/README.md` — 需同步新语义。
- `/home/lht/dev/research/bandit/pyproject.toml` — Python 元数据与依赖。
- `/home/lht/dev/research/bandit/Cargo.toml` — 计划新增 Rust 工程入口。
- `/home/lht/dev/research/bandit/src/main.rs` — 计划新增 Rust CLI 入口。
- `/home/lht/dev/research/bandit/src/lib.rs` — 计划新增算法核心模块。

**Verification**
1. 小树样例验证 `isSafe` 判定是否严格符合“子树除自身全 `isShare=true`”。
2. 验证初始化：所有节点 `W[u]` 等于其子树叶子数量。
3. 固定随机种子检查采样路径、`Pi_t`、以及 risky 节点分母项是否一致。
4. 验证回传确实总是到 root，且沿途 `W` 更新与 `Delta_t` 一致。
5. 跑一组小型对照实验，确认累计 cost / regret 数值稳定且趋势合理。

**Decisions**
- Rust 作为主实现路径，Python 不再是主执行。
- 语义统一为 `isShare`/`isSafe`，不再保留“Shared 叶集合 + gate”双轨定义。
- 所有节点统一维护 `W`，所有内部节点统一维护 `theta`。
- Back-propagation 固定为传播到 root。