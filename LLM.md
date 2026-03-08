# LLM Context: Multi-Stage Bandit Problem

## Problem

A job flows through a **multi-stage system** (modeled as a K-ary tree of depth S+1). At each internal node, an agent picks which child to forward the job to. Only the **leaf node** generates a cost (in [0,1]), and only nodes on the chosen path observe this end-to-end cost (**bandit feedback**). Agents act independently with no knowledge of others' actions.

The goal: design a **distributed no-regret policy** — each agent independently decides its action, and the total expected cost over T rounds is sublinear-regret compared to the best fixed root-to-leaf path.

## Key Insight

Beyond the classical exploration-exploitation tradeoff, multi-stage systems introduce a third challenge: **education** — a parent must send enough jobs to each child so the child can learn, even if that child currently looks bad (it may simply not have learned yet).

Standard EXP3 fails here: once a parent stops sending jobs to a child, that child can never improve, creating a vicious cycle. Regret stays Θ(pmin) forever.

## Solution: ε-EXP3 (Algorithm 2)

Each node runs a two-mode policy:
- **Uniform mode** (prob π_v): pick a child uniformly at random → ensures education
- **EXP3 mode** (prob 1−π_v): pick via exponential weights → balances exploration/exploitation

Parameters (for tree depth S+1, max branching K):
- Learning rate: η = T^{−S/(S+1)}
- Education probability: π_v = T^{−1/(S+1)} if children are non-leaf, 0 otherwise

**Regret bound**: O(T^{S/(S+1)}), i.e., time-average regret → 0.

## Experiment (Section 7.1)

- K-ary tree, leaves have Bernoulli costs with params linearly spaced from pmin to 1.0
- At t = T/100, the worst leaf (p=1.0) switches to p=0 (becomes optimal)
- Compare ε-EXP3 vs standard EXP3 over T=10^7 rounds, 20 runs
- Result: ε-EXP3 regret decays as O(1/T^{1/(S+1)}); EXP3 regret plateaus at pmin
