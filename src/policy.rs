use crate::rng::softmax;
use crate::rng::SplitMix64;
use crate::tree::{Tree, TreeConfig};

const EPSILON_FLOOR: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeMode {
    Safe,
    RiskyUniform,
    RiskySoftmax,
}

#[derive(Clone, Debug)]
pub struct RoundTrace {
    pub path_nodes: Vec<usize>,
    pub chosen_children: Vec<usize>,
    pub node_modes: Vec<NodeMode>,
    pub reach_probs: Vec<f64>,
    pub choice_probs: Vec<f64>,
    pub leaf_node_index: usize,
    pub leaf_position: usize,
}

#[derive(Clone, Debug)]
pub struct BarrierPartialSharePolicy {
    pub tree: Tree,
    pub epsilon: f64,
    pub eta: f64,
    pub rng: SplitMix64,
}

#[derive(Clone, Debug)]
pub struct EpsilonExp3Policy {
    pub tree: Tree,
    pub epsilon: f64,
    pub eta: f64,
    pub rng: SplitMix64,
}

impl BarrierPartialSharePolicy {
    pub fn new(config: TreeConfig, epsilon: f64, eta: f64, seed: u64) -> Self {
        Self {
            tree: Tree::new(config),
            epsilon,
            eta,
            rng: SplitMix64::new(seed),
        }
    }

    pub fn sample_round(&mut self) -> RoundTrace {
        let mut path_nodes = Vec::with_capacity(self.tree.depth() + 1);
        let mut chosen_children = Vec::with_capacity(self.tree.depth());
        let mut node_modes = Vec::with_capacity(self.tree.depth());
        let mut reach_probs = Vec::with_capacity(self.tree.depth() + 1);
        let mut choice_probs = Vec::with_capacity(self.tree.depth());
        let mut leaf_position = 0usize;

        let mut node_index = self.tree.root;
        let mut reach_prob = 1.0;
        reach_probs.push(reach_prob);

        for _level in 0..self.tree.depth() {
            path_nodes.push(node_index);
            let node = self.tree.node(node_index).clone();
            let (chosen_child, chosen_prob, mode) = self.choose_child(&node);
            chosen_children.push(chosen_child);
            node_modes.push(mode);
            choice_probs.push(chosen_prob);
            leaf_position = leaf_position * self.tree.branching() + chosen_child;

            node_index = node.children[chosen_child];
            reach_prob *= chosen_prob;
            reach_probs.push(reach_prob);
        }

        path_nodes.push(node_index);

        RoundTrace {
            path_nodes,
            chosen_children,
            node_modes,
            reach_probs,
            choice_probs,
            leaf_node_index: node_index,
            leaf_position,
        }
    }

    fn choose_child(&mut self, node: &crate::tree::Node) -> (usize, f64, NodeMode) {
        let branching = node.children.len();

        if node.is_safe {
            let mut weights = Vec::with_capacity(branching);
            for &child_index in &node.children {
                weights.push(self.tree.node(child_index).w.max(0.0));
            }
            let total: f64 = weights.iter().sum();
            if total <= EPSILON_FLOOR {
                let chosen = self.rng.gen_range_usize(branching);
                return (chosen, 1.0 / branching as f64, NodeMode::Safe);
            }
            let (chosen, _) = self.rng.sample_weighted(&weights);
            return (chosen, weights[chosen] / total, NodeMode::Safe);
        }

        if self.rng.next_f64() < self.epsilon {
            let chosen = self.rng.gen_range_usize(branching);
            return (chosen, 1.0 / branching as f64, NodeMode::RiskyUniform);
        }

        let probs = softmax(&node.theta, self.eta);
        let (chosen, chosen_prob) = self.rng.sample_weighted(&probs);
        (chosen, chosen_prob.max(EPSILON_FLOOR), NodeMode::RiskySoftmax)
    }

    pub fn apply_observation(&mut self, trace: &RoundTrace, cost: f64) {
        let reward = 1.0 - cost;

        for &node_index in &trace.path_nodes {
            self.tree.node_mut(node_index).w += reward;
        }

        for (level, mode) in trace.node_modes.iter().enumerate() {
            if matches!(mode, NodeMode::RiskyUniform | NodeMode::RiskySoftmax) && cost > 0.0 {
                let node_index = trace.path_nodes[level];
                let child_slot = trace.chosen_children[level];
                let reach_prob = trace.reach_probs[level + 1].max(EPSILON_FLOOR);
                let node = self.tree.node_mut(node_index);
                node.theta[child_slot] -= cost / reach_prob;
            }
        }
    }
}

impl EpsilonExp3Policy {
    pub fn new(config: TreeConfig, epsilon: f64, eta: f64, seed: u64) -> Self {
        Self {
            tree: Tree::new(config),
            epsilon,
            eta,
            rng: SplitMix64::new(seed),
        }
    }

    pub fn sample_round(&mut self) -> RoundTrace {
        let mut path_nodes = Vec::with_capacity(self.tree.depth() + 1);
        let mut chosen_children = Vec::with_capacity(self.tree.depth());
        let mut node_modes = Vec::with_capacity(self.tree.depth());
        let mut reach_probs = Vec::with_capacity(self.tree.depth() + 1);
        let mut choice_probs = Vec::with_capacity(self.tree.depth());
        let mut leaf_position = 0usize;

        let mut node_index = self.tree.root;
        let mut reach_prob = 1.0;
        reach_probs.push(reach_prob);

        for _level in 0..self.tree.depth() {
            path_nodes.push(node_index);
            let node = self.tree.node(node_index).clone();
            let (chosen_child, chosen_prob, mode) = self.choose_child(&node);
            chosen_children.push(chosen_child);
            node_modes.push(mode);
            choice_probs.push(chosen_prob);
            leaf_position = leaf_position * self.tree.branching() + chosen_child;

            node_index = node.children[chosen_child];
            reach_prob *= chosen_prob;
            reach_probs.push(reach_prob);
        }

        path_nodes.push(node_index);

        RoundTrace {
            path_nodes,
            chosen_children,
            node_modes,
            reach_probs,
            choice_probs,
            leaf_node_index: node_index,
            leaf_position,
        }
    }

    fn choose_child(&mut self, node: &crate::tree::Node) -> (usize, f64, NodeMode) {
        let branching = node.children.len();

        if self.rng.next_f64() < self.epsilon {
            let chosen = self.rng.gen_range_usize(branching);
            return (chosen, 1.0 / branching as f64, NodeMode::RiskyUniform);
        }

        let probs = softmax(&node.theta, self.eta);
        let (chosen, chosen_prob) = self.rng.sample_weighted(&probs);
        (chosen, chosen_prob.max(EPSILON_FLOOR), NodeMode::RiskySoftmax)
    }

    pub fn apply_observation(&mut self, trace: &RoundTrace, cost: f64) {
        let reward = 1.0 - cost;

        for &node_index in &trace.path_nodes {
            self.tree.node_mut(node_index).w += reward;
        }

        for (level, _) in trace.node_modes.iter().enumerate() {
            if cost > 0.0 {
                let node_index = trace.path_nodes[level];
                let child_slot = trace.chosen_children[level];
                let reach_prob = trace.reach_probs[level + 1].max(EPSILON_FLOOR);
                let node = self.tree.node_mut(node_index);
                node.theta[child_slot] -= cost / reach_prob;
            }
        }
    }
}
