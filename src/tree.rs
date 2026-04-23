#[derive(Clone, Debug)]
pub struct TreeConfig {
    pub branching: usize,
    pub depth: usize,
    pub shared_path: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub index: usize,
    pub level: usize,
    pub path: Vec<usize>,
    pub parent: Option<usize>,
    pub child_index: Option<usize>,
    pub children: Vec<usize>,
    pub is_share: bool,
    pub is_safe: bool,
    pub leaf_count: usize,
    pub w: f64,
    pub theta: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub config: TreeConfig,
    pub nodes: Vec<Node>,
    pub root: usize,
}

impl TreeConfig {
    pub fn new(branching: usize, depth: usize, shared_path: Vec<usize>) -> Self {
        Self {
            branching,
            depth,
            shared_path,
        }
    }
}

impl Tree {
    pub fn new(config: TreeConfig) -> Self {
        assert!(config.branching > 0, "branching must be positive");
        let mut tree = Self {
            config,
            nodes: Vec::new(),
            root: 0,
        };
        tree.build();
        tree.compute_cached_state();
        tree
    }

    fn build(&mut self) {
        let root = self.build_node(0, Vec::new(), None, None);
        self.root = root;
    }

    fn build_node(
        &mut self,
        level: usize,
        path: Vec<usize>,
        parent: Option<usize>,
        child_index: Option<usize>,
    ) -> usize {
        let index = self.nodes.len();
        let is_share = path.starts_with(&self.config.shared_path);
        let is_leaf = level == self.config.depth;
        let theta = if is_leaf {
            Vec::new()
        } else {
            vec![0.0; self.config.branching]
        };

        self.nodes.push(Node {
            index,
            level,
            path: path.clone(),
            parent,
            child_index,
            children: Vec::new(),
            is_share,
            is_safe: false,
            leaf_count: 0,
            w: 0.0,
            theta,
        });

        if level < self.config.depth {
            for child in 0..self.config.branching {
                let mut next_path = path.clone();
                next_path.push(child);
                let child_index_in_tree = self.build_node(
                    level + 1,
                    next_path,
                    Some(index),
                    Some(child),
                );
                self.nodes[index].children.push(child_index_in_tree);
            }
        }

        index
    }

    fn compute_cached_state(&mut self) {
        for index in (0..self.nodes.len()).rev() {
            if self.nodes[index].children.is_empty() {
                self.nodes[index].leaf_count = 1;
                self.nodes[index].is_safe = true;
            } else {
                let mut leaf_count = 0;
                let mut safe = true;
                for &child in &self.nodes[index].children {
                    leaf_count += self.nodes[child].leaf_count;
                    safe &= self.nodes[child].is_share && self.nodes[child].is_safe;
                }
                self.nodes[index].leaf_count = leaf_count;
                self.nodes[index].is_safe = safe;
            }
        }

        for node in &mut self.nodes {
            node.w = node.leaf_count as f64;
        }
    }

    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    pub fn node_mut(&mut self, index: usize) -> &mut Node {
        &mut self.nodes[index]
    }

    pub fn leaf_count(&self, index: usize) -> usize {
        self.nodes[index].leaf_count
    }

    pub fn depth(&self) -> usize {
        self.config.depth
    }

    pub fn branching(&self) -> usize {
        self.config.branching
    }

    pub fn path_to_root(&self, mut index: usize) -> Vec<usize> {
        let mut path = Vec::new();
        loop {
            path.push(index);
            match self.nodes[index].parent {
                Some(parent) => index = parent,
                None => break,
            }
        }
        path.reverse();
        path
    }

    pub fn child_position(&self, index: usize) -> Option<usize> {
        self.nodes[index].child_index
    }

    pub fn subtree_root_from_path(&self, path: &[usize]) -> Option<usize> {
        let mut current = self.root;
        for &child in path {
            if child >= self.branching() {
                return None;
            }
            current = *self.nodes[current].children.get(child)?;
        }
        Some(current)
    }
}
