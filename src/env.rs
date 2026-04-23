use crate::rng::SplitMix64;

#[derive(Clone, Debug)]
pub struct LeafDistribution {
    pub probabilities: Vec<f64>,
    pub change_time: usize,
}

#[derive(Clone, Debug)]
pub struct LeafBanditEnv {
    pub dist: LeafDistribution,
}

impl LeafDistribution {
    pub fn linear(p_min: f64, leaf_count: usize, change_time: usize) -> Self {
        let mut probabilities = Vec::with_capacity(leaf_count);
        if leaf_count == 0 {
            return Self {
                probabilities,
                change_time,
            };
        }
        if leaf_count == 1 {
            probabilities.push(1.0);
            return Self {
                probabilities,
                change_time,
            };
        }

        let step = (1.0 - p_min) / (leaf_count as f64 - 1.0);
        for idx in 0..leaf_count {
            probabilities.push((p_min + step * idx as f64).min(1.0));
        }

        Self {
            probabilities,
            change_time,
        }
    }

    pub fn optimal_expected_cost(&self, t: usize) -> f64 {
        let mut best = f64::INFINITY;
        for (idx, &prob) in self.probabilities.iter().enumerate() {
            let expected = if idx + 1 == self.probabilities.len() {
                let before = self.change_time.min(t) as f64;
                let after = t.saturating_sub(self.change_time) as f64;
                before + 0.0 * after
            } else {
                prob * t as f64
            };
            if expected < best {
                best = expected;
            }
        }
        best
    }
}

impl LeafBanditEnv {
    pub fn new(dist: LeafDistribution) -> Self {
        Self { dist }
    }

    pub fn sample_cost(&self, round: usize, leaf_index: usize, rng: &mut SplitMix64) -> f64 {
        let mut probability = self.dist.probabilities[leaf_index];
        if round >= self.dist.change_time && leaf_index + 1 == self.dist.probabilities.len() {
            probability = 0.0;
        }

        if rng.next_f64() < probability {
            1.0
        } else {
            0.0
        }
    }

    pub fn leaf_count(&self) -> usize {
        self.dist.probabilities.len()
    }
}
