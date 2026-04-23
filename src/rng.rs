#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    pub fn gen_range_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    pub fn sample_weighted(&mut self, weights: &[f64]) -> (usize, f64) {
        let mut total = 0.0;
        for &weight in weights {
            total += weight.max(0.0);
        }

        if total <= f64::EPSILON {
            let idx = self.gen_range_usize(weights.len());
            return (idx, 1.0 / weights.len() as f64);
        }

        let draw = self.next_f64() * total;
        let mut accum = 0.0;
        for (idx, &weight) in weights.iter().enumerate() {
            accum += weight.max(0.0);
            if draw <= accum || idx + 1 == weights.len() {
                return (idx, weight.max(0.0) / total);
            }
        }

        (weights.len() - 1, weights[weights.len() - 1].max(0.0) / total)
    }
}

pub fn softmax(logits: &[f64], eta: f64) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let mut max_logit = logits[0];
    for &value in &logits[1..] {
        if value > max_logit {
            max_logit = value;
        }
    }

    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &value in logits {
        let weight = ((eta * (value - max_logit)).max(-700.0)).exp();
        probs.push(weight);
        sum += weight;
    }

    if sum <= f64::EPSILON {
        let uniform = 1.0 / logits.len() as f64;
        return vec![uniform; logits.len()];
    }

    for value in &mut probs {
        *value /= sum;
    }

    probs
}
