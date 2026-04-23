use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::env::{LeafBanditEnv, LeafDistribution};
use crate::policy::{BarrierPartialSharePolicy, EpsilonExp3Policy};
use crate::tree::TreeConfig;

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub branching: usize,
    pub depth: usize,
    pub p_min: f64,
    pub rounds: usize,
    pub runs: usize,
    pub change_ratio: f64,
    pub sample_points: usize,
    pub epsilon: f64,
    pub eta: f64,
    pub shared_path: Vec<usize>,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    pub output_dir: PathBuf,
    pub configs: Vec<RunConfig>,
}

#[derive(Clone, Debug)]
pub struct ExperimentOutcome {
    pub config: RunConfig,
    pub sample_times: Vec<usize>,
    pub mean_costs: Vec<f64>,
    pub mean_regrets: Vec<f64>,
    pub std_regrets: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CompareOutcome {
    pub config: RunConfig,
    pub sample_times: Vec<usize>,
    pub barrier_mean_regrets: Vec<f64>,
    pub epsilon_mean_regrets: Vec<f64>,
}

fn linspace_rounds(rounds: usize, sample_points: usize) -> Vec<usize> {
    if sample_points == 0 {
        return Vec::new();
    }
    if sample_points == 1 {
        return vec![rounds.max(1)];
    }

    let mut sample_times = Vec::with_capacity(sample_points);
    for idx in 0..sample_points {
        let numerator = (idx + 1) * rounds;
        let mut sample = (numerator + sample_points - 1) / sample_points;
        if sample == 0 {
            sample = 1;
        }
        if sample > rounds {
            sample = rounds;
        }
        if sample_times.last().copied() != Some(sample) {
            sample_times.push(sample);
        }
    }
    if sample_times.last().copied() != Some(rounds) {
        sample_times.push(rounds);
    }
    sample_times
}

pub fn build_leaf_probabilities(branching: usize, depth: usize, p_min: f64) -> Vec<f64> {
    let leaf_count = branching.pow(depth as u32);
    LeafDistribution::linear(p_min, leaf_count, 1).probabilities
}

fn optimal_cost_at(round: usize, leaf_probs: &[f64], change_time: usize) -> f64 {
    let mut best = f64::INFINITY;
    for (index, &prob) in leaf_probs.iter().enumerate() {
        let expected = if index + 1 == leaf_probs.len() {
            let before = change_time.min(round) as f64;
            let after = round.saturating_sub(change_time) as f64;
            before + after * 0.0
        } else {
            prob * round as f64
        };
        if expected < best {
            best = expected;
        }
    }
    best
}

pub fn run_experiments(experiment: &ExperimentConfig) -> std::io::Result<Vec<ExperimentOutcome>> {
    fs::create_dir_all(&experiment.output_dir)?;

    let mut outcomes = Vec::with_capacity(experiment.configs.len());
    for config in &experiment.configs {
        outcomes.push(run_single_config(config)?);
    }

    let mut csv = String::new();
    csv.push_str("config_index,sample_round,mean_cost,mean_regret,std_regret\n");
    for (config_index, outcome) in outcomes.iter().enumerate() {
        for idx in 0..outcome.sample_times.len() {
            let _ = writeln!(
                &mut csv,
                "{},{},{:.6},{:.6},{:.6}",
                config_index,
                outcome.sample_times[idx],
                outcome.mean_costs[idx],
                outcome.mean_regrets[idx],
                outcome.std_regrets[idx]
            );
        }
    }

    let output_file = experiment.output_dir.join("barrier_partial_share.csv");
    let mut file = fs::File::create(output_file)?;
    file.write_all(csv.as_bytes())?;

    let plot_file = experiment.output_dir.join("barrier_partial_share_regret.svg");
    plot_regret_curves(&plot_file, &outcomes)?;

    Ok(outcomes)
}

pub fn run_comparison(experiment: &ExperimentConfig) -> std::io::Result<Vec<CompareOutcome>> {
    fs::create_dir_all(&experiment.output_dir)?;

    let mut outcomes = Vec::with_capacity(experiment.configs.len());
    for config in &experiment.configs {
        outcomes.push(run_compare_single_config(config)?);
    }

    let mut csv = String::new();
    csv.push_str("config_index,sample_round,barrier_mean_regret,epsilon_mean_regret\n");
    for (config_index, outcome) in outcomes.iter().enumerate() {
        for idx in 0..outcome.sample_times.len() {
            let _ = writeln!(
                &mut csv,
                "{},{},{:.6},{:.6}",
                config_index,
                outcome.sample_times[idx],
                outcome.barrier_mean_regrets[idx],
                outcome.epsilon_mean_regrets[idx]
            );
        }
    }

    let output_file = experiment.output_dir.join("epsilon_exp3_compare.csv");
    let mut file = fs::File::create(output_file)?;
    file.write_all(csv.as_bytes())?;

    let plot_file = experiment.output_dir.join("epsilon_exp3_compare.svg");
    plot_compare_curves(&plot_file, &outcomes)?;

    Ok(outcomes)
}

fn plot_regret_curves(path: &Path, outcomes: &[ExperimentOutcome]) -> std::io::Result<()> {
    if outcomes.is_empty() {
        return Ok(());
    }

    let width = 1280.0;
    let height = 720.0;
    let margin_left = 90.0;
    let margin_right = 30.0;
    let margin_top = 60.0;
    let margin_bottom = 90.0;
    let plot_width = width - margin_left - margin_right;
    let plot_height = height - margin_top - margin_bottom;

    let mut max_round = 1usize;
    let mut max_regret = 0.0f64;
    for outcome in outcomes {
        if let Some(&last_round) = outcome.sample_times.last() {
            max_round = max_round.max(last_round);
        }
        for &regret in &outcome.mean_regrets {
            if regret.is_finite() {
                max_regret = max_regret.max(regret);
            }
        }
    }

    let x_max = max_round as f64;
    let y_max = (max_regret * 1.2).max(1e-6);
    let palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#17becf", "#111111", "#bcbd22", "#ff7f0e"];

    let mut svg = String::new();
    write!(
        &mut svg,
        "<svg xmlns='http://www.w3.org/2000/svg' width='{:.0}' height='{:.0}' viewBox='0 0 {:.0} {:.0}'>",
        width, height, width, height
    )
    .expect("write svg header");
    svg.push_str("<rect width='100%' height='100%' fill='white'/>");
    svg.push_str("<text x='640' y='32' text-anchor='middle' font-family='sans-serif' font-size='28' fill='#111'>Reg_T / T vs T</text>");

    let x0 = margin_left;
    let y0 = height - margin_bottom;
    let x1 = width - margin_right;
    let y1 = margin_top;

    svg.push_str(&format!(
        "<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#111' stroke-width='2'/>"
    ));
    svg.push_str(&format!(
        "<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#111' stroke-width='2'/>"
    ));

    for tick in 0..=5 {
        let ratio = tick as f64 / 5.0;
        let x = x0 + plot_width * ratio;
        let round_value = (x_max * ratio).max(1.0);
        svg.push_str(&format!(
            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='#bbb' stroke-width='1' stroke-dasharray='4 4'/>",
            x, y0, x, y1
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' text-anchor='middle' font-family='sans-serif' font-size='14' fill='#333'>{:.0}</text>",
            x, y0 + 24.0, round_value
        ));
    }

    for tick in 0..=5 {
        let ratio = tick as f64 / 5.0;
        let y = y0 - plot_height * ratio;
        let value = y_max * ratio;
        svg.push_str(&format!(
            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='#bbb' stroke-width='1' stroke-dasharray='4 4'/>",
            x0, y, x1, y
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' text-anchor='end' font-family='sans-serif' font-size='14' fill='#333'>{:.3}</text>",
            x0 - 10.0, y + 5.0, value
        ));
    }

    svg.push_str(&format!(
        "<text x='{:.1}' y='{:.1}' text-anchor='middle' font-family='sans-serif' font-size='16' fill='#111'>T</text>",
        (x0 + x1) / 2.0,
        height - 18.0
    ));
    svg.push_str(&format!(
        "<text x='24' y='{:.1}' text-anchor='middle' font-family='sans-serif' font-size='16' fill='#111' transform='rotate(-90 24,{:.1})'>Reg_T / T</text>",
        (y0 + y1) / 2.0,
        (y0 + y1) / 2.0
    ));

    for (index, outcome) in outcomes.iter().enumerate() {
        let color = palette[index % palette.len()];
        let label = format!(
            "K={}, S={}, p_min={:.2}",
            outcome.config.branching, outcome.config.depth, outcome.config.p_min
        );
        let mut points = String::new();
        for (&round, &regret) in outcome.sample_times.iter().zip(outcome.mean_regrets.iter()) {
            let x = x0 + ((round as f64 / x_max).clamp(0.0, 1.0) * plot_width);
            let y = y0 - ((regret / y_max).clamp(0.0, 1.0) * plot_height);
            let _ = write!(&mut points, "{:.1},{:.1} ", x, y);
        }
        svg.push_str(&format!(
            "<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}'/>",
        ));
        let legend_y = margin_top + 18.0 + 24.0 * index as f64;
        let legend_x = width - 330.0;
        svg.push_str(&format!(
            "<rect x='{:.1}' y='{:.1}' width='18' height='4' fill='{color}'/>",
            legend_x, legend_y - 10.0
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' font-family='sans-serif' font-size='14' fill='#111'>{}</text>",
            legend_x + 26.0,
            legend_y,
            escape_xml(&label)
        ));
    }

    svg.push_str("</svg>");

    fs::write(path, svg)?;
    Ok(())
}

fn plot_compare_curves(path: &Path, outcomes: &[CompareOutcome]) -> std::io::Result<()> {
    if outcomes.is_empty() {
        return Ok(());
    }

    let width = 1280.0;
    let height = 720.0;
    let margin_left = 90.0;
    let margin_right = 30.0;
    let margin_top = 60.0;
    let margin_bottom = 90.0;
    let plot_width = width - margin_left - margin_right;
    let plot_height = height - margin_top - margin_bottom;

    let mut max_round = 1usize;
    let mut max_regret = 0.0f64;
    for outcome in outcomes {
        if let Some(&last_round) = outcome.sample_times.last() {
            max_round = max_round.max(last_round);
        }
        for &regret in outcome
            .barrier_mean_regrets
            .iter()
            .chain(outcome.epsilon_mean_regrets.iter())
        {
            if regret.is_finite() {
                max_regret = max_regret.max(regret);
            }
        }
    }

    let x_max = max_round as f64;
    let y_max = (max_regret * 1.2).max(1e-6);
    let mut svg = String::new();
    write!(
        &mut svg,
        "<svg xmlns='http://www.w3.org/2000/svg' width='{:.0}' height='{:.0}' viewBox='0 0 {:.0} {:.0}'>",
        width, height, width, height
    )
    .expect("write svg header");
    svg.push_str("<rect width='100%' height='100%' fill='white'/>");
    svg.push_str("<text x='640' y='32' text-anchor='middle' font-family='sans-serif' font-size='28' fill='#111'>Reg_T / T vs T</text>");

    let x0 = margin_left;
    let y0 = height - margin_bottom;
    let x1 = width - margin_right;
    let y1 = margin_top;

    svg.push_str(&format!(
        "<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#111' stroke-width='2'/>"
    ));
    svg.push_str(&format!(
        "<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#111' stroke-width='2'/>"
    ));

    for tick in 0..=5 {
        let ratio = tick as f64 / 5.0;
        let x = x0 + plot_width * ratio;
        let round_value = (x_max * ratio).max(1.0);
        svg.push_str(&format!(
            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='#bbb' stroke-width='1' stroke-dasharray='4 4'/>",
            x, y0, x, y1
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' text-anchor='middle' font-family='sans-serif' font-size='14' fill='#333'>{:.0}</text>",
            x, y0 + 24.0, round_value
        ));
    }

    for tick in 0..=5 {
        let ratio = tick as f64 / 5.0;
        let y = y0 - plot_height * ratio;
        let value = y_max * ratio;
        svg.push_str(&format!(
            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='#bbb' stroke-width='1' stroke-dasharray='4 4'/>",
            x0, y, x1, y
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' text-anchor='end' font-family='sans-serif' font-size='14' fill='#333'>{:.3}</text>",
            x0 - 10.0, y + 5.0, value
        ));
    }

    svg.push_str(&format!(
        "<text x='{:.1}' y='{:.1}' text-anchor='middle' font-family='sans-serif' font-size='16' fill='#111'>T</text>",
        (x0 + x1) / 2.0,
        height - 18.0
    ));
    svg.push_str(&format!(
        "<text x='24' y='{:.1}' text-anchor='middle' font-family='sans-serif' font-size='16' fill='#111' transform='rotate(-90 24,{:.1})'>Reg_T / T</text>",
        (y0 + y1) / 2.0,
        (y0 + y1) / 2.0
    ));

    let comparison_styles = [("Barrier Partial-Share", "#1f77b4"), ("epsilon-EXP3", "#d62728")];
    for (index, outcome) in outcomes.iter().enumerate() {
        let legend_base_y = margin_top + 18.0 + 26.0 * (index as f64 * 2.0);
        let legend_x = width - 330.0;

        for (series_index, (series_label, color)) in comparison_styles.iter().enumerate() {
            let regrets = if series_index == 0 {
                &outcome.barrier_mean_regrets
            } else {
                &outcome.epsilon_mean_regrets
            };
            let mut points = String::new();
            for (&round, &regret) in outcome.sample_times.iter().zip(regrets.iter()) {
                let x = x0 + ((round as f64 / x_max).clamp(0.0, 1.0) * plot_width);
                let y = y0 - ((regret / y_max).clamp(0.0, 1.0) * plot_height);
                let _ = write!(&mut points, "{:.1},{:.1} ", x, y);
            }
            svg.push_str(&format!(
                "<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}'/>",
            ));

            let legend_y = legend_base_y + 22.0 * series_index as f64;
            svg.push_str(&format!(
                "<rect x='{:.1}' y='{:.1}' width='18' height='4' fill='{color}'/>",
                legend_x, legend_y - 10.0
            ));
            svg.push_str(&format!(
                "<text x='{:.1}' y='{:.1}' font-family='sans-serif' font-size='14' fill='#111'>{} - {}</text>",
                legend_x + 26.0,
                legend_y,
                escape_xml(&format!("K={}, S={}, p_min={:.2}", outcome.config.branching, outcome.config.depth, outcome.config.p_min)),
                series_label
            ));
        }
    }

    svg.push_str("</svg>");
    fs::write(path, svg)?;
    Ok(())
}

fn escape_xml(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn run_single_config(config: &RunConfig) -> std::io::Result<ExperimentOutcome> {
    let sample_times = linspace_rounds(config.rounds, config.sample_points);
    let leaf_probs = build_leaf_probabilities(config.branching, config.depth, config.p_min);
    let change_time = ((config.rounds as f64) * config.change_ratio).round() as usize;
    let env = LeafBanditEnv::new(LeafDistribution::linear(
        config.p_min,
        leaf_probs.len(),
        change_time,
    ));
    let tree_config = TreeConfig::new(config.branching, config.depth, config.shared_path.clone());

    let mut cost_totals = vec![0.0; sample_times.len()];
    let mut regret_totals = vec![0.0; sample_times.len()];
    let mut regret_sq_totals = vec![0.0; sample_times.len()];

    for run_index in 0..config.runs {
        let seed = config
            .seed
            .wrapping_add((run_index as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let mut policy = BarrierPartialSharePolicy::new(
            tree_config.clone(),
            config.epsilon,
            config.eta,
            seed,
        );
        let mut rng = crate::rng::SplitMix64::new(seed ^ 0xD1B54A32D192ED03);

        let mut cumulative_cost = 0.0;
        let mut sample_index = 0;

        for round in 1..=config.rounds {
            let trace = policy.sample_round();
            let cost = env.sample_cost(round, trace.leaf_position, &mut rng);
            cumulative_cost += cost;
            policy.apply_observation(&trace, cost);

            while sample_index < sample_times.len() && round == sample_times[sample_index] {
                let optimal = optimal_cost_at(round, &leaf_probs, change_time);
                let regret = (cumulative_cost - optimal) / round as f64;
                cost_totals[sample_index] += cumulative_cost;
                regret_totals[sample_index] += regret;
                regret_sq_totals[sample_index] += regret * regret;
                sample_index += 1;
            }
        }
    }

    let runs = config.runs.max(1) as f64;
    let mut mean_costs = Vec::with_capacity(sample_times.len());
    let mut mean_regrets = Vec::with_capacity(sample_times.len());
    let mut std_regrets = Vec::with_capacity(sample_times.len());

    for idx in 0..sample_times.len() {
        let mean_cost = cost_totals[idx] / runs;
        let mean_regret = regret_totals[idx] / runs;
        let variance = (regret_sq_totals[idx] / runs) - mean_regret * mean_regret;
        mean_costs.push(mean_cost);
        mean_regrets.push(mean_regret);
        std_regrets.push(variance.max(0.0).sqrt());
    }

    Ok(ExperimentOutcome {
        config: config.clone(),
        sample_times,
        mean_costs,
        mean_regrets,
        std_regrets,
    })
}

fn run_compare_single_config(config: &RunConfig) -> std::io::Result<CompareOutcome> {
    let sample_times = linspace_rounds(config.rounds, config.sample_points);
    let leaf_probs = build_leaf_probabilities(config.branching, config.depth, config.p_min);
    let change_time = ((config.rounds as f64) * config.change_ratio).round() as usize;
    let env = LeafBanditEnv::new(LeafDistribution::linear(
        config.p_min,
        leaf_probs.len(),
        change_time,
    ));
    let tree_config = TreeConfig::new(config.branching, config.depth, config.shared_path.clone());

    let mut barrier_regret_totals = vec![0.0; sample_times.len()];
    let mut epsilon_regret_totals = vec![0.0; sample_times.len()];

    for run_index in 0..config.runs {
        let base_seed = config
            .seed
            .wrapping_add((run_index as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let mut barrier = BarrierPartialSharePolicy::new(
            tree_config.clone(),
            config.epsilon,
            config.eta,
            base_seed,
        );
        let mut epsilon = EpsilonExp3Policy::new(
            tree_config.clone(),
            config.epsilon,
            config.eta,
            base_seed ^ 0xA5A5A5A5A5A5A5A5,
        );
        let mut barrier_rng = crate::rng::SplitMix64::new(base_seed ^ 0xD1B54A32D192ED03);
        let mut epsilon_rng = crate::rng::SplitMix64::new(base_seed ^ 0xC3A5C85C97CB3127);

        let mut barrier_cost = 0.0;
        let mut epsilon_cost = 0.0;
        let mut sample_index = 0;

        for round in 1..=config.rounds {
            let barrier_trace = barrier.sample_round();
            let barrier_round_cost = env.sample_cost(round, barrier_trace.leaf_position, &mut barrier_rng);
            barrier_cost += barrier_round_cost;
            barrier.apply_observation(&barrier_trace, barrier_round_cost);

            let epsilon_trace = epsilon.sample_round();
            let epsilon_round_cost = env.sample_cost(round, epsilon_trace.leaf_position, &mut epsilon_rng);
            epsilon_cost += epsilon_round_cost;
            epsilon.apply_observation(&epsilon_trace, epsilon_round_cost);

            while sample_index < sample_times.len() && round == sample_times[sample_index] {
                let optimal = optimal_cost_at(round, &leaf_probs, change_time);
                let barrier_regret = (barrier_cost - optimal) / round as f64;
                let epsilon_regret = (epsilon_cost - optimal) / round as f64;
                barrier_regret_totals[sample_index] += barrier_regret;
                epsilon_regret_totals[sample_index] += epsilon_regret;
                sample_index += 1;
            }
        }
    }

    let runs = config.runs.max(1) as f64;
    let barrier_mean_regrets = barrier_regret_totals.into_iter().map(|value| value / runs).collect();
    let epsilon_mean_regrets = epsilon_regret_totals.into_iter().map(|value| value / runs).collect();

    Ok(CompareOutcome {
        config: config.clone(),
        sample_times,
        barrier_mean_regrets,
        epsilon_mean_regrets,
    })
}

pub fn default_configs() -> Vec<RunConfig> {
    vec![
        RunConfig {
            branching: 2,
            depth: 2,
            p_min: 0.2,
            rounds: 200_000,
            runs: 3,
            change_ratio: 0.01,
            sample_points: 50,
            epsilon: 0.01,
            eta: 0.01,
            shared_path: vec![0],
            seed: 7,
        },
        RunConfig {
            branching: 4,
            depth: 3,
            p_min: 0.1,
            rounds: 200_000,
            runs: 3,
            change_ratio: 0.01,
            sample_points: 50,
            epsilon: 0.01,
            eta: 0.01,
            shared_path: vec![0],
            seed: 17,
        },
    ]
}

pub fn format_summary(outcomes: &[ExperimentOutcome]) -> String {
    let mut summary = String::new();
    for (idx, outcome) in outcomes.iter().enumerate() {
        let final_regret = outcome.mean_regrets.last().copied().unwrap_or(0.0);
        let final_cost = outcome.mean_costs.last().copied().unwrap_or(0.0);
        let _ = writeln!(
            &mut summary,
            "config {} -> final mean cost {:.4}, final mean regret {:.6}",
            idx,
            final_cost,
            final_regret
        );
    }
    summary
}

pub fn format_compare_summary(outcomes: &[CompareOutcome]) -> String {
    let mut summary = String::new();
    for (idx, outcome) in outcomes.iter().enumerate() {
        let barrier_final = outcome.barrier_mean_regrets.last().copied().unwrap_or(0.0);
        let epsilon_final = outcome.epsilon_mean_regrets.last().copied().unwrap_or(0.0);
        let _ = writeln!(
            &mut summary,
            "config {} -> barrier {:.6}, epsilon-exp3 {:.6}",
            idx,
            barrier_final,
            epsilon_final
        );
    }
    summary
}

pub fn output_dir_from(path: impl AsRef<Path>) -> PathBuf {
    path.as_ref().to_path_buf()
}
