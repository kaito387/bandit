use std::env;
use std::path::PathBuf;

use bandit::{default_configs, format_compare_summary, output_dir_from, run_comparison, ExperimentConfig};

fn parse_args() -> (PathBuf, Vec<bandit::RunConfig>) {
    let mut args = env::args().skip(1);
    let mut output_dir = PathBuf::from("results");
    let mut configs = default_configs();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => {
                if let Some(value) = args.next() {
                    output_dir = PathBuf::from(value);
                }
            }
            "--quick" => {
                configs = vec![bandit::RunConfig {
                    branching: 2,
                    depth: 2,
                    p_min: 0.2,
                    rounds: 1_000,
                    runs: 2,
                    change_ratio: 0.01,
                    sample_points: 20,
                    epsilon: 0.01,
                    eta: 0.01,
                    shared_path: vec![0],
                    seed: 7,
                }];
            }
            "--config" => {
                if let Some(value) = args.next() {
                    if let Some(config) = parse_config(&value) {
                        configs.push(config);
                    }
                }
            }
            _ => {}
        }
    }

    (output_dir, configs)
}

fn parse_config(value: &str) -> Option<bandit::RunConfig> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() < 3 {
        return None;
    }

    let branching = parts[0].parse().ok()?;
    let depth = parts[1].parse().ok()?;
    let p_min = parts[2].parse().ok()?;

    Some(bandit::RunConfig {
        branching,
        depth,
        p_min,
        rounds: 10_000_000,
        runs: 3,
        change_ratio: 0.01,
        sample_points: 50,
        epsilon: 0.01,
        eta: 0.01,
        shared_path: vec![0],
        seed: 7,
    })
}

fn main() -> std::io::Result<()> {
    let (output_dir, configs) = parse_args();
    let experiment = ExperimentConfig {
        output_dir: output_dir_from(output_dir),
        configs,
    };

    println!("config: {:#?}", experiment);

    let comparisons = run_comparison(&experiment)?;
    print!("{}", format_compare_summary(&comparisons));
    println!("wrote {}", experiment.output_dir.join("epsilon_exp3_compare.csv").display());
    println!("wrote {}", experiment.output_dir.join("epsilon_exp3_compare.svg").display());
    Ok(())
}
