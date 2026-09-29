"""
Orchestrates curriculum and baseline training experiments.

Runs all partial curricula (E1→E6, E2→E6, …, E5→E6) and the full-curriculum
baseline, saves agent checkpoints to 02-train-data/, and saves cumulative-reward
plots to 04-results/plots/.

Usage (from 03-scripts/ with the virtual environment activated):
    python runner.py --all-stages --baseline all
"""

import argparse
import os

import random
import numpy as np

from config_loader import load_configs, DEFAULT_ENVIRONMENT_CONFIG
from curriculum_builder import build_curriculum, get_target_spec,resolve_baseline_specs
from trainer import run_adaptive_curriculum, run_curriculum, run_baseline, TrainResult, make_env, evaluate_agent
from plotter import make_training_cumulative_reward_plot_each, make_training_cumulative_reward_plot_all
from pathlib import Path

SEED_NUM = 1

def make_experiment_name(env_config_path: str, learn_config_path: str) -> str:
    """Return a deterministic experiment directory name derived from config file stems."""
    env_name = Path(env_config_path).stem
    learn_name = Path(learn_config_path).stem
    return f"{env_name}__{learn_name}"

def parse_baseline(value: str | None) -> list[int] | str | None:
    """Parse --baseline argument: None, 'all', or comma-separated int indices."""
    if value is None:
        return None
    if value == "all":
        return "all"
    return [int(i) for i in value.split(",")]

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run curriculum learning and baseline experiments across multiple seeds."
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENVIRONMENT_CONFIG),
        help="Path to environment configuration file.",
    )
    parser.add_argument(
        "--learn-config",
        default="../01-configurations/learning-config.json",
        help="Path to learning configuration file.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Which environments to run as baselines. Options: 'all', or indices like '0,2,4'"
    )
    parser.add_argument(
        "--train-data",
        default="../02-train-data",
        help="Directory where agent checkpoints and training data will be saved.",
    )
    parser.add_argument(
        "--output",
        default="../04-results/plots",
        help="Directory where results and plots will be saved.",
    )
    # parser.add_argument(
    #     "--seeds",
    #     nargs="+",
    #     type=int,
    #     default=list(range(SEED_NUM)),
    #     help="List of random seeds to run.",
    # )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="Which curriculum stage to start from (1-indexed).",
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Run all partial curricula from full down to single stage",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Run one EMF-controlled adaptive curriculum experiment.",
    )
    parser.add_argument(
        "--zmq-endpoint",
        default="tcp://127.0.0.1:5558",
        help="ZeroMQ REQ/REP endpoint for initialization and adaptation results.",
    )
    return parser.parse_args()


def setup_experiment(args, env_cfg, learn_cfg):
    """Returns full curriculum, target spec, and per-experiment dirs for train data and plots."""
    full_curriculum = build_curriculum(env_cfg, learn_cfg)
    target_spec = get_target_spec(full_curriculum)
    experiment_name = make_experiment_name(args.env_config, args.learn_config)
    experiment_train_dir = os.path.join(args.train_data, experiment_name)
    experiment_output_dir = os.path.join(args.output, experiment_name)
    os.makedirs(experiment_train_dir, exist_ok=True)
    os.makedirs(experiment_output_dir, exist_ok=True)
    return full_curriculum, target_spec, experiment_train_dir, experiment_output_dir

def build_baseline_specs_by_stage(full_curriculum, args, n_stages):
    """Build a {start_stage: {label: EnvSpec}} mapping for all requested start stages."""
    start_stages = list(range(1, n_stages)) if args.all_stages else [args.start_stage]
    baseline_specs_by_stage = {}

    for start_stage in start_stages:
        curriculum = full_curriculum[start_stage - 1:]
        raw_specs = resolve_baseline_specs(curriculum, env_indices=args.baseline_indices)
        baseline_specs_by_stage[start_stage] = {
            f"Baseline (E{spec.env_id.split('_')[1]})": spec
            for i, (_, spec) in enumerate(raw_specs.items(), start=start_stage)
        }

    return baseline_specs_by_stage


def run_all_baselines(full_curriculum, target_spec, learn_cfg, args, experiment_train_dir):
    """Run baseline training once for all. Checkpoints saved to 02-train-data."""
    evaluate_baseline = learn_cfg.get("evaluation", {}).get("evaluate_baseline", True)
    baseline_specs = resolve_baseline_specs(full_curriculum, env_indices=args.baseline_indices)
    baseline_results = {}

    if not evaluate_baseline:
        return baseline_results

    print("=" * 70)
    print("Running Baselines")
    print("=" * 70)

    for i, (_, spec) in enumerate(baseline_specs.items(), start=1):
        env_num = spec.env_id.split("_")[1]
        label = f"Baseline (E{env_num})"
        _, result = run_baseline(
            target_spec=spec,
            total_steps=sum(item["steps"] for item in full_curriculum),
            learn_cfg=learn_cfg,
            output_dir=os.path.join(experiment_train_dir, "baseline", label),
        )
        baseline_results[label] = result

    return baseline_results


def run_all_curricula(full_curriculum, target_spec, env_cfg, learn_cfg, args, experiment_train_dir, seed):
    """Run each partial curriculum. Checkpoints saved to 02-train-data."""
    n_stages = len(full_curriculum)
    start_stages = list(range(1, n_stages)) if args.all_stages else [args.start_stage]
    curriculum_results_by_stage = {}

    for start_stage in start_stages:
        print("=" * 70)
        print(f"Curriculum start_stage={start_stage}")
        print("=" * 70)

        curriculum = build_curriculum(env_cfg, learn_cfg)
        curriculum = curriculum[start_stage - 1:]
        total_steps = sum(item["steps"] for item in curriculum)

        print(f"Environments: {len(curriculum)}, Steps: {total_steps}")
        print(f"Target: {target_spec.env_id}")
        print("=" * 70)

        stage_train_dir = os.path.join(
            experiment_train_dir, f"start_stage_{start_stage}"
        )
        os.makedirs(stage_train_dir, exist_ok=True)

        _, curriculum_results = run_curriculum(
            curriculum=curriculum,
            eval_spec=target_spec,
            learn_cfg=learn_cfg,
            output_dir=stage_train_dir,
        )
        curriculum_results_by_stage[start_stage] = curriculum_results

    return curriculum_results_by_stage


def run_adaptive_experiment(target_spec, learn_cfg, args, experiment_train_dir, total_steps):
    """Run one curriculum whose active environment is selected by EMF."""
    # Imported here so existing static/random runs still work before pyzmq is
    # installed or the EMF runtime server is started.
    from emf_bridge import EmfBridge

    adaptive_dir = os.path.join(experiment_train_dir, "adaptive")
    with EmfBridge(args.zmq_endpoint) as bridge:
        return run_adaptive_curriculum(
            bridge=bridge,
            eval_spec=target_spec,
            learn_cfg=learn_cfg,
            output_dir=adaptive_dir,
            max_interaction_steps=total_steps,
        )


def main():
    """Entry point: run all curricula and baselines, then save plots."""
    args = parse_args()
    args.baseline_indices = parse_baseline(args.baseline)
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    env_cfg, learn_cfg = load_configs(args.env_config, args.learn_config)

    seed = 42
    set_global_seed(seed)
    learn_cfg["curriculum"]["seed"] = seed

    full_curriculum, target_spec, experiment_train_dir, experiment_output_dir = setup_experiment(args, env_cfg, learn_cfg)

    if args.adaptive:
        adaptive_agent, adaptive_results = run_adaptive_experiment(target_spec, learn_cfg, args, experiment_train_dir, sum(stage["steps"] for stage in full_curriculum))

        eval_env = make_env(target_spec, seed=learn_cfg["curriculum"].get("seed", 42) + 1000)
        n_eval_episodes = learn_cfg.get("evaluation", {}).get("eval_episodes",100)

        mean_reward, success_rate = evaluate_agent(adaptive_agent, eval_env, n_eval_episodes)
        eval_env.close()

        print("\n=== Adaptive Final Evaluation ===")
        print(f"Target environment: {target_spec.env_id}")
        print(f"Episodes: {n_eval_episodes}")
        print(f"Mean reward: {mean_reward:.3f}")
        print(f"Success rate: {success_rate:.3f}")

        print("\nAdaptive experiment finished.")
        print(f"Train data saved to: {experiment_train_dir}")
        return

    baseline_results = run_all_baselines(
        full_curriculum, target_spec, learn_cfg, args, experiment_train_dir
    )
    curriculum_results_by_stage = run_all_curricula(
        full_curriculum, target_spec, env_cfg, learn_cfg, args, experiment_train_dir, seed
    )
    baseline_specs_by_stage = build_baseline_specs_by_stage(
        full_curriculum, args, n_stages = len(full_curriculum)
    )
    make_training_cumulative_reward_plot_each(
        curriculum_results_by_stage, baseline_results, baseline_specs_by_stage, experiment_output_dir
    )
    make_training_cumulative_reward_plot_all(
        curriculum_results_by_stage, baseline_results, experiment_output_dir
    )

    print("\nExperiment finished.")
    print(f"Train data saved to: {experiment_train_dir}")
    print(f"Plots saved to: {experiment_output_dir}")


def set_global_seed(seed: int):
    """Set Python and NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()
