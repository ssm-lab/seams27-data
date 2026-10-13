"""Reproducible comparison: train-static/train-adaptive -> evaluate -> plot.

N GA curricula, N paired random sets, N adaptive runs, and one target-only run.
Training and final evaluation resume independently. See ../README.md.
"""
import argparse
import csv
import json
import random
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt

from curriculum_builder import build_curriculum
from config_loader import DEFAULT_ENVIRONMENT_CONFIG, load_target_spec
from evaluator import evaluate_saved_policy
from trainer import run_curriculum, run_baseline, run_adaptive_curriculum

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"adaptive": "tab:red", "curriculum": "tab:blue",
          "random_unordered": "tab:orange", "random_ordered": "tab:green", "baseline": "black"}
LABELS = {"adaptive": "Adaptive", "curriculum": "GA", "random_ordered": "Random-ordered",
          "random_unordered": "Random", "baseline": "Baseline (hardest only)"}
N_GRID = 500


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def training_done(directory):
    """The final policy is written after the training data; evaluation is separate."""
    return all((Path(directory) / name).is_file() for name in
               ("final.npy", "training_curve.npz", "evaluation_curve.npz", "episodes.csv"))


def static_runs(args):
    """Use the first N files in filename order; reuse random files for both orders."""
    for condition, folder in (("curriculum", args.ga_dir),
                              ("random_unordered", args.random_dir),
                              ("random_ordered", args.random_dir)):
        files = sorted(folder.glob("*.json"))[:args.runs]
        if len(files) < args.runs:
            raise ValueError(f"Need {args.runs} curriculum files in {folder}; found {len(files)}")
        for index, path in enumerate(files, 1):
            if args.run is None or args.run == index:
                yield condition, index, path
    if args.run is None or args.run == 1:
        yield "baseline", 1, None


def save_training(directory, results, agent, total_steps):
    """Save every reward segment, marking budget-truncated episodes separately."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    cumulative = 0.0
    for activation, result in enumerate(results):
        completed = result.episode_completed or [True] * len(result.timesteps)
        for step, reward, length, complete in zip(result.timesteps, result.episode_rewards,
                                                   result.episode_lengths, completed):
            cumulative += reward
            rows.append(dict(activation=activation, env_id=result.env_id, steps=step, episodic_return=reward,
                             episode_length=length, completed=complete, cumulative_reward=cumulative))
    if not rows or rows[-1]["steps"] != total_steps:
        raise ValueError("Training ended before the requested budget; not marking the run complete")
    with (directory / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "training_curve.npz", timesteps=[r["steps"] for r in rows],
                        cumulative_rewards=[r["cumulative_reward"] for r in rows])
    np.savez_compressed(directory / "evaluation_curve.npz",
                        timesteps=[v for r in results for v in r.eval_timesteps],
                        mean_rewards=[v for r in results for v in r.eval_mean_rewards],
                        success_rates=[v for r in results for v in r.eval_success_rates])
    agent.save(str(directory / "final.npy"))


def train_run(args, learning, target, condition, index, config_path=None):
    directory = args.output / condition / f"run_{index:02d}"
    if training_done(directory):
        print(f"[skip training] {directory}")
        return
    total_steps = learning["curriculum"]["size"] * learning["curriculum"]["steps_per_environment"]
    seed = learning["curriculum"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    print(f"Training {condition}/run_{index:02d}")
    if condition == "adaptive":
        from emf_bridge import EmfBridge
        generation_seed = read_json(ROOT / "01-configurations/adaptive/experiment-10-seed-42/seeds.json")["seeds"][index - 1]
        print(f"Adaptive run {index}: generation seed={generation_seed}")
        with EmfBridge(args.zmq_endpoint, generation_seed=generation_seed,
                       trace_path=directory / "session.jsonl") as bridge:
            agent, results = run_adaptive_curriculum(bridge, target, learning, str(directory),
                                                    max_interaction_steps=total_steps)
    elif condition == "baseline":
        agent, result = run_baseline(target, total_steps, learning, str(directory))
        results = [result]
    else:
        curriculum = build_curriculum(read_json(config_path), learning, ordered=condition != "random_unordered")
        agent, results = run_curriculum(curriculum, target, learning, str(directory))
    save_training(directory, results, agent, total_steps)


def evaluate_runs(args, learning, target):
    settings = learning["evaluation"]
    for condition, directories in evaluation_run_directories(args.output).items():
        for name in directories:
            directory = Path(name)
            if args.run and directory.name != f"run_{args.run:02d}":
                continue
            path = args.results / condition / directory.name / "eval_results.json"
            if path.exists():
                print(f"[skip evaluation] {directory}")
                continue
            result = evaluate_saved_policy(directory / "final.npy", target, settings["eval_episodes"],
                                           settings.get("eval_seed", 123))
            write_json(path, result)
            print(f"Evaluated {directory}: {result}")


def load_curve(run_dir: str, grid: np.ndarray) -> np.ndarray:
    """Load a saved training curve and interpolate onto *grid*."""
    data = np.load(os.path.join(run_dir, "training_curve.npz"))
    ts   = np.concatenate([[0], data["timesteps"]])
    cr   = np.concatenate([[0], data["cumulative_rewards"]])
    return np.interp(grid, ts, cr)


def load_evaluation_curve(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load periodic greedy-policy evaluations on the common E6 target."""
    data = np.load(os.path.join(run_dir, "evaluation_curve.npz"))
    return data["timesteps"], data["mean_rewards"], data["success_rates"]


def interpolate_evaluation_curve(
    timesteps: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Interpolate only inside a run's observed training interval."""
    return np.interp(grid, timesteps, values, left=np.nan, right=np.nan)


def aggregate_evaluation_curves(
    curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, population standard deviation, and sample count per grid point."""
    samples = np.stack([
        interpolate_evaluation_curve(timesteps, values, grid)
        for timesteps, values in curves
    ])
    present = np.isfinite(samples)
    counts = present.sum(axis=0)
    totals = np.nansum(samples, axis=0)
    means = np.divide(
        totals, counts, out=np.full(grid.shape, np.nan, dtype=float), where=counts > 0
    )

    deviations = np.where(present, samples - means, 0.0)
    variance = np.divide(
        np.square(deviations).sum(axis=0),
        counts,
        out=np.zeros(grid.shape, dtype=float),
        where=counts > 0,
    )
    std = np.sqrt(variance)
    return means, std, counts


def evaluation_run_directories(output_dir):
    """Discover saved runs directly from condition folders."""
    grouped = {}
    for condition in LABELS:
        directories = [str(p) for p in sorted((Path(output_dir) / condition).glob("run_*"))
                       if all((p / name).is_file() for name in
                              ("final.npy", "training_curve.npz", "evaluation_curve.npz"))]
        if directories:
            grouped[condition] = directories
    return grouped


def save_comparison_figure(fig, axis, output_dir, name):
    """Save an untitled figure with its legend below, in PNG and vector PDF."""
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=5, fontsize=9, columnspacing=1.2, handlelength=2, frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path(output_dir) / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_target_evaluation_plot(train_dir, output_dir, total_steps: int) -> None:
    """Save target success rate and mean return as separate figures."""
    directories = evaluation_run_directories(train_dir)
    if not directories:
        print("No periodic target evaluation curves found.")
        return
    grid = np.linspace(0, total_steps, N_GRID)
    loaded = {condition: [load_evaluation_curve(d) for d in runs]
              for condition, runs in directories.items()}
    for metric, ylabel, filename in (
        (2, "Success rate", "target_success_rate"),
        (1, "Mean episodic return", "target_mean_return"),
    ):
        fig, axis = plt.subplots(figsize=(9, 5))
        for condition, runs in loaded.items():
            curves = [(run[0], run[metric]) for run in runs]
            mean, std, counts = aggregate_evaluation_curves(curves, grid)
            valid = counts > 0
            axis.plot(grid[valid], mean[valid], color=COLORS[condition], linewidth=2,
                      label=LABELS[condition], linestyle="--" if condition == "baseline" else "-")
            if len(runs) > 1:
                axis.fill_between(grid[valid], (mean - std)[valid], (mean + std)[valid],
                                  color=COLORS[condition], alpha=.18)
        axis.set(xlabel="Timesteps", ylabel=ylabel)
        if metric == 2:
            axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=.3)
        save_comparison_figure(fig, axis, output_dir, filename)


def make_summary_plot(train_dir, output_dir, total_steps):
    grid = np.linspace(0, total_steps, N_GRID)
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition, directories in evaluation_run_directories(train_dir).items():
        curves = np.stack([load_curve(d, grid) for d in directories])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        ax.plot(grid, mean, color=COLORS[condition], label=LABELS[condition],
                linestyle="--" if condition == "baseline" else "-")
        if len(curves) > 1:
            ax.fill_between(grid, mean - std, mean + std, color=COLORS[condition], alpha=.18)
        for directory in directories:
            data = np.load(Path(directory) / "training_curve.npz")
            run_fig, run_ax = plt.subplots()
            run_ax.plot(data["timesteps"], data["cumulative_rewards"])
            run_ax.set(xlabel="Timesteps", ylabel="Cumulative training reward")
            run_fig.tight_layout()
            result_dir = Path(output_dir) / condition / Path(directory).name
            result_dir.mkdir(parents=True, exist_ok=True)
            run_fig.savefig(result_dir / "cumulative_reward.png")
            plt.close(run_fig)
    ax.set(xlabel="Timesteps", ylabel="Cumulative training reward")
    ax.grid(True, alpha=.3)
    save_comparison_figure(fig, ax, output_dir, "summary_cumulative_reward")


def make_adaptation_plot(output_dir, total_steps):
    """Plot successful actions on combined and per-run condition/action timelines."""
    events_path = Path(output_dir) / "adaptation_events.csv"
    if not events_path.exists():
        print("Run analyze_adaptation_logs.py to generate the adaptation timeline data.")
        return
    with events_path.open(newline="") as stream:
        events = list(csv.DictReader(stream))

    patterns = [
        ("Mastery", "Mastery\n→ Advance", "tab:blue"),
        ("ExcessiveChallenge", "Excessive challenge\n→ Simplify", "tab:orange"),
        ("StagnationNonFinal", "Stagnation\n→ Diversify", "teal"),
    ]
    runs = sorted({event["run"] for event in events})
    events = [event for event in events if event["outcome"] == "successful"]
    fig, axis = plt.subplots(figsize=(11, 2.4))
    for lane, (condition, label, color) in enumerate(patterns):
        steps = [int(e["global_step"]) for e in events if e["condition"] == condition]
        axis.scatter(steps, [lane] * len(steps), s=22,
                     color=color, alpha=.8, zorder=3)
    axis.set_yticks(range(len(patterns)), [p[1] for p in patterns], fontsize=10)
    axis.set_ylim(len(patterns) - .5, -.5)
    axis.set_xlim(0, total_steps)
    ticks = np.linspace(0, total_steps, 7)
    axis.set_xticks(ticks, [f"{int(t / 1000)}k" if t else "0" for t in ticks])
    axis.set_xlabel("Timestep")
    axis.set_axisbelow(True)
    axis.grid(True, alpha=.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path(output_dir) / f"adaptation_timeline.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    nrows = (len(runs) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 1.5 * nrows + .6),
                             sharex=True, sharey=True, squeeze=False)
    for axis, run in zip(axes.flat, runs):
        for lane, (condition, label, color) in enumerate(patterns):
            steps = [int(e["global_step"]) for e in events
                     if e["run"] == run and e["condition"] == condition]
            axis.scatter(steps, [lane] * len(steps), s=24, color=color, zorder=3)
        axis.set_title(run.replace("run_", "Run "), loc="left", fontsize=11)
        axis.set_yticks(range(len(patterns)), [p[1] for p in patterns], fontsize=9)
        axis.set_ylim(len(patterns) - .5, -.5)
        axis.set_xlim(0, total_steps)
        axis.set_xticks(ticks, [f"{int(t / 1000)}k" if t else "0" for t in ticks])
        axis.tick_params(axis="x", labelbottom=True, labelsize=9)
        axis.set_axisbelow(True)
        axis.grid(True, alpha=.25)
    for axis in list(axes.flat)[len(runs):]:
        axis.set_visible(False)
    fig.supxlabel("Timesteps", y=.015)
    fig.tight_layout(rect=(0, .045, 1, 1), h_pad=.8, w_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(Path(output_dir) / f"adaptation_timeline_per_run.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_eval_csvs(train_dir, output_dir):
    rows = []
    for condition, directories in evaluation_run_directories(train_dir).items():
        for directory in directories:
            path = Path(output_dir) / condition / Path(directory).name / "eval_results.json"
            if path.exists():
                result = read_json(path)
                rows.append(dict(condition=condition, run_id=Path(directory).name,
                                 mean_reward=result["mean_reward"], success_rate=result["success_rate"]))
    if not rows:
        return
    with (Path(output_dir) / "evaluation_per_run.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for condition in LABELS:
        group = [r for r in rows if r["condition"] == condition]
        if not group:
            continue
        summary = dict(condition=condition, n_runs=len(group))
        for metric in ("mean_reward", "success_rate"):
            values = [r[metric] for r in group]
            summary[metric + "_mean"] = np.mean(values)
            summary[metric + "_std"] = np.std(values) if len(group) > 1 else "N/A"
        summaries.append(summary)
    with (Path(output_dir) / "evaluation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train-static", "train-adaptive", "evaluate", "plot"])
    parser.add_argument("--output", type=Path, default=ROOT / "02-train-data/comparison",
                        help="Training artifact directory (also read by evaluate and plot)")
    parser.add_argument("--results", type=Path, default=ROOT / "04-results/comparison",
                        help="Directory for evaluation results, plots, and summary CSVs")
    parser.add_argument("--ga-dir", type=Path, default=ROOT / "01-configurations/ga")
    parser.add_argument("--random-dir", type=Path,
                        default=ROOT / "01-configurations/random/experiment-10-seed-42")
    parser.add_argument("--target-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--learn-config", type=Path, default=ROOT / "01-configurations/learning-config.json")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--run", type=int, help="Run number; required for adaptive training")
    parser.add_argument("--zmq-endpoint", default="tcp://127.0.0.1:5558")
    args = parser.parse_args()
    if args.runs < 1 or (args.run is not None and not 1 <= args.run <= args.runs):
        parser.error("Use a run number between 1 and --runs")
    if args.phase == "train-adaptive" and args.run is None:
        parser.error("Use --run N after starting a fresh Eclipse server")
    learning = read_json(args.learn_config)
    if args.phase != "plot":
        target = load_target_spec(args.target_config)
    total_steps = learning["curriculum"]["size"] * learning["curriculum"]["steps_per_environment"]
    if args.phase in ("train-static", "train-adaptive"):
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.results.mkdir(parents=True, exist_ok=True)
    if args.phase == "train-static":
        for condition, index, path in static_runs(args):
            train_run(args, learning, target, condition, index, path)
    elif args.phase == "train-adaptive":
        train_run(args, learning, target, "adaptive", args.run)
    elif args.phase == "evaluate":
        evaluate_runs(args, learning, target)
    else:
        make_summary_plot(args.output, args.results, total_steps)
        make_target_evaluation_plot(args.output, args.results, total_steps)
        save_eval_csvs(args.output, args.results)
        make_adaptation_plot(args.results, total_steps)


if __name__ == "__main__":
    main()
