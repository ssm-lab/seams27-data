"""
Plotting utilities for cumulative-reward training curves.

Produces per-start-stage individual PNGs, a combined 2×3 grid PNG, and a
standalone all-curricula summary PNG.  All figures share the same Y-axis
ceiling (Y_MAX) so panels are directly comparable.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from trainer import TrainResult

Y_MAX = 225_000  # shared y-axis ceiling across all plots

CURRICULUM_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
BASELINE_COLORS   = ["tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:blue"]

# CURRICULUM_COLORS = [
#     "tab:blue",      # 0
#     "tab:orange",    # 1
#     "tab:green",     # 2
#     "tab:red",       # 3
#     "tab:purple",    # 4
#     "tab:brown",     # 5
#     "tab:pink",      # 6
#     "tab:gray",      # 7
#     "tab:olive",     # 8
#     "tab:cyan"       # 9
# ]
#
# BASELINE_COLORS = [
#     "darkred",       # 0
#     "darkblue",      # 1
#     "darkgreen",     # 2
#     "gold",          # 3
#     "indigo",        # 4
#     "sienna",        # 5
#     "navy",          # 6
#     "khaki",         # 7
#     "salmon",        # 8
#     "teal"           # 9
# ]

def _curriculum_color(start_stage: int) -> str:
    """Return a consistent color for the curriculum line of a given start stage."""
    return CURRICULUM_COLORS[(start_stage - 1) % len(CURRICULUM_COLORS)]

def _baseline_color(label: str, all_labels: list[str]) -> str:
    """Return a consistent color for a baseline line identified by *label*."""
    idx = all_labels.index(label)
    return BASELINE_COLORS[idx % len(BASELINE_COLORS)]

def _plot_individual_panel(ax, start_stage, cur_results, baseline_results, baseline_specs_by_stage, all_baseline_labels):
    """Shared helper: draw one start-stage panel onto ax."""
    n_stages_total = start_stage + len(cur_results) - 1
    x_cur = np.array([ts for r in cur_results for ts in r.timesteps], dtype=float)
    y_cur = np.cumsum([rew for r in cur_results for rew in r.episode_rewards])
    ax.plot(x_cur, y_cur,
            label=f"Curriculum (E{start_stage}–E{n_stages_total})",
            color=_curriculum_color(start_stage),
            linewidth=1.5)

    x_max = x_cur[-1]
    for label in baseline_specs_by_stage[start_stage].keys():
        result = baseline_results[label]
        x_bas = np.array(result.timesteps, dtype=float)
        y_bas = np.cumsum(result.episode_rewards)
        mask = x_bas <= x_max
        ax.plot(x_bas[mask], y_bas[mask],
                label=label,
                color=_baseline_color(label, all_baseline_labels),
                linewidth=1.5, linestyle="dashed")

    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_summary_panel(ax, curriculum_results_by_stage, baseline_results, all_baseline_labels):
    """Shared helper: draw the all-curricula summary onto ax."""
    x_max = 0
    for start_stage, cur_results in curriculum_results_by_stage.items():
        n_stages_total = start_stage + len(cur_results) - 1
        x = np.array([ts for r in cur_results for ts in r.timesteps], dtype=float)
        y = np.cumsum([rew for r in cur_results for rew in r.episode_rewards])
        ax.plot(x, y,
                label=f"Curriculum (E{start_stage}–E{n_stages_total})",
                color=_curriculum_color(start_stage),
                linewidth=1.5)
        x_max = max(x_max, x[-1])

    for label, result in baseline_results.items():
        x = np.array(result.timesteps, dtype=float)
        y = np.cumsum(result.episode_rewards)
        mask = x <= x_max
        ax.plot(x[mask], y[mask],
                label=label,
                color=_baseline_color(label, all_baseline_labels),
                linewidth=1.5, linestyle="dashed")

    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def make_training_cumulative_reward_plot_each(
    curriculum_results_by_stage: dict[int, list],
    baseline_results: dict[str, TrainResult],
    baseline_specs_by_stage: dict[int, dict],
    output_dir: str,
):
    """Individual PNG per start-stage (same y-axis for all) + combined 2×3 grid
    where the last panel is the all-curricula summary."""
    all_baseline_labels = list(baseline_results.keys())
    start_stages = sorted(curriculum_results_by_stage.keys())

    # --- individual files ---
    for start_stage in start_stages:
        cur_results = curriculum_results_by_stage[start_stage]
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_individual_panel(ax, start_stage, cur_results, baseline_results,
                               baseline_specs_by_stage, all_baseline_labels)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cumulative_reward_start_{start_stage}.png"), dpi=300)
        plt.close(fig)

    # --- combined grid: N individual panels + 1 summary panel ---
    n_panels = len(start_stages) + 1  # last panel = summary
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axes_flat = axes.flatten()

    for idx, start_stage in enumerate(start_stages):
        _plot_individual_panel(axes_flat[idx], start_stage,
                               curriculum_results_by_stage[start_stage],
                               baseline_results, baseline_specs_by_stage, all_baseline_labels)

    _plot_summary_panel(axes_flat[len(start_stages)], curriculum_results_by_stage,
                        baseline_results, all_baseline_labels)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_reward_combined.png"), dpi=300)
    plt.close(fig)


def make_training_cumulative_reward_plot_all(
    curriculum_results_by_stage: dict[int, list],
    baseline_results: dict[str, TrainResult],
    output_dir: str,
):
    """Standalone all-curricula summary figure (same y-axis)."""
    all_baseline_labels = list(baseline_results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_summary_panel(ax, curriculum_results_by_stage, baseline_results, all_baseline_labels)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_reward_all.png"), dpi=300)
    plt.close(fig)




