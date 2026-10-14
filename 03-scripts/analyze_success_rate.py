"""Extract success rate from saved target-evaluation checkpoints.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "02-train-data/comparison"
RESULTS = ROOT / "04-results/comparison"
CONDITIONS = ("baseline", "random_ordered", "random_unordered", "curriculum", "adaptive")
THRESHOLD = 0.8


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_curve_milestones(condition, curves):
    """Find crossings of the across-run mean at shared recorded checkpoints."""
    steps = curves[0][0]
    for run_steps, _ in curves[1:]:
        steps = np.intersect1d(steps, run_steps)
    rates = np.stack([run_rates[np.searchsorted(run_steps, steps)]
                      for run_steps, run_rates in curves])
    mean_rates = rates.mean(axis=0)
    milestones = []
    for label, threshold in (("first_positive", 0), ("reaches_0.1", .1),
                             ("reaches_0.8", .8), ("reaches_0.9", .9)):
        reached = np.flatnonzero(mean_rates > 0 if threshold == 0
                                 else mean_rates >= threshold)
        index = int(reached[0]) if reached.size else None
        milestones.append(dict(
            condition=condition, n_runs=len(curves), milestone=label,
            steps=int(steps[index]) if index is not None else "",
            mean_success_rate=float(mean_rates[index]) if index is not None else "",
        ))
    return milestones


def main():
    runs = []
    milestones = []
    for condition in CONDITIONS:
        curves = []
        for path in sorted((TRAINING / condition).glob("run_*/evaluation_curve.npz")):
            with np.load(path) as curve:
                steps = curve["timesteps"]
                rates = curve["success_rates"]
                curves.append((steps, rates))
                first_success = steps[rates > 0]
                threshold_steps = steps[rates >= THRESHOLD]
                runs.append(dict(
                    condition=condition,
                    run=path.parent.name,
                    final_success_rate=float(rates[-1]),
                    first_success_steps=int(first_success[0]) if first_success.size else "",
                    threshold_steps=int(threshold_steps[0]) if threshold_steps.size else "",
                ))
        milestones.extend(mean_curve_milestones(condition, curves))

    summaries = []
    for condition in CONDITIONS:
        group = [row for row in runs if row["condition"] == condition]
        summary = dict(
            condition=condition,
            n_runs=len(group),
            success_threshold=THRESHOLD,
            runs_with_success=sum(row["first_success_steps"] != "" for row in group),
            runs_reaching_threshold=sum(row["threshold_steps"] != "" for row in group),
        )
        for metric in ("final_success_rate", "first_success_steps", "threshold_steps"):
            values = [row[metric] for row in group if row[metric] != ""]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            summary[f"{metric}_population_sd"] = (
                float(np.std(values)) if len(values) > 1 else ""
            )
        summaries.append(summary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "success_rate_per_run.csv", runs)
    write_csv(RESULTS / "success_rate_summary.csv", summaries)
    write_csv(RESULTS / "success_rate_curve_milestones.csv", milestones)
    print(f"Extracted {len(runs)} runs. Times are in training steps; blank means unavailable.")
    print(RESULTS / "success_rate_per_run.csv")
    print(RESULTS / "success_rate_summary.csv")
    print(RESULTS / "success_rate_curve_milestones.csv")


if __name__ == "__main__":
    main()
