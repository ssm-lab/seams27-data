"""Extract logged actions and summarize counts and stage durations across runs.
"""
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev, variance

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04-results/comparison"
TRAINING = ROOT / "02-train-data/comparison/adaptive"
RULES = {
    "Mastery": ("Mastery", "Advance"),
    "Excessive challenge": ("ExcessiveChallenge", "Simplify"),
    "Stagnation not on FinalStage": ("StagnationNonFinal", "Diversify"),
    "Stagnation on FinalStage": ("StagnationFinal", "Simplify"),
}
DETECTION = re.compile(r" - (.+) detected on stage .*; attempting")
NO_CHANGE = {
    "No next stage exists": "no_next_stage",
    "Stage limit reached": "stage_limit_reached",
    "Generate the identical environment": "identical_environment",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_run(log):
    run = log.parent.name
    episodes = read_csv(TRAINING / run / "episodes.csv")
    steps = [0] + [int(row["steps"]) for row in episodes if row["completed"] == "True"]
    events = []
    event = None
    episode = 0
    activation_episode = 0
    activation_step = 0

    with log.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if "Received request:" in line:
                request = json.loads(line.split("Received request: ", 1)[1])
                if request["type"] == "episodeOutcome":
                    episode = request["episode"]
                event = None

            detection = DETECTION.search(line)
            if detection:
                condition, action = RULES[detection[1]]
                event = dict(
                    run=run, episode=episode, global_step=steps[episode],
                    condition=condition, action=action,
                    episodes_in_stage=episode - activation_episode,
                    steps_in_stage=steps[episode] - activation_step,
                    outcome="", reason="", log_line=line_number,
                )
                events.append(event)

            if event is None:
                continue
            if ("Current stage is now set to:" in line or
                    "After activation, Current stage is set to:" in line):
                event["outcome"] = "successful"
                event["reason"] = "stage_activated"
                activation_episode = episode
                activation_step = steps[episode]
            for message, reason in NO_CHANGE.items():
                if message in line:
                    event["outcome"] = "no_change"
                    event["reason"] = reason
            if "Sent response:" in line:
                event = None
    return events


def summarize_run(run, events):
    successful = [event for event in events if event["outcome"] == "successful"]
    summary = {"run": run, "total_successful_adaptations": len(successful)}
    for condition, action in RULES.values():
        if condition == "StagnationFinal":
            continue
        selected = [event for event in events if event["condition"] == condition]
        prefix = f"{condition}_{action}".lower()
        summary[f"{prefix}_attempts"] = len(selected)
        for outcome in ("successful", "no_change"):
            summary[f"{prefix}_{outcome}"] = sum(e["outcome"] == outcome for e in selected)
    for action in ("Advance", "Simplify", "Diversify"):
        summary[f"{action.lower()}_count"] = sum(e["action"] == action for e in successful)
    durations = [int(e["steps_in_stage"]) for e in successful
                 if e["condition"] == "Mastery" and e["action"] == "Advance"]
    summary["mean_steps_before_mastery_advance"] = mean(durations) if durations else ""
    return summary


def main():
    logs = sorted((RESULTS / "adaptive").glob("run_*/adaptive-run.log"))
    events = [event for log in logs for event in extract_run(log)]
    events_path = RESULTS / "adaptation_events.csv"
    write_csv(events_path, events)

    # Summaries read the extracted CSV so each result is traceable to its rows.
    events = read_csv(events_path)
    summaries = [summarize_run(log.parent.name,
                              [e for e in events if e["run"] == log.parent.name])
                 for log in logs]
    overall = []
    for label, calculation in (("mean", mean), ("sample_sd", stdev),
                               ("sample_variance", variance)):
        row = {"run": label}
        for metric in list(summaries[0])[1:]:
            values = [s[metric] for s in summaries if s[metric] != ""]
            minimum = 1 if label == "mean" else 2
            row[metric] = calculation(values) if len(values) >= minimum else ""
        overall.append(row)
    write_csv(RESULTS / "adaptation_summary.csv", summaries + overall)
    print(f"Extracted {len(events)} action attempts from {len(logs)} runs.")
    print(events_path)
    print(RESULTS / "adaptation_summary.csv")


if __name__ == "__main__":
    main()
