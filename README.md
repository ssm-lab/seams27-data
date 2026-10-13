# Data Package
**For the article *Model-Driven Self-Adaptive Curricula for Efficient Reinforcement Learning.***

---
## About

Curriculum learning (CL) can improve reinforcement learning (RL) by exposing an agent to progressively challenging training environments. However, existing approaches often rely on predefined curricula or technique-specific adaptation mechanisms.
We propose a model-driven approach for self-adaptive CL.
Curricula and training environments are represented in a runtime model and evolved through model transformations in a MAPE-K loop.
Analysis is achieved through semantically lifted, highly contextual learning conditions, captured in model queries.
Our evaluation shows that the approach acquires target-task competence substantially faster than random and genetic-algorithm-based curriculum baselines, while matching or exceeding their final performance.

## Content description
- `01-configurations` - Contains the configurations for experiments.
- `02-train-data` - Contains trained agents' policies.
- `03-scripts` - Contains Python analysis scripts to obtain the results in the 04-results folder.
- `04-results`- Contains the data, plots and visualization that are used in the publication.

## Setup

Requires Python 3.10+. Download and extract the data package, then run from its root directory:

```bash
cd seams27-data
python3 -m venv .venv
.venv/bin/python -m pip install -r 03-scripts/requirements.txt
cd 03-scripts
```

Run the commands below from `03-scripts/`.
Only adaptive training requires the companion adaptation-engine implementation and its Eclipse dependencies; follow its setup guide before starting the server.

## 1. Configurations

- `01-configurations/environment-config.json`: local environment definitions; `target_environment_id` selects the common evaluation target (`env_6`). Keep this target consistent with the EMF server configuration for adaptive training.
- `01-configurations/ga/`: 10 genetic-algorithm-based static curricula.
- `01-configurations/random/experiment-10-seed-42/`: 10 generated random sets of environments. Each is used in sampled order and complexity order.
  - To regenerate the random sets:
    ```bash
    ../.venv/bin/python generate_random_config.py --all-seeds --n-seeds 10 --master-seed 42 \
    --output-dir ../01-configurations/random/experiment-10-seed-42
    ```
- `01-configurations/adaptive/experiment-10-seed-42/seeds.json`: 10 generated seeds for adaptive learning.
  - To regenerate adaptive seeds:
    ```bash
    ../.venv/bin/python generate_adaptive_seeds.py
    ```
- `01-configurations/learning-config.json`: training and evaluation settings.
  - Defaults: 300,000 training steps, 1,000-step evaluation interval, and 100 episodes per evaluation.

## 2.1. Training (static baselines)

```bash
../.venv/bin/python run_comparison.py train-static
```
Runs 10 GA, 10 random, 10 ordered random baselines, and one target-only (hardest) baseline.


## 2.2. Training (adaptive curricula)
For each run:
1. Stop the previous EMF server. In Eclipse, open **Run Configurations → AdaptationServer → Common → Output File** and select `<repo>/04-results/comparison/adaptive/run_N/adaptive-run.log`, using a zero-padded folder number (`run_01` through `run_10`). Create the folder if needed, leave **Append** unchecked, and click **Apply**.
2. Launch `AdaptationServer.launch` from the bridge plugin. Wait for `Bind successfully. Waiting for request...`.
3. In the terminal, replace `N` with each run number from 1 through 10:
```bash
../.venv/bin/python run_comparison.py train-adaptive --run N
```

## 3. Evaluation & Plotting
```bash
../.venv/bin/python run_comparison.py evaluate
../.venv/bin/python analyze_adaptation_logs.py
../.venv/bin/python run_comparison.py plot
```

## 4. Results

Training artifacts are saved under `02-train-data/comparison/`. Evaluation results, plots, and summary CSVs are saved under `04-results/comparison/`. Override these with `--output` (training data) and `--results` (results), using consistent paths across phases.
