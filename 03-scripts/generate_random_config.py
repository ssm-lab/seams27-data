"""
Generate randomised environment configurations for multi-seed baseline comparison.

Ordering (random vs sorted-by-complexity) is a runtime result made in
run_comparison.py — it is NOT baked into the config files.  Every generated
file stores environments in their raw generation order; run_comparison.py
decides whether to sort them (ordered mode) or keep that order (unordered mode).

30 fixed seeds are defined in SEEDS (reproducible, generated from Random(42)).

Usage (from python/ directory):

  # Generate all 30 configs at once:
  python generate_random_config.py --all-seeds \\
      --target-config configs/8x8/environments-8x8-6.json \\
      --output-dir   configs/8x8

  # Generate a single config by 1-indexed position in SEEDS:
  python generate_random_config.py --seed-index 5 \\
      --target-config configs/8x8/environments-8x8-6.json \\
      --output-dir   configs/8x8

  # Generate one config with an explicit seed value:
  python generate_random_config.py --seed 42 \\
      --target-config configs/8x8/environments-8x8-6.json \\
      --output        configs/8x8/environments-8x8-random-custom.json
"""

import argparse
import json
import os
import random
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fire import generate_random_map
from config_loader import DEFAULT_ENVIRONMENT_CONFIG, target_environment

def make_seeds(master_seed: int = 42, n: int = 30) -> list[int]:
    """Derive n reproducible env-generation seeds from a single master seed."""
    return sorted(random.Random(master_seed).sample(range(10_000), n))


# ---------------------------------------------------------------------------
# Complexity & density helpers
# ---------------------------------------------------------------------------

def compute_complexity(desc: list[str]) -> float:
    """
    c(E) = 1 - sqrt(P_E / P)

    P   = C(2n-2, n-1)  — total down-right paths in an n×n grid
    P_E = number of down-right paths from S(0,0) to G(n-1,n-1) avoiding B tiles
          (computed via DP).
    """
    n = len(desc)
    P = comb(2 * n - 2, n - 1)
    dp = [[0] * n for _ in range(n)]
    dp[0][0] = 1
    for r in range(n):
        for c in range(n):
            if r == 0 and c == 0:
                continue
            if desc[r][c] == "B":
                continue
            dp[r][c] = (dp[r - 1][c] if r > 0 else 0) + (dp[r][c - 1] if c > 0 else 0)
    P_E = dp[n - 1][n - 1]
    return 1.0 - (P_E / P) ** 0.5


def compute_fire_density(desc: list[str]) -> float:
    """Fraction of all cells that are on fire (B). Used for display only."""
    total = sum(len(row) for row in desc)
    return sum(row.count("B") for row in desc) / total


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def generate_random_envs(
    n: int,
    size: int = 8,
    max_fire_density: float = 0.30,
    max_complexity: float = 1.0,
    n_water: int = 1,
    seed: int = 0,
) -> list[list[str]]:
    """
    Generate n valid random maps with complexity < max_complexity.

    For each map:
      - p_burn  ~ Uniform(0, max_fire_density)
      - p_road  ~ Uniform(0, 1 - p_burn)   (fully random road/veg split)
      - p_veg   = 1 - p_burn - p_road
    Maps are rejected and resampled if complexity >= max_complexity so that
    E6 always remains the hardest (last) stage.
    """
    rng = random.Random(seed)
    maps, attempts = [], 0
    while len(maps) < n:
        p_burn    = rng.uniform(0.0, max_fire_density)
        remaining = 1.0 - p_burn
        p_road    = rng.uniform(0.0, remaining)
        p_veg     = remaining - p_road
        env_seed  = rng.randint(0, 2**31 - 1)
        desc      = generate_random_map(
            size=size, p_road=p_road, p_veg=p_veg, n_water=n_water, seed=env_seed
        )
        attempts += 1
        if compute_complexity(desc) < max_complexity:
            maps.append(desc)
        if attempts > n * 500:
            raise RuntimeError(
                f"Could not generate {n} maps with complexity < {max_complexity:.4f} "
                f"after {attempts} attempts. Try lowering --max-fire-density."
            )
    return maps


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_random_config(
    n_random: int,
    target_env: dict,
    size: int = 8,
    max_fire_density: float = 0.30,
    n_water: int = 1,
    seed: int = 0,
) -> dict:
    """
    Build a config dict: n_random randomly generated environments + fixed target.

    Environments are stored in raw generation order.  Ordering at training time
    (sorted by complexity vs random order) is controlled by the `ordered` flag
    passed to build_curriculum() in run_comparison.py — not by this file.

    All random envs are guaranteed complexity < target's complexity so E6 stays last
    regardless of whether curriculum_builder sorts or not.
    """
    target_complexity = target_env.get("complexity")
    if target_complexity is None:
        target_complexity = compute_complexity(target_env["desc"])
    maps         = generate_random_envs(
        n_random, size, max_fire_density,
        max_complexity=target_complexity,
        n_water=n_water, seed=seed,
    )
    complexities = [compute_complexity(m) for m in maps]
    reward_schedule = target_env["reward_schedule"]

    environments = [
        {
            "env_id":          f"env_{i + 1}",
            "desc":            desc,
            "reward_schedule": reward_schedule,
            "complexity":      complexity,
        }
        for i, (desc, complexity) in enumerate(zip(maps, complexities))
    ]
    # Target (E6) appended last; its high original complexity keeps it last
    # whether curriculum_builder sorts or not.
    environments.append({
        "env_id":          f"env_{n_random + 1}",
        "desc":            target_env["desc"],
        "reward_schedule": target_env["reward_schedule"],
        "complexity":      target_complexity,
    })

    return {"environments": environments}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def output_path(output_dir: str, size: int, index: int) -> str:
    """Return canonical output path for the i-th random config (1-indexed)."""
    return os.path.join(output_dir, f"environments-{size}x{size}-random-{index:02d}.json")


def print_summary(config: dict, path: str):
    print(f"  → {path}")
    print(f"    {'env_id':<8} {'fire_density':>14} {'complexity':>22}")
    print("    " + "-" * 46)
    for env in config["environments"]:
        fd = compute_fire_density(env["desc"])
        print(f"    {env['env_id']:<8} {fd:>14.4f} {env['complexity']:>22.10f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate random environment configs for multi-seed comparison."
    )
    parser.add_argument(
        "--target-config",
        default=str(DEFAULT_ENVIRONMENT_CONFIG),
        help="Shared environment config; target_environment_id selects the target.",
    )
    parser.add_argument(
        "--output-dir",
        default="../01-configurations/random-configs",
        help="Directory in which to write config files (used with --all-seeds / --seed-index).",
    )
    parser.add_argument(
        "--output",
        help="Exact output path for a single config (used with --seed).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all-seeds", action="store_true",
        help="Generate configs for all seeds derived from --master-seed.",
    )
    mode.add_argument(
        "--seed-index", type=int, metavar="N",
        help="Generate config for the N-th seed (1-indexed, 1–30).",
    )
    mode.add_argument(
        "--seed", type=int,
        help="Generate a single config with an explicit seed value.",
    )

    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--max-fire-density", type=float, default=0.30)
    parser.add_argument("--n-water", type=int, default=1)
    parser.add_argument(
        "--master-seed", type=int, default=42,
        help="Master seed used to derive all env-generation seeds (default: 42).",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=30,
        help="How many seeds to derive from --master-seed (default: 30).",
    )
    args = parser.parse_args()

    seeds = make_seeds(args.master_seed, args.n_seeds)

    with open(args.target_config) as f:
        original_cfg = json.load(f)
    target_env = target_environment(original_cfg)

    def make_and_save(seed: int, out_path: str):
        cfg = build_random_config(
            n_random=args.n_random,
            target_env=target_env,
            size=args.size,
            max_fire_density=args.max_fire_density,
            n_water=args.n_water,
            seed=seed,
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print_summary(cfg, out_path)

    if args.all_seeds:
        if not args.output_dir:
            parser.error("--all-seeds requires --output-dir")
        print(f"master_seed={args.master_seed} → {len(seeds)} seeds: {seeds}\n")
        for i, seed in enumerate(seeds, start=1):
            make_and_save(seed, output_path(args.output_dir, args.size, i))

    elif args.seed_index is not None:
        if not 1 <= args.seed_index <= len(seeds):
            parser.error(f"--seed-index must be between 1 and {len(seeds)}")
        if not args.output_dir:
            parser.error("--seed-index requires --output-dir")
        make_and_save(seeds[args.seed_index - 1], output_path(args.output_dir, args.size, args.seed_index))

    else:  # --seed (explicit single seed, bypasses derived list)
        if not args.output:
            parser.error("--seed requires --output")
        make_and_save(args.seed, args.output)


if __name__ == "__main__":
    main()
