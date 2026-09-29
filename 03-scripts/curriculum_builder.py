"""
Builds the curriculum artifact (ordered list of environments) from raw config dicts.
"""

from config_loader import EnvSpec
from generate_random_config import compute_complexity
def build_curriculum(env_cfg: dict, learn_cfg: dict, ordered: bool = True) -> list[dict]:
    """
    Build curriculum artifact from configs.
    Returns a list of dicts ordered by complexity from easy to hard.
        [
            {"spec": EnvSpec, "steps": int},
            ...
        ]
    The last one (highest complexity) will be the target & baseline environment.
    """
    steps_per_environment = learn_cfg["curriculum"]["steps_per_environment"]
    specs = []
    for env in env_cfg["environments"]:
        spec = EnvSpec(
            env_id=env["env_id"],
            desc=env["desc"],
            reward_schedule=tuple(env["reward_schedule"]),
            complexity=env["complexity"] if "complexity" in env else compute_complexity(env["desc"]),
        )
        specs.append(spec)

    # Sort by complexity easy → hard, unless caller passes ordered=False.
    # ordered=False preserves the JSON list order (used for random-order baselines).
    if ordered:
        specs.sort(key=lambda s: s.complexity)

    # TODO: curriculum builder, based on these ordered population, try to sampling the expected size of environments

    # Validate actual size matches config size
    expected_size = learn_cfg["curriculum"]["size"]
    if len(specs) != expected_size:
        raise ValueError(
            f"curriculum.size={expected_size} but found {len(specs)} environments"
        )

    # Build curriculum artifact
    curriculum = [{"spec": spec, "steps": steps_per_environment} for spec in specs]
    return curriculum


def get_target_spec(curriculum: list[dict]) -> EnvSpec:
    """Return the EnvSpec for the final (target) environment in the curriculum."""
    return curriculum[-1]["spec"]


def resolve_baseline_specs(
    curriculum: list[dict],
    env_indices: list[int] | str | None = None,
) -> dict[str, EnvSpec]:
    """
    Return a label→EnvSpec mapping for the requested baseline environments.

    env_indices=None  → target only; "all" → every env; list[int] → those indices.
    """
    if env_indices is None:
        target = curriculum[-1]["spec"]
        return {"Baseline (target)": target}

    indices = range(len(curriculum)) if env_indices == "all" else env_indices

    return {
        f"Baseline (complexity={curriculum[i]['spec'].complexity:.2f})": curriculum[i]["spec"]
        for i in indices
    }
