"""
Load environments.json and learning.json, build the curriculum artifact.
The curriculum is a list of (EnvSpec, steps) ordered by complexity.
"""
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENVIRONMENT_CONFIG = (Path(__file__).resolve().parents[1] /
    "01-configurations/environment-config.json")


@dataclass
class EnvSpec:
    """Specification for a single environment."""
    env_id: str
    desc: list[str]
    reward_schedule: tuple[float, ...]
    # Generated adaptive environments need not have a pre-computed complexity.
    complexity: float | None = None


def load_configs(env_config_path: str, learn_config_path: str) -> tuple[dict, dict]:
    """Load and return raw JSON dicts."""
    with open(env_config_path) as f:
        env_cfg = json.load(f)
    with open(learn_config_path) as f:
        learn_cfg = json.load(f)
    return env_cfg, learn_cfg


def target_environment(config):
    """Use the explicit target ID; older generated configs put the target last."""
    target_id = config.get("target_environment_id")
    if target_id is None:
        return config["environments"][-1]
    return next(environment for environment in config["environments"] if environment["env_id"] == target_id)


def load_target_spec(path=DEFAULT_ENVIRONMENT_CONFIG):
    with open(path) as stream:
        target = target_environment(json.load(stream))
    return EnvSpec(env_id=target["env_id"], desc=target["desc"],
                   reward_schedule=tuple(target["reward_schedule"]), complexity=target.get("complexity"))
