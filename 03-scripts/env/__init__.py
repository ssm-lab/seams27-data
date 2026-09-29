"""
Gymnasium environment package for the BurningForest wildfire-mitigation task.

Importing this package registers BurningForest-v0 with Gymnasium so it can be
created via gym.make("BurningForest-v0", ...).
"""

from gymnasium.envs.registration import register
from env.fire import BurningForest

register(
    id="BurningForest-v0",
    entry_point = BurningForest,
    max_episode_steps=200
)