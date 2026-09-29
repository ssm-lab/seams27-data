"""
Training and evaluation pipeline for curriculum learning experiments.
"""

import os
import numpy as np
from dataclasses import dataclass, field

import env
import gymnasium as gym
from config_loader import EnvSpec
from agent import QLearningAgent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Metrics collected during training on one environment."""
    env_id: str
    # Per-episode metrics (collected via callback)
    timesteps: list[int] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_completed: list[bool] = field(default_factory=list)
    # Periodic eval metrics (collected at checkpoint_frequency)
    eval_timesteps: list[int] = field(default_factory=list)
    eval_mean_rewards: list[float] = field(default_factory=list)
    eval_success_rates: list[float] = field(default_factory=list)
    # Saved agent paths
    checkpoint_paths: list[str] = field(default_factory=list)


@dataclass
class EpisodeOutcome:
    """Outcome of one completed (or budget-limited) training episode."""
    episodic_return: float
    success: bool
    steps: int
    completed: bool

# ---------------------------------------------------------------------------
# Make environment
# ---------------------------------------------------------------------------
def make_env(spec: EnvSpec, seed: int | None = None):
    """Create a BurningForest."""
    env = gym.make(
        "BurningForest-v0",
        desc=spec.desc,
        reward_schedule=spec.reward_schedule,
        # is_slippery=False,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env

# ---------------------------------------------------------------------------
# Create Q-learning agent
# ---------------------------------------------------------------------------
def create_agent(rl_cfg, env, seed):
    """Instantiate a QLearningAgent from the *rl* section of the learning config."""
    return QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        lr=rl_cfg["learning_rate"],
        gamma=rl_cfg["discount_factor"],
        epsilon=rl_cfg["epsilon_start"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        epsilon_min=rl_cfg["epsilon_min"],
        seed=seed,
    )

# ---------------------------------------------------------------------------
# Evaluate the trained agent
# ---------------------------------------------------------------------------
def evaluate_agent(agent, env, n_episodes: int = 100) -> tuple[float, float]:
    """
    Evaluate a trained agent on an environment.
    Returns:
        (mean_reward, success_rate)
    """
    total_rewards = []
    successes = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        if terminated and env.unwrapped.desc.flat[env.unwrapped.s] == b"G":
            successes += 1

    mean_reward = float(np.mean(total_rewards))
    success_rate = successes / n_episodes
    return mean_reward, success_rate


# ---------------------------------------------------------------------------
# Train one episode (used by the adaptive runtime)
# ---------------------------------------------------------------------------
def train_one_episode(
    agent: QLearningAgent,
    env,
    max_steps: int | None = None,
    on_step=None,
) -> EpisodeOutcome:
    """Train the existing agent for one episode and return its raw outcome.

    ``max_steps`` is used only for the final global-budget remainder. If that
    limit is reached before the environment terminates, the partial episode is
    deliberately not reported to EMF as a monitoring observation.
    """
    obs, _ = env.reset()
    episodic_return = 0.0
    episode_steps = 0

    while True:
        if max_steps is not None and episode_steps >= max_steps:
            return EpisodeOutcome(
                episodic_return=episodic_return,
                success=False,
                steps=episode_steps,
                completed=False,
            )

        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated)

        episodic_return += reward
        episode_steps += 1
        if on_step is not None:
            on_step()
        done = terminated or truncated

        if done:
            success = terminated and env.unwrapped.desc.flat[env.unwrapped.s] == b"G"
            agent.decay_epsilon()
            return EpisodeOutcome(
                episodic_return=episodic_return,
                success=success,
                steps=episode_steps,
                completed=True,
            )

        obs = next_obs


# ---------------------------------------------------------------------------
# Train on a single environment
# ---------------------------------------------------------------------------
def train_on_env(
    agent: QLearningAgent,
    env,
    eval_env,
    steps: int,
    env_id: str,
    checkpoint_dir: str,
    checkpoint_freq: int = 2000,
    eval_episodes: int = 20,
    global_step_offset: int = 0,
    verbose: int = 1,
) -> TrainResult:
    result = TrainResult(env_id=env_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0

    for step in range(1, steps + 1):
        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated)

        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

        if done:
            global_ts = step + global_step_offset
            result.timesteps.append(global_ts)
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(episode_length)
            result.episode_completed.append(True)
            agent.decay_epsilon()
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # Periodic checkpoint + eval <- remove checkpoint
        if (step + global_step_offset) % checkpoint_freq == 0:
            global_ts = step + global_step_offset

            # ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{global_ts}.npy")
            # agent.save(ckpt_path)
            # result.checkpoint_paths.append(ckpt_path)

            mean_reward, success_rate = evaluate_agent(agent, eval_env, eval_episodes)
            result.eval_timesteps.append(global_ts)
            result.eval_mean_rewards.append(mean_reward)
            result.eval_success_rates.append(success_rate)

            if verbose:
                print(f"[step {global_ts}] eval: mean_reward={mean_reward:.3f}, "
                      f"success_rate={success_rate:.3f}, epsilon={agent.epsilon:.3f}")

    # Count rewards earned before a stage budget cuts an episode short.
    if episode_length:
        result.timesteps.append(global_step_offset + steps)
        result.episode_rewards.append(episode_reward)
        result.episode_lengths.append(episode_length)
        result.episode_completed.append(False)

    # Save final
    final_path = os.path.join(checkpoint_dir, f"final_{env_id}.npy")
    agent.save(final_path)
    result.checkpoint_paths.append(final_path)

    return result



# ---------------------------------------------------------------------------
# Curriculum training: loop over envs
# ---------------------------------------------------------------------------
def run_curriculum(curriculum, eval_spec, learn_cfg, output_dir):
    """
    Train a single agent sequentially on every environment in *curriculum*.

    The agent's Q-table is carried forward between stages; only epsilon is reset.
    Returns the trained agent and a list of TrainResult, one per stage.
    """
    rl_cfg = learn_cfg["rl"]
    seed = learn_cfg["curriculum"].get("seed", 42)
    checkpoint_freq = learn_cfg["curriculum"].get("checkpoint_frequency", 2000)
    evaluation_episodes = learn_cfg.get("evaluation", {}).get("eval_episodes", 20)

    first_env = make_env(curriculum[0]["spec"], seed=seed)
    agent = create_agent(rl_cfg, first_env, seed=seed)
    first_env.close()

    all_results = []
    global_steps = 0

    for i, environment in enumerate(curriculum):
        spec = environment["spec"]
        steps = environment["steps"]
        print(f"\n=== Curriculum Environment {i+1}/{len(curriculum)}: {spec.env_id} "
              f"(complexity={spec.complexity}, steps={steps}) ===")

        # Reset exploration for new environment
        agent.epsilon = rl_cfg.get("epsilon_start", 1.0)

        env = make_env(spec, seed=seed + i)
        eval_env = make_env(eval_spec, seed=seed + 1000)

        result = train_on_env(
            agent=agent, env=env, eval_env=eval_env,
            steps=steps, env_id=spec.env_id,
            checkpoint_dir=os.path.join(output_dir, spec.env_id),
            checkpoint_freq=checkpoint_freq,
            eval_episodes=evaluation_episodes,
            global_step_offset=global_steps,
        )

        env.close()
        eval_env.close()
        all_results.append(result)
        global_steps += steps

    return agent, all_results


# ---------------------------------------------------------------------------
# Adaptive curriculum training: EMF controls environment activation
# ---------------------------------------------------------------------------
@dataclass
class _TrainingProgress:
    """Run-wide counters retained when the active stage changes."""
    episodes: int = 0
    steps: int = 0
    last_evaluation_step: int = 0


def run_adaptive_curriculum(bridge, eval_spec, learn_cfg, output_dir, *, max_interaction_steps=None):
    """Initialize one agent, train each activated stage, and save its policy.

    The engine supplies adaptation results. Python may end the run earlier at
    the experiment's total-step cap; partial episodes are not reported.
    """
    if max_interaction_steps is not None and max_interaction_steps <= 0:
        raise ValueError("max_interaction_steps must be positive")
    seed = learn_cfg["curriculum"].get("seed", 42)
    active_spec = bridge.initialize()
    env = make_env(active_spec, seed=seed)
    eval_env = None
    try:
        agent = create_agent(learn_cfg["rl"], env, seed=seed)
        eval_env = make_env(eval_spec, seed=seed + 1000)
        _validate_q_table_size(agent, eval_env)
        progress = _TrainingProgress()
        results = []

        while True:
            stage_result = TrainResult(env_id=active_spec.env_id)
            adaptation_result = _train_adaptive_stage(
                bridge, agent, env, eval_env, stage_result, progress,
                learn_cfg, max_interaction_steps,
            )
            _save_stage_policy(agent, stage_result, output_dir, len(results))
            results.append(stage_result)

            if adaptation_result is None or adaptation_result["instruction"] == "STOP":
                break
            if max_interaction_steps is not None and progress.steps >= max_interaction_steps:
                break

            # The stage trainer returns on PROCEED_TO_ENVIRONMENT, STOP, or budget exhaustion.
            env.close()
            active_spec = adaptation_result["environment"]
            env = make_env(active_spec, seed=seed + len(results))
            _validate_q_table_size(agent, env)
            agent.epsilon = learn_cfg["rl"]["epsilon_start"]

        return agent, results
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


def _train_adaptive_stage(bridge, agent, env, eval_env, stage_result, progress,
                          learn_cfg, max_interaction_steps):
    """Train until activation/stop; return None if the Python budget is exhausted."""
    def record_step():
        progress.steps += 1
        _evaluate_at_checkpoint(agent, eval_env, stage_result, progress, learn_cfg)

    while max_interaction_steps is None or progress.steps < max_interaction_steps:
        remaining = None if max_interaction_steps is None else max_interaction_steps - progress.steps
        episode_outcome = train_one_episode(agent, env, max_steps=remaining, on_step=record_step)
        if not episode_outcome.completed:
            stage_result.timesteps.append(progress.steps)
            stage_result.episode_rewards.append(episode_outcome.episodic_return)
            stage_result.episode_lengths.append(episode_outcome.steps)
            stage_result.episode_completed.append(False)
            return None

        progress.episodes += 1
        stage_result.timesteps.append(progress.steps)
        stage_result.episode_rewards.append(episode_outcome.episodic_return)
        stage_result.episode_lengths.append(episode_outcome.steps)
        stage_result.episode_completed.append(True)
        adaptation_result = bridge.report_episode_outcome(
            episode=progress.episodes,
            episodic_return=episode_outcome.episodic_return,
            success=episode_outcome.success,
        )
        if adaptation_result["instruction"] != "CONTINUE":
            return adaptation_result
    return None


def _evaluate_at_checkpoint(agent, eval_env, stage_result, progress, learn_cfg):
    """Keep evaluation on a global training-step schedule across activations."""
    frequency = learn_cfg["curriculum"].get("checkpoint_frequency", 2000)
    if progress.steps - progress.last_evaluation_step < frequency:
        return
    episodes = learn_cfg.get("evaluation", {}).get("eval_episodes", 20)
    mean_return, success_rate = evaluate_agent(agent, eval_env, episodes)
    stage_result.eval_timesteps.append(progress.steps)
    stage_result.eval_mean_rewards.append(mean_return)
    stage_result.eval_success_rates.append(success_rate)
    progress.last_evaluation_step = progress.steps


def _validate_q_table_size(agent, env):
    """Reject a discrete environment that cannot use the existing Q-table."""
    if (env.observation_space.n != agent.n_states
            or env.action_space.n != agent.n_actions):
        raise ValueError("environment state/action counts do not match the agent's Q-table")


def _save_stage_policy(agent, stage_result, output_dir, activation_index):
    """Save the Q-table after a stage and record its path for later evaluation."""
    stage_dir = os.path.join(output_dir, f"activation_{activation_index:02d}")
    os.makedirs(stage_dir, exist_ok=True)
    path = os.path.join(stage_dir, "final.npy")
    agent.save(path)
    stage_result.checkpoint_paths.append(path)


# ---------------------------------------------------------------------------
# Baseline training: single env, same total steps as CL
# ---------------------------------------------------------------------------
def run_baseline(
    target_spec: EnvSpec,
    total_steps: int,
    learn_cfg: dict,
    output_dir: str,
):
    """Train baseline: single environment, same total steps as curriculum."""
    rl_cfg = learn_cfg["rl"]
    seed = learn_cfg["curriculum"].get("seed", 42)
    checkpoint_freq = learn_cfg["curriculum"].get("checkpoint_frequency", 2000)
    evaluation_episodes = learn_cfg.get("evaluation", {}).get("eval_episodes", 20)

    env = make_env(target_spec, seed=seed)
    eval_env = make_env(target_spec, seed=seed + 1000)
    agent = create_agent(rl_cfg, env, seed=seed)

    print(f"\n=== Baseline Training: {target_spec.env_id} "
          f"(steps={total_steps}) ===")

    result = train_on_env(
        agent=agent,
        env=env,
        eval_env=eval_env,
        steps=total_steps,
        env_id=target_spec.env_id,
        checkpoint_dir=os.path.join(output_dir),
        checkpoint_freq=checkpoint_freq,
        eval_episodes=evaluation_episodes,
    )

    env.close()
    eval_env.close()
    return agent, result




# ---------------------------------------------------------------------------
# Final evaluation: curriculum vs baseline on target env
# ---------------------------------------------------------------------------
def run_final_evaluation(
    curriculum_agent: QLearningAgent,
    baseline_agent: QLearningAgent | None,
    eval_spec: EnvSpec,
    n_episodes: int,
    seed: int = 123,
) -> dict:
    """Evaluate curriculum and baseline agents on the target environment."""
    eval_env = make_env(eval_spec, seed=seed)

    print(f"\n=== Final Evaluation ({n_episodes} episodes) ===")
    results = {}

    curriculum_reward, curriculum_success = evaluate_agent(curriculum_agent, eval_env, n_episodes)
    results["curriculum"] = {
        "mean_reward": curriculum_reward,
        "success_rate": curriculum_success,
    }
    print(f"Curriculum: mean_reward={curriculum_reward:.3f}, success_rate={curriculum_success:.3f}")

    if baseline_agent is not None:
        base_reward, base_success = evaluate_agent(baseline_agent, eval_env, n_episodes)
        results["baseline"] = {
            "mean_reward": base_reward,
            "success_rate": base_success,
        }
        print(f"Baseline: mean_reward={base_reward:.3f}, success_rate={base_success:.3f}")

    return results
