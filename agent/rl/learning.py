"""Small SB3 DQN runner and sequential random-policy comparison on one instance."""

import json
from pathlib import Path
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from .observation import schema


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_model(model, path):
    path = Path(path)
    model.save(path)
    model.save_replay_buffer(path.with_suffix(".replay.pkl"))
    path.with_suffix(".schema.json").write_text(json.dumps(schema(), indent=2) + "\n")


def load_model(path, env=None, replay=None):
    path = Path(path)
    stored = json.loads(path.with_suffix(".schema.json").read_text())
    if stored != json.loads(json.dumps(schema())):
        raise ValueError("Model observation schema does not match this RL encoder")
    model = DQN.load(path, env=env, device="cpu")
    if replay is not None:
        model.load_replay_buffer(replay)
    return model


class TrainingLog(BaseCallback):
    def __init__(self, output, checkpoint_every):
        super().__init__()
        self.output = Path(output)
        self.checkpoint_every = checkpoint_every
        self.stream = None

    def _on_training_start(self):
        self.stream = (self.output / "transitions.jsonl").open("a")

    def _on_step(self):
        for info in self.locals["infos"]:
            row = dict(info)
            row["training_step"] = self.num_timesteps
            row["exploration_rate"] = self.model.exploration_rate
            row["reward"] = float(self.locals["rewards"][0])
            self.stream.write(json.dumps(row, allow_nan=False, default=json_value) + "\n")
        self.stream.flush()
        if self.n_calls % self.checkpoint_every == 0:
            save_model(self.model, self.output / f"dqn_{self.num_timesteps}")
        return True

    def close(self):
        if self.stream is not None:
            self.stream.close()


def train(env, output, steps=10_000, seed=0, checkpoint_every=1000,
          learning_starts=500, buffer_size=20_000, resume=None, replay=None):
    if min(steps, checkpoint_every, buffer_size) < 1 or learning_starts < 0:
        raise ValueError("Invalid training step/buffer limits")
    if replay is not None and resume is None:
        raise ValueError("Restoring replay requires a model checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "observation_schema.json").write_text(json.dumps(schema(), indent=2) + "\n")
    if resume is not None:
        model = load_model(resume, env, replay)
    else:
        model = DQN("MlpPolicy", env, device="cpu", seed=seed, learning_rate=1e-4,
                    buffer_size=buffer_size, learning_starts=learning_starts,
                    batch_size=32, gamma=0.99, train_freq=4, gradient_steps=1,
                    target_update_interval=500, exploration_fraction=0.3,
                    exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64]),
                    verbose=1)
    model.set_logger(configure(str(output), ["stdout", "csv"]))
    callback = TrainingLog(output, checkpoint_every)
    status, error = "completed", None
    try:
        model.learn(total_timesteps=steps, callback=callback,
                    reset_num_timesteps=resume is None, log_interval=1)
    except BaseException as exc:
        status, error = "interrupted", str(exc)
        raise
    finally:
        callback.close()
        model.logger.close()
        # Preserve completed replay transitions even when focus/capture stops a run.
        save_model(model, output / "dqn_final")
        (output / "training.json").write_text(json.dumps(dict(
            status=status, error=error, timesteps=model.num_timesteps, seed=model.seed,
            requested_steps=steps, buffer_size=model.buffer_size,
            learning_starts=model.learning_starts,
        ), indent=2) + "\n")
    return model


def evaluate(env, model, output, episodes=5, seed=1000):
    """Alternate policy order; one desktop never serves two active environments."""
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output / "evaluation_episodes.jsonl").open("x") as stream:
        for episode in range(episodes):
            order = ("random", "dqn") if episode % 2 == 0 else ("dqn", "random")
            for policy in order:
                episode_seed = seed + episode
                rng = np.random.default_rng(episode_seed)
                observation, _ = env.reset(seed=episode_seed)
                total, steps = 0.0, 0
                while True:
                    if policy == "random":
                        action = int(rng.integers(env.action_space.n))
                    else:
                        action, _ = model.predict(observation, deterministic=True)
                        action = int(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    steps += 1
                    if terminated or truncated:
                        break
                row = dict(policy=policy, episode=episode, seed=episode_seed, reward=total,
                           survival_seconds=info["survival_seconds"], steps=steps,
                           terminated=terminated, truncated=truncated)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{policy}: {row['survival_seconds']:.2f}s, reward {total:.2f}", flush=True)
    summary = dict(episodes_per_policy=episodes,
                   caveat="Live browser resets are not seeded game states; survival is UI-detector dependent.")
    for policy in ("random", "dqn"):
        survival, rewards, truncations = [], [], 0
        for row in rows:
            if row["policy"] == policy:
                survival.append(row["survival_seconds"])
                rewards.append(row["reward"])
                truncations += row["truncated"]
        summary[policy] = dict(mean_survival_seconds=float(np.mean(survival)),
                               std_survival_seconds=float(np.std(survival)),
                               mean_reward=float(np.mean(rewards)), truncations=truncations)
    summary["dqn_minus_random_survival_seconds"] = (
        summary["dqn"]["mean_survival_seconds"] - summary["random"]["mean_survival_seconds"])
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
