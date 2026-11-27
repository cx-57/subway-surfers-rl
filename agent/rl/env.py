"""Gymnasium adapter. A backend owns one instance's capture, controls and lifecycle."""

from dataclasses import dataclass
import math
from typing import Protocol
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agent.state import ACTIONS, GameState
from .observation import FIELDS, ObservationEncoder


@dataclass
class Sample:
    state: GameState
    terminal: bool = False
    terminal_score: float = 0.0
    terminal_at: float | None = None
    observation_wait_seconds: float = 0.0


class Backend(Protocol):
    def reset(self) -> Sample: ...
    def act(self, action: int) -> tuple[float, float]: ...
    def observe(self, after: float) -> Sample: ...
    def close(self): ...


class SubwayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, backend: Backend, action_seconds=0.12, max_episode_seconds=120.0,
                 death_penalty=1.0, max_observation_age=0.5, max_transition_seconds=2.0):
        super().__init__()
        for value in (action_seconds, max_episode_seconds, max_observation_age,
                      max_transition_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Environment timing limits must be positive and finite")
        if not math.isfinite(death_penalty) or death_penalty < 0:
            raise ValueError("Death penalty must be finite and nonnegative")
        self.backend = backend
        self.action_seconds = action_seconds
        self.max_episode_seconds = max_episode_seconds
        self.death_penalty = death_penalty
        self.max_age = max_observation_age
        self.max_transition_seconds = max_transition_seconds
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(len(FIELDS),), dtype=np.float32)
        self.encoder = ObservationEncoder()
        self.finished = True

    def _validate(self, sample):
        age = sample.state.observation_age
        if not math.isfinite(age) or not 0 <= age <= self.max_age:
            raise RuntimeError("RL observation is stale; transition was not recorded")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.finished = True
        self.encoder.reset()
        sample = self.backend.reset()
        self._validate(sample)
        if sample.terminal:
            raise RuntimeError("Reset returned a game-over screen")
        self.start_timestamp = sample.state.timestamp
        self.last_timestamp = self.start_timestamp
        self.episode_reward = 0.0
        self.survival_seconds = 0.0
        self.steps = 0
        observation = self.encoder.encode(sample.state)
        self.finished = False
        return observation, self._info(sample)

    def _info(self, sample):
        return dict(observation_timestamp=sample.state.timestamp,
                    observation_age=sample.state.observation_age,
                    survival_seconds=self.survival_seconds,
                    episode_wall_seconds=sample.state.timestamp - self.start_timestamp,
                    game_over_score=sample.terminal_score,
                    episode_steps=self.steps, episode_reward=self.episode_reward)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 4")
        # Backend returns actual key issue/completion times in the capture clock domain.
        try:
            issued_at, finished_at = self.backend.act(int(action))
            if not self.last_timestamp <= issued_at <= finished_at:
                raise RuntimeError("Action clock precedes its source observation")
            sample = self.backend.observe(finished_at + self.action_seconds)
            self._validate(sample)
            timestamp = sample.state.timestamp
            dt = timestamp - self.last_timestamp
            waited = sample.observation_wait_seconds
            if (not math.isfinite(waited) or waited < 0 or waited >= dt
                    or (waited > 0 and (not sample.terminal or sample.terminal_at is None))):
                raise RuntimeError("Invalid terminal observation wait")
            if timestamp < finished_at + self.action_seconds or not 0 < dt - waited <= self.max_transition_seconds:
                raise RuntimeError("Unsynchronized or delayed transition; transition was not recorded")
            survival = timestamp - self.start_timestamp
            if sample.terminal_at is not None:
                if (not sample.terminal or not math.isfinite(sample.terminal_at)
                        or not self.start_timestamp <= sample.terminal_at <= timestamp):
                    raise RuntimeError("Invalid inferred terminal timestamp")
                survival = sample.terminal_at - self.start_timestamp
            # Correct earlier provisional survival credit if death UI arrived late.
            reward = survival - self.survival_seconds - (self.death_penalty if sample.terminal else 0.0)
            terminated = bool(sample.terminal)
            truncated = bool(not terminated and timestamp - self.start_timestamp >= self.max_episode_seconds)
            observation = self.encoder.encode(sample.state, int(action))
        except Exception:
            self.finished = True
            raise
        self.last_timestamp = timestamp
        self.survival_seconds = survival
        self.steps += 1
        self.episode_reward += reward
        self.finished = terminated or truncated
        info = self._info(sample)
        info.update(action=ACTIONS[int(action)], action_issued_at=issued_at,
                    action_finished_at=finished_at, transition_seconds=dt,
                    terminal_at=sample.terminal_at, terminal_wait_seconds=waited,
                    termination_reason="game_over" if terminated else "time_limit" if truncated else None)
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.finished = True
        self.backend.close()
