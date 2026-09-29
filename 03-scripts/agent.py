"""
Tabular Q-learning agent used in all curriculum and baseline experiments.
"""

import numpy as np
import json

class QLearningAgent:
    """
    Tabular Q-learning.
    """
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, seed=None):
        self.q_table = np.zeros((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.RandomState(seed)

    def predict(self, state, deterministic=False):
        """Return an action for *state*. Uses epsilon-greedy unless deterministic=True."""
        if not deterministic and self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, terminated):
        """Apply a single Q-learning update."""
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def decay_epsilon(self):
        """Multiply epsilon by decay rate, clamped at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Persist the Q-table to *path* (numpy .npy format)."""
        np.save(path, self.q_table)

    def load(self, path):
        """Load a Q-table from *path* (numpy .npy format)."""
        self.q_table = np.load(path)