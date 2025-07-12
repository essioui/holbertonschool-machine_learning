#!/usr/bin/env python3
"""
SARSA(λ) algorithm implementation.
"""
import numpy as np


def sarsa_lambtha(
    env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
    epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """
    Implements the SARSA(λ) algorithm.
    Args:
        env: The environment to train on.
        Q: Initial action-value function.
        lambtha: The lambda parameter for eligibility traces.
        episodes: Number of episodes to train.
        max_steps: Maximum steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        min_epsilon: Minimum exploration rate.
        epsilon_decay: Decay rate for epsilon.
    Returns:
        Q, the updated Q table
    """
    n_states, n_actions = Q.shape

    def choose_action(state, epsilon):
        """
        Choose an action using epsilon-greedy policy.
        """
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        return np.argmax(Q[state])

    for ep in range(episodes):
        state, _ = env.reset()
        action = choose_action(state, epsilon)

        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            next_action = choose_action(next_state, epsilon)

            TD_error = reward + gamma * Q[next_state, next_action] * (not done)
            - Q[state, action]

            eligibility[state, action] += 1

            Q += alpha * TD_error * eligibility

            eligibility *= gamma * lambtha

            if done:
                break

            state, action = next_state, next_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

        return Q
