#!/usr/bin/env python3
"""
Implement the training
"""
import numpy as np
import random
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training.
    Args:
        env: the environment to train on
        nb_episodes: the number of episodes to train for
        alpha: the learning rate
        gamma: the discount factor
    Returns:
        all values of the score (sum of all rewards during one episode loop)
    """
    np.random.seed(0)
    random.seed(0)
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )

    scores = []

    for episode in range(nb_episodes):

        state, _ = env.reset(seed=0)

        episode_rewards = []

        episode_gradients = []

        done = False

        while not done:
            action, gradient = policy_gradient(
                state,
                weights
            )

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            episode_rewards.append(reward)

            episode_gradients.append(gradient)

            state = next_state

        G = 0
        returns = []

        for reward in reversed(episode_rewards):

            G = reward + gamma * G

            returns.insert(0, G)

        for gradient, G in zip(episode_gradients, returns):

            weights += alpha * G * gradient

        score = sum(episode_rewards)

        scores.append(score)

        print(f"Episode {episode}, Score: {score}")

    return scores
