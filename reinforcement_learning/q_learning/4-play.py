#!/usr/bin/env python3
"""
Play
"""
import numpy as np
load_frozen_lake = __import__('0-load_env').load_frozen_lake


def play(env, Q, max_steps=100):
    """
    Play the FrozenLake environment using the Q-table
    Args:
        env: the environment to play
        Q: the Q-table
        max_steps: maximum number of steps to take in the environment
    Returns:
        The total rewards for the episode and a list of rendered outputs
    """
    original_env = env.unwrapped
    env = load_frozen_lake(
        desc=getattr(original_env, 'desc', None),
        map_name=getattr(original_env, 'map_name', None),
        is_slippery=getattr(original_env, 'is_slippery', False),
        render_mode="ansi"
    )
    
    state, _ = env.reset()
    render_output = [env.render()]
    
    total_rewards = 0
    
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        render_output.append(env.render())
        total_rewards += reward
        if truncated or terminated:
            break
        
    return total_rewards, render_output
