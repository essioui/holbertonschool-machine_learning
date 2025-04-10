#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the likely sequence of hidden states for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,)
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M)
            Emission[i, j] is the probability of observing
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N)
            Transition[i, j] is the probability of transitioning
        Initial a numpy.ndarray of shape (N, 1)
    Returns:
        path, P, or None, None on failure
            path is the a list of length T containing the most likely sequence
            P is the probability of obtaining the path sequence
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray)\
            or len(Transition.shape) != 2\
            or Transition.shape[0] != Transition.shape[1]:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    if Emission.shape[0] != Transition.shape[0] != Transition.shape[0] !=\
       Initial.shape[0]:
        return None, None

    if Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]

    N = Transition.shape[0]

    F = np.zeros((N, T))

    verti = np.zeros((T, N), dtype=int)

    F[:, 0] = (Initial.T * Emission[:, Observation[0]]).flatten()

    for t in range(1, Observation.shape[0]):
        for j in range(Transition.shape[0]):
            prob = F[:, t - 1] * Transition[:, j]

            F[j, t] = np.max(prob) * Emission[j, Observation[t]]

            verti[t, j] = np.argmax(prob)

    path = np.zeros(T, dtype=int)

    path[T - 1] = np.argmax(F[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = verti[t + 1, path[t + 1]]

    p = np.max(F[:, T - 1])

    return path.tolist(), p
