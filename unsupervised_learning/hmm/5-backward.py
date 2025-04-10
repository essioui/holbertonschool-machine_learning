#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,)
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M)
            Emission[i, j]: the probability of observing j and hidden state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N)
            Transition[i, j] is the probability of transitioning
        Initial a numpy.ndarray of shape (N, 1)
    Returns:
        P, B, or None, None
            Pis the likelihood of the observations given the model
            B is a numpy.ndarray of shape (N, T)
                B[i, j] the probability of generating the future observations
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if Transition.shape[0] != Transition.shape[1]:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    if not (Emission.shape[0] == Transition.shape[0] == Initial.shape[0]):
        return None, None

    if Initial.shape[1] != 1:
        return None, None

    try:
        T = Observation.shape[0]
        N, _ = Emission.shape

        B = np.zeros((N, T))
        B[:, T - 1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(N):
                B[i, t] = np.sum(
                    (Transition[i, :] * Emission[:, Observation[t + 1]]
                        * B[:, t + 1])
                )

        p = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

        return p, B
    except Exception:
        return None, None
