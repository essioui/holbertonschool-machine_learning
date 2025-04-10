#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
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
        P, F, or None, None on failure
            P is the likelihood of the observations
            F is a numpy.ndarray of shape (N, T)
                F[i, j] is the probability of being in hidden
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
    F[:, 0, np.newaxis] = (Initial.T * Emission[:, Observation[0]]).T

    for t in range(1, Observation.shape[0]):
        for j in range(Transition.shape[0]):
            F[j, t] = (
                (F[:, t - 1].dot(Transition[:, j])
                 * Emission[j, Observation[t]])
            )

    prob = np.sum(F[:, -1])

    return prob, F
