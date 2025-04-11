#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    Args:
        Observations is a numpy.ndarray of shape (T,)
            T is the number of observations
        Transition is a numpy.ndarray of shape (M, M)
            M is the number of hidden states
        Emission is a numpy.ndarray of shape (M, N)
            N is the number of output states
        Initial is a numpy.ndarray of shape (M, 1)
        iterations is the number of times expectation-maximization
    Returns:
        the converged Transition, Emission, or None, None
    """
    if (not isinstance(Observations, np.ndarray)
            or len(Observations.shape) != 1):
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

    T = Observations.shape[0]
    M, N = Emission.shape

    for _ in range(iterations):
        # Forward
        alpha = np.zeros((M, T))
        alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            for j in range(M):
                alpha[j, t] = (
                    np.sum(alpha[:, t - 1] * Transition[:, j])
                    * Emission[j, Observations[t]]
                )

        # Backward
        beta = np.zeros((M, T))
        beta[:, -1] = 1
        for t in reversed(range(T - 1)):
            for i in range(M):
                beta[i, t] = (
                    np.sum(Transition[i, :] * Emission[:, Observations[t + 1]]
                           * beta[:, t + 1])
                )

        # xi and gamma
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denom = (
                np.sum(alpha[:, t] * np.dot(Transition,
                                            Emission[:, Observations[t + 1]]
                                            * beta[:, t + 1]))
            )
            if denom == 0:
                denom = 1e-16
            for i in range(M):
                numer = (
                    (alpha[i, t] * Transition[i, :] *
                        Emission[:, Observations[t + 1]] * beta[:, t + 1])
                )
                xi[i, :, t] = numer / denom

        gamma = np.sum(xi, axis=1)

        # Final gamma step
        final_denom = np.sum(alpha[:, -1] * beta[:, -1])
        if final_denom == 0:
            final_denom = 1e-16
        final_gamma = (alpha[:, -1] * beta[:, -1]) / final_denom
        gamma = np.hstack((gamma, final_gamma.reshape(-1, 1)))

        # Update Transition
        denom = np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
        denom[denom == 0] = 1e-16  # avoid division by zero
        Transition = np.sum(xi, axis=2) / denom

        # Update Emission
        for j in range(M):
            denom = np.sum(gamma[j, :])
            if denom == 0:
                denom = 1e-16
            for k in range(N):
                mask = Observations == k
                Emission[j, k] = np.sum(gamma[j, mask]) / denom

    return Transition, Emission
