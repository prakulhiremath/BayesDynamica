import numpy as np


class NonlinearStateSpaceModel:
    """
    General nonlinear state-space model.

    State equation:
        z_t = f(z_{t-1}) + w_t
        w_t ~ N(0, Q)

    Observation equation:
        x_t = h(z_t) + v_t
        v_t ~ N(0, R)
    """

    def __init__(self, f, h, Q, R, mu0, P0):
        """
        f   : state transition function
        h   : observation function
        Q   : process noise covariance
        R   : observation noise covariance
        mu0 : initial state mean
        P0  : initial state covariance
        """

        self.f = f
        self.h = h

        self.Q = Q
        self.R = R

        self.mu0 = mu0
        self.P0 = P0

    def generate(self, T):
        """
        Generate synthetic sequence from nonlinear model.
        """

        z_dim = self.mu0.shape[0]
        x_dim = self.R.shape[0]

        z = np.zeros((T, z_dim))
        x = np.zeros((T, x_dim))

        # initial state
        z[0] = np.random.multivariate_normal(self.mu0, self.P0)

        for t in range(1, T):

            noise = np.random.multivariate_normal(
                np.zeros(z_dim), self.Q
            )

            z[t] = self.f(z[t - 1]) + noise

        for t in range(T):

            obs_noise = np.random.multivariate_normal(
                np.zeros(x_dim), self.R
            )

            x[t] = self.h(z[t]) + obs_noise

        return z, x
