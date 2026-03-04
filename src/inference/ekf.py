import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.
    """

    def __init__(self, f, h, F_jac, H_jac, Q, R, mu0, P0):

        self.f = f
        self.h = h

        self.F_jac = F_jac
        self.H_jac = H_jac

        self.Q = Q
        self.R = R

        self.mu = mu0
        self.P = P0

    def predict(self):

        F = self.F_jac(self.mu)

        self.mu = self.f(self.mu)
        self.P = F @ self.P @ F.T + self.Q

        return self.mu, self.P

    def update(self, observation):

        H = self.H_jac(self.mu)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        innovation = observation - self.h(self.mu)

        self.mu = self.mu + K @ innovation

        I = np.eye(self.P.shape[0])
        self.P = (
            (I - K @ H) @ self.P @ (I - K @ H).T
            + K @ self.R @ K.T
        )

        return innovation, S

    def filter(self, observations):

        T = observations.shape[0]
        dim = self.mu.shape[0]

        mus = np.zeros((T, dim))
        covs = np.zeros((T, dim, dim))

        for t in range(T):

            self.predict()
            self.update(observations[t])

            mus[t] = self.mu
            covs[t] = self.P

        return mus, covs
