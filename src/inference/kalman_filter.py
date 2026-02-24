import numpy as np


class KalmanFilter:
    def __init__(self, A, C, Q, R, mu0, P0):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.mu = mu0
        self.P = P0

    def predict(self):
        """
        Prediction step:
        z_t|t-1 = A z_t-1
        P_t|t-1 = A P_t-1 A^T + Q
        """
        self.mu = self.A @ self.mu
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x):
        """
        Update step:
        K = P C^T (C P C^T + R)^-1
        mu = mu + K (x - C mu)
        P = (I - K C) P
        """
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        innovation = x - self.C @ self.mu
        self.mu = self.mu + K @ innovation

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.C) @ self.P

        return innovation, S

    def filter(self, observations):
        """
        Run full filtering over sequence.
        Returns filtered means, covariances, log-likelihood.
        """
        T = observations.shape[0]
        dim = self.mu.shape[0]

        mus = np.zeros((T, dim))
        covs = np.zeros((T, dim, dim))

        log_likelihood = 0.0

        for t in range(T):
            self.predict()
            innovation, S = self.update(observations[t])

            mus[t] = self.mu
            covs[t] = self.P

            # log p(x_t | x_1:t-1)
            log_likelihood += self._log_gaussian(innovation, S)

        return mus, covs, log_likelihood

    @staticmethod
    def _log_gaussian(e, S):
        """
        Log density of N(0, S) evaluated at innovation e.
        """
        d = e.shape[0]
        term1 = -0.5 * d * np.log(2 * np.pi)
        term2 = -0.5 * np.log(np.linalg.det(S))
        term3 = -0.5 * e.T @ np.linalg.inv(S) @ e
        return term1 + term2 + term3
