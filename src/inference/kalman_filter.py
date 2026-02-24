import numpy as np


class KalmanFilter:
    def __init__(self, A, C, Q, R, mu0, P0):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.mu0 = mu0
        self.P0 = P0

        self.reset()

    def reset(self):
        self.mu = self.mu0.copy()
        self.P = self.P0.copy()

    def predict(self):
        """
        z_t|t-1 = A z_t-1
        P_t|t-1 = A P_t-1 A^T + Q
        """
        self.mu = self.A @ self.mu
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.mu, self.P

    def update(self, x):
        """
        Stable Joseph-form update
        """
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        innovation = x - self.C @ self.mu
        self.mu = self.mu + K @ innovation

        I = np.eye(self.P.shape[0])
        self.P = (
            (I - K @ self.C) @ self.P @ (I - K @ self.C).T
            + K @ self.R @ K.T
        )

        return innovation, S

    def filter(self, observations):
        """
        Returns:
        - filtered means
        - filtered covariances
        - predicted covariances
        - total log-likelihood
        """

        T = observations.shape[0]
        dim = self.mu.shape[0]

        mus = np.zeros((T, dim))
        covs = np.zeros((T, dim, dim))
        predicted_covs = np.zeros((T, dim, dim))

        log_likelihood = 0.0

        for t in range(T):

            _, P_pred = self.predict()
            predicted_covs[t] = P_pred.copy()

            innovation, S = self.update(observations[t])

            mus[t] = self.mu
            covs[t] = self.P

            log_likelihood += self._log_gaussian(innovation, S)

        return mus, covs, predicted_covs, log_likelihood

    @staticmethod
    def _log_gaussian(e, S):
        """
        Stable log-density using slogdet
        """
        d = e.shape[0]
        sign, logdet = np.linalg.slogdet(S)
        inv = np.linalg.inv(S)

        return (
            -0.5 * d * np.log(2 * np.pi)
            -0.5 * logdet
            -0.5 * e.T @ inv @ e
        )
