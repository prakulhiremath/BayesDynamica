import numpy as np
from .kalman_filter import KalmanFilter
from .kalman_smoother import KalmanSmoother


class EMLinearSSM:
    def __init__(self, model):
        self.model = model

    def fit(self, observations, n_iter=10):
        T = observations.shape[0]

        for iteration in range(n_iter):

            # --- E STEP ---
            kf = KalmanFilter(
                self.model.A,
                self.model.C,
                self.model.Q,
                self.model.R,
                self.model.mu0,
                self.model.P0,
            )

            mus, covs, _ = kf.filter(observations)

            ks = KalmanSmoother(self.model.A)
            smoothed_means, smoothed_covs = ks.smooth(mus, covs)

            # Expected cross-covariances
            Exx = smoothed_covs + np.einsum(
                "ti,tj->tij",
                smoothed_means,
                smoothed_means,
            )

            Exx_lag = np.zeros_like(Exx)

            for t in range(1, T):
                Exx_lag[t] = (
                    np.outer(smoothed_means[t], smoothed_means[t - 1])
                )

            # --- M STEP ---

            # Update A
            sum1 = np.sum(Exx_lag[1:], axis=0)
            sum2 = np.sum(Exx[:-1], axis=0)
            self.model.A = sum1 @ np.linalg.inv(sum2)

            # Update C
            sum1 = observations.T @ smoothed_means
            sum2 = np.sum(Exx, axis=0)
            self.model.C = sum1 @ np.linalg.inv(sum2)

            # Update Q
            Q_sum = np.zeros_like(self.model.Q)
            for t in range(1, T):
                diff = smoothed_means[t] - self.model.A @ smoothed_means[t - 1]
                Q_sum += np.outer(diff, diff)
            self.model.Q = Q_sum / (T - 1)

            # Update R
            R_sum = np.zeros_like(self.model.R)
            for t in range(T):
                diff = observations[t] - self.model.C @ smoothed_means[t]
                R_sum += np.outer(diff, diff)
            self.model.R = R_sum / T

        return self.model
