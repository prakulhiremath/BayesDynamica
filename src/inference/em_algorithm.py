import numpy as np
from .kalman_filter import KalmanFilter
from .kalman_smoother import KalmanSmoother


class EMLinearSSM:
    def __init__(self, model):
        self.model = model
        self.log_likelihoods = []

    def fit(self, observations, n_iter=50, tol=1e-4):
        T = observations.shape[0]

        prev_ll = -np.inf

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

            mus, covs, log_likelihood = kf.filter(observations)

            ks = KalmanSmoother(self.model.A)
            smoothed_means, smoothed_covs = ks.smooth(mus, covs)

            self.log_likelihoods.append(log_likelihood)

            # Convergence check
            if np.abs(log_likelihood - prev_ll) < tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_ll = log_likelihood

            # Expected sufficient statistics
            Exx = smoothed_covs + np.einsum(
                "ti,tj->tij",
                smoothed_means,
                smoothed_means,
            )

            Exx_lag = np.zeros_like(Exx)

            for t in range(1, T):
                Exx_lag[t] = np.outer(
                    smoothed_means[t],
                    smoothed_means[t - 1]
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
            self.model.Q = 0.5 * (self.model.Q + self.model.Q.T)

            # Update R
            R_sum = np.zeros_like(self.model.R)
            for t in range(T):
                diff = observations[t] - self.model.C @ smoothed_means[t]
                R_sum += np.outer(diff, diff)

            self.model.R = R_sum / T
            self.model.R = 0.5 * (self.model.R + self.model.R.T)

            # Optional: Stabilize A
            self._stabilize_transition()

        return self.model

    def _stabilize_transition(self):
        """
        Enforce spectral radius < 1 for stability.
        """
        eigvals = np.linalg.eigvals(self.model.A)
        spectral_radius = max(abs(eigvals))

        if spectral_radius >= 1:
            self.model.A = self.model.A / (1.05 * spectral_radius)
