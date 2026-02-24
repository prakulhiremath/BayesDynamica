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

            # ===== E STEP =====
            kf = KalmanFilter(
                self.model.A,
                self.model.C,
                self.model.Q,
                self.model.R,
                self.model.mu0,
                self.model.P0,
            )

            mus, covs, predicted_covs, log_likelihood = \
                kf.filter(observations)

            ks = KalmanSmoother(self.model.A, self.model.Q)

            sm_means, sm_covs, cross_covs = ks.smooth(
                mus,
                covs,
                predicted_covs,
            )

            self.log_likelihoods.append(log_likelihood)

            if np.abs(log_likelihood - prev_ll) < tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_ll = log_likelihood

            # ===== Sufficient Statistics =====

            Exx = sm_covs + np.einsum(
                "ti,tj->tij",
                sm_means,
                sm_means,
            )

            Exx_lag = np.zeros((T - 1,
                                self.model.A.shape[0],
                                self.model.A.shape[1]))

            for t in range(T - 1):
                Exx_lag[t] = (
                    cross_covs[t]
                    + np.outer(sm_means[t + 1],
                               sm_means[t])
                )

            # ===== M STEP =====

            # ---- Update A ----
            sum1 = np.sum(Exx_lag, axis=0)
            sum2 = np.sum(Exx[:-1], axis=0)

            self.model.A = sum1 @ np.linalg.inv(sum2)

            # ---- Update C ----
            sum1 = observations.T @ sm_means
            sum2 = np.sum(Exx, axis=0)

            self.model.C = sum1 @ np.linalg.inv(sum2)

            # ---- Update Q ----
            Q_sum = np.zeros_like(self.model.Q)

            for t in range(T - 1):
                term = (
                    Exx[t + 1]
                    - self.model.A @ Exx_lag[t].T
                )
                Q_sum += term

            self.model.Q = Q_sum / (T - 1)
            self.model.Q = 0.5 * (self.model.Q + self.model.Q.T)

            # ---- Update R ----
            R_sum = np.zeros_like(self.model.R)

            for t in range(T):
                obs = observations[t]
                R_sum += (
                    np.outer(obs, obs)
                    - self.model.C
                    @ sm_means[t][:, None]
                    @ obs[None, :]
                )

            R_sum += (
                self.model.C
                @ np.sum(Exx, axis=0)
                @ self.model.C.T
            )

            self.model.R = R_sum / T
            self.model.R = 0.5 * (self.model.R + self.model.R.T)

            # Stability enforcement
            self._stabilize_transition()

        return self.model

    def _stabilize_transition(self):
        eigvals = np.linalg.eigvals(self.model.A)
        spectral_radius = max(abs(eigvals))

        if spectral_radius >= 1:
            self.model.A = \
                self.model.A / (1.05 * spectral_radius)
