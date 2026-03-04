import numpy as np
from .kalman_filter import KalmanFilter


class VariationalSwitchingKalmanFilter:
    """
    Variational inference for Switching Linear State Space Models.

    Approximates:
        p(z_{1:T}, s_{1:T} | x_{1:T})

    using factorization:
        q(z_{1:T}) q(s_{1:T})
    """

    def __init__(self, model, max_iter=20, tol=1e-4):
        self.model = model
        self.K = model.K
        self.max_iter = max_iter
        self.tol = tol

    def filter(self, observations):

        T = observations.shape[0]
        dim = self.model.mu0.shape[0]

        # Initialize regime probabilities
        regime_probs = np.ones((T, self.K)) / self.K

        state_means = np.zeros((T, dim))
        state_covs = np.zeros((T, dim, dim))

        prev_ll = -np.inf

        for iteration in range(self.max_iter):

            # ===== Expected dynamics =====

            A_bar = np.zeros_like(self.model.A[0])
            C_bar = np.zeros_like(self.model.C[0])
            Q_bar = np.zeros_like(self.model.Q[0])
            R_bar = np.zeros_like(self.model.R[0])

            for k in range(self.K):
                weight = np.mean(regime_probs[:, k])
                A_bar += weight * self.model.A[k]
                C_bar += weight * self.model.C[k]
                Q_bar += weight * self.model.Q[k]
                R_bar += weight * self.model.R[k]

            # ===== Kalman filtering =====

            kf = KalmanFilter(
                A_bar,
                C_bar,
                Q_bar,
                R_bar,
                self.model.mu0,
                self.model.P0,
            )

            mus, covs, _, log_likelihood = kf.filter(observations)

            state_means = mus
            state_covs = covs

            # ===== Update regime probabilities =====

            log_probs = np.zeros((T, self.K))

            for k in range(self.K):

                A = self.model.A[k]
                C = self.model.C[k]
                Q = self.model.Q[k]
                R = self.model.R[k]

                for t in range(T):

                    mu = mus[t]
                    P = covs[t]

                    pred_obs = C @ mu
                    S = C @ P @ C.T + R

                    innovation = observations[t] - pred_obs

                    sign, logdet = np.linalg.slogdet(S)
                    invS = np.linalg.inv(S)

                    ll = (
                        -0.5 * innovation.T @ invS @ innovation
                        -0.5 * logdet
                    )

                    log_probs[t, k] = ll

            # Normalize
            m = np.max(log_probs, axis=1, keepdims=True)
            probs = np.exp(log_probs - m)
            regime_probs = probs / np.sum(probs, axis=1, keepdims=True)

            if np.abs(log_likelihood - prev_ll) < self.tol:
                break

            prev_ll = log_likelihood

        return regime_probs, state_means, state_covs
