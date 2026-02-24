import numpy as np
from .kalman_filter import KalmanFilter


class SwitchingKalmanFilter:
    def __init__(self, model):
        self.model = model
        self.K = model.K

    def filter(self, observations):
        T = observations.shape[0]
        dim = self.model.mu0.shape[0]

        regime_probs = self.model.pi0.copy()
        log_regime_probs = np.log(regime_probs + 1e-16)

        mus = [self.model.mu0.copy() for _ in range(self.K)]
        covs = [self.model.P0.copy() for _ in range(self.K)]

        regime_history = np.zeros((T, self.K))
        state_history = np.zeros((T, self.K, dim))

        total_log_likelihood = 0.0

        for t in range(T):

            # --- MIXING STEP ---
            mixed_mus = []
            mixed_covs = []

            for j in range(self.K):

                denom = np.sum(
                    self.model.Pi[:, j] * np.exp(log_regime_probs)
                )

                mixing_probs = (
                    self.model.Pi[:, j] * np.exp(log_regime_probs)
                ) / (denom + 1e-16)

                mu_bar = np.zeros(dim)
                for i in range(self.K):
                    mu_bar += mixing_probs[i] * mus[i]

                P_bar = np.zeros((dim, dim))
                for i in range(self.K):
                    diff = mus[i] - mu_bar
                    P_bar += mixing_probs[i] * (
                        covs[i] + np.outer(diff, diff)
                    )

                mixed_mus.append(mu_bar)
                mixed_covs.append(P_bar)

            # --- REGIME PREDICTION (log space) ---
            log_regime_pred = np.log(
                self.model.Pi.T @ np.exp(log_regime_probs) + 1e-16
            )

            log_likelihoods = np.zeros(self.K)
            new_mus = []
            new_covs = []

            # --- RUN FILTERS ---
            for k in range(self.K):

                kf = KalmanFilter(
                    self.model.A[k],
                    self.model.C[k],
                    self.model.Q[k],
                    self.model.R[k],
                    mixed_mus[k],
                    mixed_covs[k],
                )

                kf.predict()
                innovation, S = kf.update(observations[t])

                new_mus.append(kf.mu)
                new_covs.append(kf.P)

                log_likelihoods[k] = self._log_gaussian(
                    innovation, S
                )

            # --- LOG PROB UPDATE ---
            log_regime_post = log_regime_pred + log_likelihoods

            # log-sum-exp normalization
            m = np.max(log_regime_post)
            log_sum = m + np.log(
                np.sum(np.exp(log_regime_post - m))
            )

            log_regime_probs = log_regime_post - log_sum
            regime_probs = np.exp(log_regime_probs)

            total_log_likelihood += log_sum

            mus = new_mus
            covs = new_covs

            regime_history[t] = regime_probs
            for k in range(self.K):
                state_history[t, k] = mus[k]

        return regime_history, state_history, total_log_likelihood

    @staticmethod
    def _log_gaussian(e, S):
        d = e.shape[0]
        sign, logdet = np.linalg.slogdet(S)
        inv = np.linalg.inv(S)

        return (
            -0.5 * d * np.log(2 * np.pi)
            -0.5 * logdet
            -0.5 * e.T @ inv @ e
        )
