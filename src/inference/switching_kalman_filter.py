import numpy as np
from .kalman_filter import KalmanFilter


class SwitchingKalmanFilter:
    def __init__(self, model):
        self.model = model
        self.K = model.K

    def filter(self, observations):
        T = observations.shape[0]
        dim = self.model.mu0.shape[0]

        # Initialize regime probabilities
        regime_probs = self.model.pi0.copy()

        # Initialize state estimates for each regime
        mus = [self.model.mu0.copy() for _ in range(self.K)]
        covs = [self.model.P0.copy() for _ in range(self.K)]

        # Storage
        regime_history = np.zeros((T, self.K))
        state_history = np.zeros((T, self.K, dim))

        log_likelihood = 0.0

        for t in range(T):

            # --- 1. Regime prediction ---
            regime_probs = self.model.Pi.T @ regime_probs

            likelihoods = np.zeros(self.K)
            new_mus = []
            new_covs = []

            # --- 2. Run K Kalman filters ---
            for k in range(self.K):

                kf = KalmanFilter(
                    self.model.A[k],
                    self.model.C[k],
                    self.model.Q[k],
                    self.model.R[k],
                    mus[k],
                    covs[k],
                )

                kf.predict()
                innovation, S = kf.update(observations[t])

                new_mus.append(kf.mu)
                new_covs.append(kf.P)

                # Likelihood under regime k
                likelihoods[k] = self._gaussian_pdf(innovation, S)

            # --- 3. Update regime probabilities ---
            regime_probs = regime_probs * likelihoods
            regime_probs /= np.sum(regime_probs)

            log_likelihood += np.log(np.sum(likelihoods))

            mus = new_mus
            covs = new_covs

            regime_history[t] = regime_probs
            for k in range(self.K):
                state_history[t, k] = mus[k]

        return regime_history, state_history, log_likelihood

    @staticmethod
    def _gaussian_pdf(e, S):
        d = e.shape[0]
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det)
        exponent = -0.5 * e.T @ inv @ e
        return norm_const * np.exp(exponent)
