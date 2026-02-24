import numpy as np


class KalmanSmoother:
    def __init__(self, A, Q):
        self.A = A
        self.Q = Q

    def smooth(self, filtered_means, filtered_covs):
        """
        Rauch–Tung–Striebel (RTS) Smoother
        """

        T, dim = filtered_means.shape

        smoothed_means = np.copy(filtered_means)
        smoothed_covs = np.copy(filtered_covs)

        for t in reversed(range(T - 1)):

            P_t = filtered_covs[t]

            # Predicted covariance P_{t+1|t}
            P_pred = (
                self.A @ P_t @ self.A.T + self.Q
            )

            # Smoother gain
            G = P_t @ self.A.T @ np.linalg.inv(P_pred)

            # Mean update
            smoothed_means[t] = (
                filtered_means[t]
                + G @ (
                    smoothed_means[t + 1]
                    - self.A @ filtered_means[t]
                )
            )

            # Covariance update
            smoothed_covs[t] = (
                P_t
                + G @ (
                    smoothed_covs[t + 1]
                    - P_pred
                ) @ G.T
            )

        return smoothed_means, smoothed_covs
