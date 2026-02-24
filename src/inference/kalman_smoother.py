import numpy as np


class KalmanSmoother:
    def __init__(self, A):
        self.A = A

    def smooth(self, filtered_means, filtered_covs):
        """
        Rauch-Tung-Striebel (RTS) Smoother
        """

        T, dim = filtered_means.shape

        smoothed_means = np.copy(filtered_means)
        smoothed_covs = np.copy(filtered_covs)

        for t in reversed(range(T - 1)):

            P = filtered_covs[t]
            P_next = filtered_covs[t + 1]

            # Smoother gain
            G = P @ self.A.T @ np.linalg.inv(
                self.A @ P @ self.A.T
            )

            smoothed_means[t] += G @ (
                smoothed_means[t + 1]
                - self.A @ filtered_means[t]
            )

            smoothed_covs[t] += G @ (
                smoothed_covs[t + 1]
                - P_next
            ) @ G.T

        return smoothed_means, smoothed_covs
