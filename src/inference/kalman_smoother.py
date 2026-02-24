import numpy as np


class KalmanSmoother:
    def __init__(self, A, Q):
        self.A = A
        self.Q = Q

    def smooth(
        self,
        filtered_means,
        filtered_covs,
        predicted_covs,
    ):
        """
        Full RTS smoother with lag-one cross-covariances.

        predicted_covs: P_{t|t-1} from forward pass
        """

        T, dim = filtered_means.shape

        smoothed_means = np.copy(filtered_means)
        smoothed_covs = np.copy(filtered_covs)

        cross_covs = np.zeros((T - 1, dim, dim))

        for t in reversed(range(T - 1)):

            P_t = filtered_covs[t]
            P_pred_next = predicted_covs[t + 1]

            # Smoother gain
            G = P_t @ self.A.T @ np.linalg.inv(P_pred_next)

            # Mean smoothing
            smoothed_means[t] = (
                filtered_means[t]
                + G @ (
                    smoothed_means[t + 1]
                    - self.A @ filtered_means[t]
                )
            )

            # Covariance smoothing
            smoothed_covs[t] = (
                P_t
                + G @ (
                    smoothed_covs[t + 1]
                    - P_pred_next
                ) @ G.T
            )

            # Cross covariance
            cross_covs[t] = (
                G @ smoothed_covs[t + 1]
            )

        return smoothed_means, smoothed_covs, cross_covs
