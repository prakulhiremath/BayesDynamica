import numpy as np

class LinearStateSpaceModel:
    def __init__(self, A, C, Q, R, mu0, P0):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.mu0 = mu0
        self.P0 = P0

    def generate(self, T):
        z_dim = self.A.shape[0]
        x_dim = self.C.shape[0]

        z = np.zeros((T, z_dim))
        x = np.zeros((T, x_dim))

        z[0] = np.random.multivariate_normal(self.mu0, self.P0)

        for t in range(1, T):
            z[t] = self.A @ z[t-1] + \
                   np.random.multivariate_normal(
                       np.zeros(z_dim), self.Q
                   )

        for t in range(T):
            x[t] = self.C @ z[t] + \
                   np.random.multivariate_normal(
                       np.zeros(x_dim), self.R
                   )

        return z, x
