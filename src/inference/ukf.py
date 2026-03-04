import numpy as np


class UnscentedKalmanFilter:

    def __init__(self, f, h, Q, R, mu0, P0,
                 alpha=1e-3, beta=2, kappa=0):

        self.f = f
        self.h = h

        self.Q = Q
        self.R = R

        self.mu = mu0
        self.P = P0

        self.n = mu0.shape[0]

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

    def sigma_points(self):

        sigma_points = np.zeros((2*self.n + 1, self.n))

        S = np.linalg.cholesky(
            (self.n + self.lambda_) * self.P
        )

        sigma_points[0] = self.mu

        for i in range(self.n):
            sigma_points[i+1] = self.mu + S[:, i]
            sigma_points[self.n+i+1] = self.mu - S[:, i]

        return sigma_points

    def weights(self):

        Wm = np.full(2*self.n+1,
                     1/(2*(self.n+self.lambda_)))
        Wc = Wm.copy()

        Wm[0] = self.lambda_/(self.n+self.lambda_)
        Wc[0] = Wm[0] + (1-self.alpha**2+self.beta)

        return Wm, Wc

    def predict(self):

        sigma = self.sigma_points()
        Wm, Wc = self.weights()

        propagated = np.array([self.f(s) for s in sigma])

        mu_pred = np.sum(Wm[:, None] * propagated, axis=0)

        P_pred = self.Q.copy()

        for i in range(2*self.n+1):
            diff = propagated[i] - mu_pred
            P_pred += Wc[i] * np.outer(diff, diff)

        self.mu = mu_pred
        self.P = P_pred

        return sigma, propagated

    def update(self, observation, sigma, propagated):

        Wm, Wc = self.weights()

        obs_sigma = np.array([self.h(s) for s in propagated])

        z_pred = np.sum(Wm[:, None] * obs_sigma, axis=0)

        S = self.R.copy()
        Cxz = np.zeros((self.n, observation.shape[0]))

        for i in range(2*self.n+1):

            dz = obs_sigma[i] - z_pred
            dx = propagated[i] - self.mu

            S += Wc[i] * np.outer(dz, dz)
            Cxz += Wc[i] * np.outer(dx, dz)

        K = Cxz @ np.linalg.inv(S)

        innovation = observation - z_pred

        self.mu = self.mu + K @ innovation
        self.P = self.P - K @ S @ K.T

    def filter(self, observations):

        T = observations.shape[0]

        dim = self.mu.shape[0]

        mus = np.zeros((T, dim))
        covs = np.zeros((T, dim, dim))

        for t in range(T):

            sigma, prop = self.predict()
            self.update(observations[t], sigma, prop)

            mus[t] = self.mu
            covs[t] = self.P

        return mus, covs
