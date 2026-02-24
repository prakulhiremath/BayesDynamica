import numpy as np


class SwitchingLinearSSM:
    def __init__(self, A_list, C_list, Q_list, R_list, Pi, mu0, P0, pi0):
        """
        A_list: list of transition matrices (one per regime)
        C_list: list of emission matrices
        Q_list: list of process covariances
        R_list: list of observation covariances
        Pi: regime transition matrix
        pi0: initial regime probabilities
        """
        self.A = A_list
        self.C = C_list
        self.Q = Q_list
        self.R = R_list
        self.Pi = Pi
        self.mu0 = mu0
        self.P0 = P0
        self.pi0 = pi0
        self.K = len(A_list)
