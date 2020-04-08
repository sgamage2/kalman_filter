# Implement Kalman filter from scratch as a class

import numpy as np


class KalmanFilter:
    def __init__(self, F, H, R, Q, x_init, P_init):
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q
        self.x = x_init
        self.P = P_init

    # Time update (state prediction according to F, Q
    def predict(self):
        x = self.x
        P = self.P
        F = self.F
        Q = self.Q

        # x = Fx + Bu
        self.x = F @ x

        # P = FPF' + Q
        self.P = F @ P @ F.T + Q

    # Measurement update (innovation) according to observed z, H, R
    def update(self, z):
        x = self.x
        P = self.P
        H = self.H
        R = self.R

        # K = PH' inv(HPH' + R)
        inv = np.linalg.inv(H @ P @ H.T + R)
        K = P @ H.T @ inv # Kalman gain

        # x = x + K(z-Hx)
        self.x = x + K @ (z - H @ x)

        # P = P - KHP = (I-KH)P
        self.P = P - K @ H @ P

        # Alternative: P = (I-KH)P(I-KH)' + KRK'
        # I = np.eye(P.shape[0])
        # I_KH = (I - K @ H)
        # self.P = I_KH @ P @ I_KH.T + K @ R @ K.T

