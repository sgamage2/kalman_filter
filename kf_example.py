# Application of Kalman filter using a Python lib

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
import kf
import utility


def create_kalman_filter(F, H, Q, R, x_init, P_init):
    M = x_init.shape[0]
    D = H.shape[0]

    filter = KalmanFilter(dim_x=M, dim_z=D)
    filter.x = x_init
    filter.P = P_init
    filter.F = F
    filter.H = H
    filter.R = R
    filter.Q = Q

    return filter


def create_ukf(F, H, Q, R, dt, x_init, P_init):
    M = x_init.shape[0]
    D = H.shape[0]
    # alpha, beta, kappa = 1, 2, -1
    alpha, beta, kappa = 1, 2, 1
    points = MerweScaledSigmaPoints(M, alpha=alpha, beta=beta, kappa=kappa)

    def fx(x, dt):
        return F @ x

    def hx(x):
        return H @ x

    ukf = UnscentedKalmanFilter(dim_x=M, dim_z=D, dt=dt, fx=fx, hx=hx, points=points)
    ukf.x = x_init
    ukf.P = P_init
    ukf.R = R
    ukf.Q = Q

    return ukf


if __name__ == "__main__":
    # -------------------------------------------
    # Setting parameters

    # Known paramaters are F, H, Q, R, x_init
    # No. of state variables = M = 2
    # No. of measurement variables = D = 1

    # F = np.array([[1., 0.],
    #               [0., 1.]])    # Process state transition matrix (MxM)
    # H = np.array([[1., 0.]])    # Measurement function matrix (DxM)
    # Q = np.array([[0.1, 0.],
    #               [0., 0.1]])  # Process noise covariance matrix (MxM)
    # R = np.array([[0.1]])  # Measurement noise covariance matrix (DxD)
    # x_init = np.array([[1.],
    #                    [3.]]) # Initial values of state variables (mean) (Mx1)
    # P_init = np.array([[0., 0.],
    #                    [0., 0.]])  # Initial values of covariance matrix of state variables (MxM)

    # Simple 1-D example (constant series)
    F = np.array([[1.]])  # Process state transition matrix (MxM)
    H = np.array([[1.]])  # Measurement function matrix (DxM)
    Q = np.array([[0.01]])  # Process noise covariance matrix (MxM)
    R = np.array([[0.01]])  # Measurement noise covariance matrix (DxD)
    x_init = np.array([[1.5]])  # Initial values of state variables (mean) (Mx1)
    M = x_init.shape[0]
    P_init = np.array([[0.01,]])  # Initial values of covariance matrix of state variables (MxM)
    x_true_init = np.array([[1.0]])
    dt = 0.1

    # MATLAB Kalman filtering example - need to include u
    # F = np.array([[1.1269, -0.4940, 0.1129],
    #                 [1.0, 0, 0],
    #                 [0, 1.0, 0]])  # Process state transition matrix (MxM)
    # H = np.array([[1., 0, 0]])  # Measurement function matrix (DxM)
    # Q = np.array([[0.005]])  # Process noise covariance matrix (MxM)
    # R = np.array([[0.1]])  # Measurement noise covariance matrix (DxD)
    # x_init = np.array([[3.], [3.], [3.]])  # Initial values of state variables (mean) (Mx1)
    # M = x_init.shape[0]
    # P_init = np.zeros((M, M))  # Initial values of covariance matrix of state variables (MxM)
    # x_true_init = np.array([[1.0], [1.0], [1.0]])

    n_samples = 100


    # -------------------------------------------
    # Generating data
    x_true_series = utility.generate_state_series(x_true_init, F, Q, n_samples) # M x n_samples
    print(x_true_series)

    z_true_series = utility.generate_measurement_series(x_true_series, H, R=None)
    # print(z_true_series)

    z_noisy_series = utility.generate_measurement_series(x_true_series, H, R)
    # print(z_noisy_series)

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)
    filter = create_kalman_filter(F, H, Q, R, x_init, P_init)
    my_filter = kf.KalmanFilter(F, H, R, Q, x_init, P_init)
    ukf = create_ukf(F, H, Q, R, dt, x_init, P_init)

    # Pre-allocate output variables
    filter_x = np.zeros((M, n_samples))
    my_filter_x = np.zeros((M, n_samples))
    ukf_x = np.zeros((M, n_samples))


    # -------------------------------------------
    # Kalman filtering

    for i in range(n_samples):
        # Time update (state prediction according to F, Q
        filter.predict()
        my_filter.predict()
        ukf.predict()

        # Measurement update (innovation) according to observed z, H, R
        z = z_noisy_series[:, i]
        filter.update(z)
        my_filter.update(z)
        ukf.update(z)

        # filter.x has shape Mx1
        filter_x[:, i] = filter.x[:, 0]
        my_filter_x[:, i] = my_filter.x[:, 0]
        ukf_x[:, i] = ukf.x[:]

    # -------------------------------------------
    # Results analysis

    x_var = range(n_samples)
    utility.plot(x_var, x_true_series[0, :], label='True state x_true_series (x_true_series)')
    utility.plot(x_var, my_filter_x[0, :], 'n', '', new_figure=False, label='My KF predicted state (kf_x)')
    utility.plot(x_var, filter_x[0, :], 'n', '', new_figure=False, label='filterpy KF predicted state (kf_x)', linestyle='--')
    # utility.plot(x_var, z_noisy_series[0, :],  new_figure=False, label='Noisy measurement (z_noisy_series)', linestyle='--', linewidth=1)
    plt.scatter(x_var, z_noisy_series[0, :], label='Noisy measurement (z_noisy_series)', marker='x', c='gray', s=10, alpha=0.7)

    # utility.plot(x_var, ukf_x[0, :], 'n', '', new_figure=False, label='filterpy UKF predicted state (kf_x)')

    plt.show()

