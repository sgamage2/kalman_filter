# Application of Kalman filter using a Python lib

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import kf


def plot(x, y, xlabel=None, y_label=None, title=None, new_figure=True, **kwargs):
    if new_figure:
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
    plt.plot(x, y, **kwargs)
    plt.legend()


def generate_state_series(x_init, F, Q, n_samples):
    """ Generates a series of state variables
        # Arguments
            x_init: Initial value of state variables (Mx1)
            F: Process state transition matrix (MxM)
            Q: Process noise covariance matrix (MxM)
        # Returns
            A numpy array (M x n_samples)
    """
    M = x_init.shape[0]
    x_series = np.zeros(shape=(M, n_samples))
    x_series[:, 0] = x_init[:, 0]

    for i in range(1, n_samples):
        x_series[:, i] = np.matmul(x_series[:, i-1], F)

    return x_series


def generate_measurement_series(x_series, H, R=None):
    """ Generates a series of measurements for a series of state variables
            # Arguments
                x_series: Series of state variables (M x n_samples)
                H: Measurement function matrix (DxM)
                R: Measurement noise covariance matrix (DxD)
            # Returns
                A numpy array (D x n_samples)
        """
    n_samples = x_series.shape[1]
    D = H.shape[0]

    z_series = np.matmul(H, x_series)   # D x n_samples

    if R is not None:
        for dim in range(D):
            mean = 0.
            var = R.diagonal()[dim]
            noise = np.random.normal(mean, var, n_samples)
            z_series[dim, :] += noise

    return z_series


def create_kalman_filter(x_init, P_init, F, H, Q, R):
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
    Q = np.array([[0.08]])  # Process noise covariance matrix (MxM)
    R = np.array([[0.1]])  # Measurement noise covariance matrix (DxD)
    x_init = np.array([[3.]])  # Initial values of state variables (mean) (Mx1)
    M = x_init.shape[0]
    P_init = np.array([[0.,]])  # Initial values of covariance matrix of state variables (MxM)
    x_true_init = np.array([[1.0]])

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
    x_true_series = generate_state_series(x_true_init, F, Q, n_samples) # M x n_samples
    print(x_true_series)

    z_true_series = generate_measurement_series(x_true_series, H, R=None)
    # print(z_true_series)

    z_noisy_series = generate_measurement_series(x_true_series, H, R)
    # print(z_noisy_series)

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)
    filter = create_kalman_filter(x_init, P_init, F, H, Q, R)
    my_filter = kf.KalmanFilter(F, H, Q, R, x_init, P_init)

    # Pre-allocate output variables
    filter_x = np.zeros((M, n_samples))
    my_filter_x = np.zeros((M, n_samples))


    # -------------------------------------------
    # Kalman filtering

    for i in range(n_samples):
        # Time update (state prediction according to F, Q
        filter.predict()
        my_filter.predict()

        # Measurement update (innovation) according to observed z, H, R
        z = z_noisy_series[:, i]
        filter.update(z)
        my_filter.update(z)

        # filter.x has shape Mx1
        filter_x[:, i] = filter.x[:, 0]
        my_filter_x[:, i] = my_filter.x[:, 0]

    # -------------------------------------------
    # Results analysis

    x_var = range(n_samples)
    plot(x_var, x_true_series[0, :] , label='True state x_true_series (x_true_series)')
    plot(x_var, filter_x[0, :], 'n', '', new_figure=False, label='filterpy KF predicted state (kf_x)')
    plot(x_var, my_filter_x[0, :], 'n', '', new_figure=False, label='My KF predicted state (kf_x)')
    plot(x_var, z_noisy_series[0, :],  new_figure=False, label='Noisy measurement (z_noisy_series)', linestyle='--', linewidth=1)

    plt.show()

