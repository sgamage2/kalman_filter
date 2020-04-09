# Application of Kalman filter using a Python lib

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from nolitsa import data
import kf
import utility
from keras.models import Sequential
from keras.layers import Dense, Dropout
import math, os

epochs = 50


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


def create_ukf(fx_model, H, Q, R, dt, x_init, P_init):
    M = x_init.shape[0]
    D = H.shape[0]
    # alpha, beta, kappa = 1, 2, -1 # Division by zero error
    # alpha, beta, kappa = 1, 2, 1  # Worked well
    alpha, beta, kappa = 0.001, 2, 1
    points = MerweScaledSigmaPoints(M, alpha=alpha, beta=beta, kappa=kappa)

    def fx(x, dt):
        # assert False    # Need the 4 x 4 F matrix here
        M = len(x)
        output = np.zeros(M)
        y = fx_model.predict(x.reshape(1, M))   # Reshape needed to feed x as 1 sample to ANN model
        output[0] = y
        # output[0] = x[0]  # Just for testing, set last val as current val
        output[1:M] = x[0:M-1]  # Last M-1 points of x are set to the rest of the output
        # print('x = {} --> output = {}'.format(x, output))
        return output

    def hx(x):
        return H @ x

    ukf = UnscentedKalmanFilter(dim_x=M, dim_z=D, dt=dt, fx=fx, hx=hx, points=points)
    ukf.x = x_init
    ukf.P = P_init
    ukf.R = R
    ukf.Q = Q

    return ukf


def prepare_dataset(series, M, stride):
    X, y = [], []
    for i in range(0, len(series) - M - 1, stride):
        window = series[i:(i + M)]  #
        X.append(window)
        y.append(series[i + M])
    return np.array(X), np.array(y)


def test_neural_net(ann, history):
    sample_len = 1000

    test_mg_series = utility.mackey_glass(sample_len=sample_len, tau=30)
    test_mg_series = np.array(test_mg_series[0]).reshape((sample_len))
    X_test, y_test = prepare_dataset(test_mg_series, M, stride=1)
    y_pred = ann.predict(X_test)
    y_pred_series = np.zeros(sample_len)
    y_pred_series[M+1 :] = y_pred.reshape(len(y_pred))

    y_self_pred_series = np.zeros(sample_len)
    y_self_pred_series[:M] = X_test[0]
    for i in range(M, sample_len):
        X_window = y_self_pred_series[i-M:i]
        y = ann.predict(X_window.reshape(1, M))   # Reshape needed
        y_self_pred_series[i] = y

    hist = history.history['loss']
    utility.plot(range(len(hist)), hist, label='Training history')

    # utility.plot(range(sample_len), train_mg_series, label='Train series')
    utility.plot(range(sample_len), test_mg_series, label='Test series')
    utility.plot(range(sample_len), y_pred_series, new_figure=False,
                 label='Predicted test series (assisted with true vals of each window)')
    utility.plot(range(sample_len), y_self_pred_series, new_figure=False,
                 label='Predicted test series (rolling prediction: no true vals used)')


def train_neural_net(M):
    sample_len = 8000
    train_mg_series = utility.mackey_glass(sample_len=sample_len, tau=30)
    train_mg_series = np.array(train_mg_series[0]).reshape((sample_len))
    X_train, y_train = prepare_dataset(train_mg_series, M, stride=1)

    ann = Sequential()
    ann.add(Dense(math.ceil(M/2), input_dim=M, activation='relu'))
    ann.add(Dense(1,))    # output (x_k) - no activation because we don't want to limit the range of output
    ann.compile(optimizer='adam', loss='mse')
    history = ann.fit(X_train, y_train, epochs=epochs, verbose=3)

    test_neural_net(ann, history)

    return ann


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    # -------------------------------------------
    # Setting parameters

    # Known paramaters are F, H, Q, R, x_init
    # No. of state variables = M = 4
    # No. of measurement variables = D = 1

    M = 32

    # Simple 1-D example (constant series)
    # F = np.eye(M)  # Process state transition matrix (MxM)
    F = np.zeros((M, M))  # Process state transition matrix (MxM)
    H = np.zeros((1, M))  # Measurement function matrix (DxM)
    H[0][0] = 1.0   # Measure x_k
    Q = 0.05 * np.eye(M)  # Process noise covariance matrix (MxM)
    R = np.array([[0.1]])  # Measurement noise covariance matrix (DxD)
    x_init = 3 * np.ones((M))  # Initial values of state variables (mean) (M,)
    P_init = 0.1 * np.eye(M)  # Initial values of covariance matrix of state variables (MxM)
    # x_true_init = 1.0 * np.ones((M, 1))
    dt = 0.01

    n_samples = 500
    tau = 30


    # -------------------------------------------
    # Generating data
    mg_series = utility.mackey_glass(sample_len=n_samples, tau=tau, n_samples=M)
    mg_series = np.array(mg_series).reshape((M, n_samples))
    x_true_series = mg_series
    # assert False    # how does the state variables become the past M-sized-window?

    # mg_series_2 = data.mackey_glass(tau=30, sample=0.46, length=n_samples)
    # mg_series_2 = np.array(mg_series_2).reshape((1, n_samples))

    ann_model = train_neural_net(M)

    z_true_series = utility.generate_measurement_series(x_true_series, H, R=None)
    # print(z_true_series)

    z_noisy_series = utility.generate_measurement_series(x_true_series, H, R)
    # print(z_noisy_series)

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)
    filter = create_kalman_filter(F, H, Q, R, x_init, P_init)
    my_filter = kf.KalmanFilter(F, H, R, Q, x_init, P_init)
    ukf = create_ukf(ann_model, H, Q, R, dt, x_init, P_init)

    # Pre-allocate output variables
    filter_x = np.zeros((M, n_samples))
    my_filter_x = np.zeros((M, n_samples))
    ukf_x = np.zeros((M, n_samples))
    ukf_x_prior = np.zeros((M, n_samples))


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
        filter_x[:, i] = filter.x
        my_filter_x[:, i] = my_filter.x
        ukf_x[:, i] = ukf.x[:]
        ukf_x_prior[:, i] = ukf.x_prior[:]

    # -------------------------------------------
    # Results analysis

    # ukf_x = mg_series_2

    x_var = range(n_samples)
    utility.plot(x_var, x_true_series[0, :], label='True state x_true_series (x_true_series)')
    utility.plot(x_var, ukf_x[0, :], new_figure=False, label='filterpy UKF predicted state (kf_x)')
    utility.plot(x_var, filter_x[0, :], new_figure=False, label='filterpy KF predicted state (kf_x)', linestyle='--')
    # # utility.plot(x_var, my_filter_x[0, :], new_figure=False, label='My KF predicted state (kf_x)')
    plt.scatter(x_var, z_noisy_series[0, :], label='Noisy measurement (z_noisy_series)', marker='x', c='gray', s=10)

    utility.plot(x_var, x_true_series[0, :], label='True state x_true_series (x_true_series)')
    utility.plot(x_var, ukf_x_prior[0, :], new_figure=False, label='UKF x_prior state (no measurement update) (ukf_x_prior)')

    plt.show()

