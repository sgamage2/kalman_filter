# Application of Kalman filter using a Python lib

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
import ukf
import utility
from keras.models import Sequential
from keras.layers import Dense, Dropout
import math, os
from sklearn.metrics import mean_squared_error


class Params:
    pass


params = Params()
params.epochs = 30
params.train_series_length = 20000
params.test_series_length = 1000
params.mg_tau = 30
params.filter_series_length = 500
params.state_vector_size = 64    # M
params.ukf_dt = 0.1
params.alpha, params.beta, params.kappa = 1, 2, 1  # Worked well
# params.alpha, params.beta, params.kappa = 0.001, 2, 1
params.measurement_noise = 'gaussian'   # gaussian, uniform
params.Q_var = 0.05
params.R_var = 0.05

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


def process_func(x, dt, fx_model):
    M = len(x)
    output = np.zeros(M)
    y = fx_model.predict(x.reshape(1, M))   # Reshape needed to feed x as 1 sample to ANN model
    output[M-1] = y
    output[0:M-1] = x[1:M]  # First M-1 points of x are set to the rest of the output
    # print('x_in = {} --> xout = {}'.format(x, output))
    return output


def measurement_func(x, H):
    # return np.array(x[M-1]).reshape((1, 1))
    return H @ x


def create_ukf(fx_model, H, Q, R, dt, x_init, P_init):
    M = x_init.shape[0]
    D = H.shape[0]

    points = MerweScaledSigmaPoints(M, params.alpha, params.beta, params.kappa)

    def fx(x, dt):
        return process_func(x, dt, fx_model)

    def hx(x):
        return measurement_func(x, H)

    ukf = UnscentedKalmanFilter(dim_x=M, dim_z=D, dt=dt, fx=fx, hx=hx, points=points)
    ukf.x = x_init
    ukf.P = P_init
    ukf.R = R
    ukf.Q = Q

    return ukf


def create_my_ukf(fx_model, H, Q, R, dt, x_init, P_init):
    def fx(x, dt=None):
        return process_func(x, dt, fx_model)

    def hx(x):
        return measurement_func(x, H)

    my_ukf = ukf.UnscentedKalmanFilter(fx, hx, R, Q, x_init, P_init, params.alpha, params.beta, params.kappa)
    return my_ukf


def prepare_dataset(series, M, stride, y_series=None):
    if y_series is None:
        y_series = series

    X, y = [], []
    for i in range(0, len(series) - M - 1, stride):
        window = series[i:(i + M)]  #
        X.append(window)
        y.append(y_series[i + M])
    return np.array(X), np.array(y)


def test_neural_net(ann, history):
    sample_len = params.test_series_length
    M = params.state_vector_size

    test_mg_series = utility.mackey_glass(sample_len=sample_len, tau=params.mg_tau)
    test_mg_series = np.array(test_mg_series[0]).reshape((sample_len))

    orig_series = test_mg_series

    noise = utility.get_noise_series(params.measurement_noise, 0, params.R_var, len(test_mg_series))
    test_mg_series = test_mg_series + noise

    X_test, y_test = prepare_dataset(test_mg_series, M, stride=1)
    y_pred_series, y_self_pred_series = utility.predict_series(ann, X_test, sample_len, M)

    hist = history.history['loss']
    utility.plot(range(len(hist)), hist, label='Training history')

    # utility.plot(range(sample_len), train_mg_series, label='Train series')
    utility.plot(range(sample_len), test_mg_series, label='Test series')
    utility.plot(range(sample_len), y_pred_series, new_figure=False, label='Predicted test series (assisted with true vals of each window)')
    utility.plot(range(sample_len), orig_series, new_figure=False, label='orig_series')
    # utility.plot(range(sample_len), y_self_pred_series, new_figure=False,
    #              label='Predicted test series (rolling prediction: no true vals used)')


def train_neural_net(M):
    sample_len = params.train_series_length
    train_mg_series = utility.mackey_glass(sample_len=sample_len, tau=params.mg_tau)
    train_mg_series = np.array(train_mg_series[0]).reshape((sample_len))

    noise = utility.get_noise_series(params.measurement_noise, 0, params.R_var * 1.5, len(train_mg_series))
    noisy_train_series = train_mg_series + noise

    utility.plot(range(sample_len), noisy_train_series, label='noisy_train_series')
    utility.plot(range(sample_len), train_mg_series, new_figure=False, label='train_mg_series')

    # X_train, y_train = prepare_dataset(train_mg_series, M, stride=1, y_series=None)
    X_train, y_train = prepare_dataset(noisy_train_series, M, stride=1, y_series=train_mg_series)

    print("Training neural network")
    ann = Sequential()
    # ann.add(Dense(math.ceil(M/2), input_dim=M, activation='tanh'))
    ann.add(Dense(32, input_dim=M, activation='relu'))
    ann.add(Dense(16, input_dim=M, activation='relu'))
    ann.add(Dense(1, activation='tanh'))    # output (x_k) - no activation because we don't want to limit the range of output
    ann.compile(optimizer='adam', loss='mse')
    history = ann.fit(X_train, y_train, epochs=params.epochs, verbose=3)

    test_neural_net(ann, history)

    return ann


def main():

    # -------------------------------------------
    # Setting parameters

    # Known paramaters are F, H, Q, R, x_init
    # No. of state variables = M = 4
    # No. of measurement variables = D = 1

    M = params.state_vector_size

    # Simple 1-D example (constant series)
    # F = np.eye(M)  # Process state transition matrix (MxM)
    F = np.zeros((M, M))  # Process state transition matrix (MxM)
    H = np.zeros((1, M))  # Measurement function matrix (DxM)
    H[0][M-1] = 1.0   # Measure x_k
    # Q = params.Q_var * np.eye(M)  # Process noise covariance matrix (MxM)
    Q = np.zeros((M,M))  # Process noise covariance matrix (MxM)
    Q[M-1, M-1] = params.Q_var
    R = np.array([[params.R_var]])  # Measurement noise covariance matrix (DxD)
    x_init = np.zeros((M))  # Initial values of state variables (mean) (M,)
    P_init = 0.1 * np.eye(M)  # Initial values of covariance matrix of state variables (MxM)
    # x_true_init = 1.0 * np.ones((M, 1))
    dt = 0.01
    n_samples = params.filter_series_length


    # -------------------------------------------
    # Generating data

    # True series: only the last (M-1)th state contains the true series
    # True series is only used to generate a noisy observation series and for plotting later
    mg_series = utility.mackey_glass(sample_len=n_samples, tau=params.mg_tau)
    x_true_series = np.zeros(shape=(M, n_samples))
    x_true_series[M-1] = mg_series[0].reshape(n_samples)

    # assert False    # how does the state variables become the past M-sized-window?

    # mg_series_2 = data.mackey_glass(tau=params.mg_tau, sample=0.46, length=n_samples)
    # mg_series_2 = np.array(mg_series_2).reshape((1, n_samples))

    ann_model = train_neural_net(M)


    # z_true_series = utility.generate_measurement_series(x_true_series, H, R=None)
    # print(z_true_series)

    z_noisy_series = utility.generate_measurement_series(x_true_series, H, R, noise_type=params.measurement_noise)
    # print(z_noisy_series)

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)
    kalman_filter = create_kalman_filter(F, H, Q, R, x_init, P_init)
    ukf_filter = create_ukf(ann_model, H, Q, R, dt, x_init, P_init)
    my_ukf = create_my_ukf(ann_model, H, Q, R, dt, x_init, P_init)

    # Pre-allocate output variables
    kf_x = np.zeros((M, n_samples))
    ukf_x = np.zeros((M, n_samples))
    my_ukf_x = np.zeros((M, n_samples))
    ukf_x_prior = np.zeros((M, n_samples))
    my_ukf_x_after_pred = np.zeros((M, n_samples))
    ann_output_of_state = np.zeros((M, n_samples))

    # Predict on true state series and noisy observation series with ANN
    X_test, y_test = prepare_dataset(x_true_series[M-1], M, stride=1)
    y_pred_series, y_self_pred_series = utility.predict_series(ann_model, X_test, n_samples, M)

    X_test, y_test = prepare_dataset(z_noisy_series[0], M, stride=1)
    y_noisy_pred_series, y_self_noisy_pred_series = utility.predict_series(ann_model, X_test, n_samples, M)

    # -------------------------------------------
    # Kalman filtering
    print("Running UKF state estimation loop")
    for i in range(n_samples):
        ann_output_of_state[:, i] = process_func(my_ukf.x, None, ann_model)

        # Time update (state prediction according to F, Q
        kalman_filter.predict()
        # ukf_filter.predict()
        my_ukf.predict()
        my_ukf_x_after_pred[:, i] = my_ukf.x


        # Measurement update (innovation) according to observed z, H, R
        z = z_noisy_series[:, i]
        kalman_filter.update(z)
        # ukf_filter.update(z)
        my_ukf.update(z)

        # filter.x has shape Mx1
        kf_x[:, i] = kalman_filter.x
        ukf_x[:, i] = ukf_filter.x[:]
        my_ukf_x[:, i] = my_ukf.x[:]
        ukf_x_prior[:, i] = ukf_filter.x_prior[:]

    # -------------------------------------------
    # Results analysis

    mse_series = (x_true_series[M-1, :] - my_ukf_x[M-1, :])**2
    normalized_mse_series = (mse_series - np.mean(mse_series)) / np.std(mse_series)

    ukf_mse = mean_squared_error(x_true_series[M - 1, :], my_ukf_x[M - 1, :])
    kf_mse = mean_squared_error(x_true_series[M - 1, :], kf_x[M - 1, :])
    print('UKF MSE = {}, KF MSE = {}'.format(ukf_mse, kf_mse))

    # ukf_x = mg_series_2

    x_var = range(n_samples)
    utility.plot(x_var, x_true_series[M-1, :], label='True state', c='red')
    utility.plot(x_var, my_ukf_x[M-1, :], new_figure=False, label='UKF predicted state', c='blue', alpha=0.7)
    # utility.plot(x_var, ukf_x[0, :], new_figure=False, label='filterpy UKF predicted state', linestyle='--', c='orange', alpha=0.7)
    plt.scatter(x_var, z_noisy_series[0, :], label='Noisy measurement', marker='x', c='gray', s=10, alpha=0.7)
    utility.plot(x_var, kf_x[M-1, :], new_figure=False, label='KF predicted state', c='lightseagreen', alpha=0.4)

    utility.plot(x_var, normalized_mse_series, label='MSE (UKF vs. true)', c='blue', alpha=0.7)

    utility.plot(x_var, x_true_series[M-1, :], label='True state x_true_series (x_true_series)', c='red')
    # utility.plot(x_var, my_ukf_x_after_pred[0, :], new_figure=False, label='My UKF x_after_pred (no measurement update) (ukf_x_prior)')
    utility.plot(x_var, ann_output_of_state[M-1, :], new_figure=False, label='KF state before measurement update (ANN prediction on prev state)')
    utility.plot(x_var, y_pred_series, new_figure=False, label='ANN prediction on true series (y_pred_series)')
    utility.plot(x_var, y_noisy_pred_series, new_figure=False, label='ANN prediction on noisy observations (y_pred_series)', linestyle='--')

    utility.save_all_figures('output')
    plt.show()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    main()
