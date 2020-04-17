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
params.epochs = 3
params.train_series_length = 500
params.test_series_length = 1000
params.mg_tau = 30
params.window_size = 4    # M
params.ukf_dt = 0.1
params.alpha, params.beta, params.kappa = 1, 2, 1  # Worked well
params.alpha, params.beta, params.kappa = 0.001, 2, 1

# To make training data and related variables accessible across functions
params.X_data = None
params.y_data = None
params.hxw_model = None
params.iter = 0


def measurement_func(w, x):
    hxw_model = params.hxw_model
    set_weights(hxw_model, w)
    hxw = hxw_model.predict(x.reshape(1, len(x)))   # Reshape needed to feed x as 1 sample to ANN model
    hxw = hxw.flatten() # Flatten to make shape = (1,)
    return hxw


def fw(w, dt=None):
    return w    # Identity


def hw(w):
    k = params.iter
    x = params.X_data[k]

    hxw = measurement_func(w, x)

    k = k + 1
    if k == params.X_data.shape[0]:
        k = 0
    params.iter = k

    return hxw


def create_ukf(Q, R, dt, w_init, P_init):
    M = w_init.shape[0]

    points = MerweScaledSigmaPoints(M, params.alpha, params.beta, params.kappa)

    ukf = UnscentedKalmanFilter(dim_x=M, dim_z=1, dt=dt, fx=fw, hx=hw, points=points)
    ukf.x = w_init
    ukf.P = P_init
    ukf.R = R
    ukf.Q = Q

    return ukf


def create_my_ukf(Q, R, dt, w_init, P_init):
    my_ukf = ukf.UnscentedKalmanFilter(fw, hw, R, Q, w_init, P_init, params.alpha, params.beta, params.kappa)
    return my_ukf


def prepare_dataset(series, M, stride):
    X, y = [], []
    for i in range(0, len(series) - M - 1, stride):
        window = series[i:(i + M)]  #
        X.append(window)
        y.append(series[i + M])
    return np.array(X), np.array(y)


def test_neural_net(ann, history):
    sample_len = params.test_series_length

    test_mg_series = utility.mackey_glass(sample_len=sample_len, tau=params.mg_tau)
    test_mg_series = np.array(test_mg_series[0]).reshape((sample_len))
    X_test, y_test = prepare_dataset(test_mg_series, window, stride=1)
    y_pred = ann.predict(X_test)
    y_pred_series = np.zeros(sample_len)
    y_pred_series[window + 1:] = y_pred.reshape(len(y_pred))

    y_self_pred_series = np.zeros(sample_len)
    y_self_pred_series[:window] = X_test[0]
    for i in range(window, sample_len):
        X_window = y_self_pred_series[i - window:i]
        y = ann.predict(X_window.reshape(1, window))   # Reshape needed
        y_self_pred_series[i] = y

    hist = history.history['loss']
    utility.plot(range(len(hist)), hist, label='Training history')

    # utility.plot(range(sample_len), train_mg_series, label='Train series')
    utility.plot(range(sample_len), test_mg_series, label='Test series')
    utility.plot(range(sample_len), y_pred_series, new_figure=False,
                 label='Predicted test series (assisted with true vals of each window)')
    utility.plot(range(sample_len), y_self_pred_series, new_figure=False,
                 label='Predicted test series (rolling prediction: no true vals used)')


def create_neural_net(M):
    ann = Sequential()
    ann.add(Dense(math.ceil(M / 2), input_dim=M, activation='relu'))
    ann.add(Dense(1, ))  # output (x_k) - no activation because we don't want to limit the range of output
    ann.compile(optimizer='adam', loss='mse')

    return ann


def train_neural_net(M):
    sample_len = params.train_series_length
    train_mg_series = utility.mackey_glass(sample_len=sample_len, tau=params.mg_tau)
    train_mg_series = np.array(train_mg_series[0]).reshape((sample_len))
    X_train, y_train = prepare_dataset(train_mg_series, M, stride=1)

    ann = Sequential()
    ann.add(Dense(math.ceil(M/2), input_dim=M, activation='relu'))
    ann.add(Dense(1,))    # output (x_k) - no activation because we don't want to limit the range of output
    ann.compile(optimizer='adam', loss='mse')
    history = ann.fit(X_train, y_train, epochs=params.epochs, verbose=3)

    test_neural_net(ann, history)

    return ann


def get_weights_vector(model):
    weights = model.get_weights()
    # print(weights)
    weights_vec = []
    for w_mat in weights:
        weights_vec.extend(w_mat.reshape(w_mat.size))

    weights_vec = np.array(weights_vec)
    return weights_vec


def set_weights(model, weights_vec):
    prev_weights = model.get_weights()
    # print(prev_weights)
    new_weights = []
    start = 0

    for prev_w_mat in prev_weights:
        end = start + prev_w_mat.size
        new_w_mat = np.array(weights_vec[start: end]).reshape(prev_w_mat.shape)
        new_weights.append(new_w_mat)
        start = end

    model.set_weights(new_weights)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    # -------------------------------------------
    # Setting parameters

    # Known paramaters are hx function (neural net), Q, R, w_init
    # No. of state variables = no. of weights in neural net
    # No. of measurement variables = D = 1 (y)

    window = params.window_size


    dt = 0.01
    n_samples = params.train_series_length


    # -------------------------------------------
    # Generating data
    X_series = utility.mackey_glass(sample_len=n_samples, tau=params.mg_tau, n_samples=window)
    X_series = np.array(X_series[0]).reshape((n_samples))

    params.X_data, params.y_data = prepare_dataset(X_series, window, stride=1)

    # Create ANN, get its initial weights
    params.hxw_model = create_neural_net(window)
    w_init = get_weights_vector(params.hxw_model)
    num_weights = w_init.shape[0]

    P_init = 0.1 * np.eye(num_weights)  # Initial values of covariance matrix of state variables (MxM)
    Q = 0.05 * np.eye(num_weights)  # Process noise covariance matrix (MxM)
    R = np.array([[0.1]])  # Measurement noise covariance matrix (DxD)

    sgd_ann = create_neural_net(window)
    sgd_ann.set_weights(params.hxw_model.get_weights()) # Same starting point as the UKF_ANN

    z_true_series = params.y_data
    num_iter = params.epochs * len(z_true_series)

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)
    ukf_filter = create_ukf(Q, R, dt, w_init, P_init)
    my_ukf = create_my_ukf(Q, R, dt, w_init, P_init)

    # Pre-allocate output variables
    ukf_w = np.zeros((num_weights, num_iter))
    my_ukf_w = np.zeros((num_weights, num_iter))
    ukf_train_mse = np.zeros(params.epochs)
    my_ukf_train_mse = np.zeros(params.epochs)


    # -------------------------------------------
    # Training loop with UKF
    print("Training neural net with UKF")
    epoch = 0
    for i in range(num_iter):
        idx = i % len(z_true_series)
        # print(idx)
        if idx == 0:
            # Compute MSE
            preds = params.hxw_model.predict(params.X_data)
            mse = mean_squared_error(z_true_series, preds)
            ukf_train_mse[epoch] = mse
            my_ukf_train_mse[epoch] = mse

            epoch += 1
            print('Epoch: {}'.format(epoch))

        # Time update (state prediction according to F, Q
        ukf_filter.predict()
        # my_ukf.predict()

        # Measurement update (innovation) according to observed z, H, R
        z = z_true_series[idx]
        ukf_filter.update(z)
        # my_ukf.update(z)

        # filter.x has shape Mx1
        ukf_w[:, i] = ukf_filter.x[:]
        my_ukf_w[:, i] = my_ukf.x[:]



    # Train SGD ANN (for comparison)
    print("Training neural net with SGD")
    history = sgd_ann.fit(params.X_data, params.y_data, batch_size=1, epochs=params.epochs)
    hist = history.history['loss']
    utility.plot(range(len(hist)), hist, label='SGD ANN Training history')

    # -------------------------------------------
    # Results analysis

    # Visualize evolution of 3 ANN weights
    x_var = range(num_iter)
    utility.plot(x_var, ukf_w[0, :], xlabel='Iteration', label='Weight 0')
    utility.plot(x_var, ukf_w[1, :], new_figure=False, label='Weight 1')
    utility.plot(x_var, ukf_w[2, :], new_figure=False, label='Weight 2')

    # Visualize evolution of true y vs. hxw(x,w)

    # Visualize error curve
    x_var = range(params.epochs)
    utility.plot(x_var, ukf_train_mse, xlabel='Iteration', label='MSE')

    # utility.plot(x_var, ukf_x[0, :], new_figure=False, label='filterpy UKF predicted state (kf_x)')
    # # utility.plot(x_var, kf_x[0, :], new_figure=False, label='filterpy KF predicted state (kf_x)', linestyle='--')
    # utility.plot(x_var, my_ukf_x[0, :], new_figure=False, label='My UKF predicted state (kf_x)', linestyle='--', c='purple')
    # plt.scatter(x_var, z_true_series[0, :], label='Noisy measurement (z_noisy_series)', marker='x', c='gray', s=10, alpha=0.5)


    plt.show()

