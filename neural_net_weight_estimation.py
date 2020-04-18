# Application of Kalman filter using a Python lib

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
import ukf
import utility
from keras.models import Sequential
from keras.layers import Dense, Dropout
import math, os, time
from sklearn.metrics import mean_squared_error
from keras.callbacks import Callback


class EpochInfoTracker(Callback):
    def __init__(self):
        self.weights_history = []

    def on_epoch_end(self, epoch, logs=None):
        weights_vec = get_weights_vector(self.model)
        self.weights_history.append(weights_vec)


class Params:
    pass


params = Params()
params.epochs = 5
params.train_series_length = 500
params.test_series_length = 1000
params.mg_tau = 30
params.window_size = 4    # M
params.ukf_dt = 0.1
params.alpha, params.beta, params.kappa = 1, 2, 1  # Worked well
params.alpha, params.beta, params.kappa = 0.001, 2, 1

# To make training data and related variables accessible across functions
params.train_ukf_ann = True
params.X_data = None
params.y_data = None
params.hxw_model = None
params.curr_idx = 0


def measurement_func(w, x):
    hxw_model = params.hxw_model
    set_weights(hxw_model, w)
    hxw = hxw_model.predict(x.reshape(1, len(x)))   # Reshape needed to feed x as 1 sample to ANN model
    hxw = hxw.flatten() # Flatten to make shape = (1,)
    return hxw


def fw(w, dt=None):
    return w    # Identity


def hw(w):
    x = params.X_data[params.curr_idx]
    hxw = measurement_func(w, x)
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


def predict_series(ann, X_data, series_length):
    pred = ann.predict(X_data)
    y_pred_series = np.zeros(series_length)
    y_pred_series[window + 1:] = pred.reshape(len(pred))

    y_self_pred_series = np.zeros(series_length)
    y_self_pred_series[:window] = X_data[0]
    for i in range(window, series_length):
        X_window = y_self_pred_series[i - window:i]
        y = sgd_ann.predict(X_window.reshape(1, window))  # Reshape needed
        y_self_pred_series[i] = y

    return y_pred_series, y_self_pred_series


def evaluate_neural_nets(sgd_ann, ukf_ann, use_train_series=False, train_series=None):

    if use_train_series:
        X_data, y_data = params.X_data, params.y_data
        series = train_series
        sample_len = params.train_series_length
        title = "Train series (true vs. predicted)"
    else:
        sample_len = params.test_series_length
        series = utility.mackey_glass(sample_len=sample_len, tau=params.mg_tau)
        series = np.array(series[0]).reshape((sample_len))
        X_data, y_data = prepare_dataset(series, window, stride=1)
        title = "Test series (true vs. predicted)"

    sgd_pred, sgd_self_pred = predict_series(sgd_ann, X_data, sample_len)
    ukf_pred, ukf_self_pred = predict_series(ukf_ann, X_data, sample_len)

    utility.plot(range(sample_len), series, title=title, label='True series')
    utility.plot(range(sample_len), sgd_pred, new_figure=False, label='SGD ANN prediction (based on true windows)')
    utility.plot(range(sample_len), ukf_pred, new_figure=False, label='UKF ANN prediction (based on true windows)')

    # utility.plot(range(sample_len), y_self_pred_series, new_figure=False,
    #              label='Predicted test series (rolling prediction: no true vals used)')


def create_neural_net(M):
    ann = Sequential()
    ann.add(Dense(1, input_dim=M, activation='tanh'))
    ann.add(Dense(1, ))  # output (x_k) - no activation because we don't want to limit the range of output
    # ann.add(Dense(1, input_dim=M, activation='tanh'))  # output (x_k) - no activation because we don't want to limit the range of output
    ann.compile(optimizer='sgd', loss='mse')

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


def test_weights_functions():
    ann = create_neural_net(10)
    prev_weights = ann.get_weights()
    vec = get_weights_vector(ann)
    # vec = [elem + 1 for elem in vec]

    ann2 = create_neural_net(10)
    set_weights(ann2, vec)
    post_weights = ann2.get_weights()

    for w_mat1, w_mat2 in zip(prev_weights, post_weights):
        assert np.array_equal(w_mat1, w_mat2)

    print(prev_weights)
    print(post_weights)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    # test_weights_functions()
    # assert False

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
    Q = 0.001 * np.eye(num_weights)  # Process noise covariance matrix (MxM)
    R = np.array([[0.001]])  # Measurement noise covariance matrix (DxD)

    sgd_ann = create_neural_net(window)
    sgd_ann.set_weights(params.hxw_model.get_weights()) # Same starting point as the UKF_ANN

    ukf_ann = create_neural_net(window)
    ukf_ann.set_weights(params.hxw_model.get_weights())  # Same starting point as the UKF_ANN

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
    sgd_train_mse = np.zeros(params.epochs)


    # -------------------------------------------
    # Training loop with UKF
    print("Training neural net with UKF")
    t0 = time.time()
    epoch = 0
    for i in range(num_iter):
        idx = i % len(z_true_series)
        # print(idx)
        if idx == 0:
            # Compute MSE
            if not params.train_ukf_ann:
                break

            preds = ukf_ann.predict(params.X_data)
            mse = mean_squared_error(z_true_series, preds)
            ukf_train_mse[epoch] = mse
            my_ukf_train_mse[epoch] = mse

            epoch += 1
            print('Epoch: {}'.format(epoch))

        params.curr_idx = idx   # For use in hw() to fetch correct x_k sample

        # Time update (state prediction according to F, Q
        ukf_filter.predict()
        # my_ukf.predict()

        # Measurement update (innovation) according to observed z, H, R
        z = z_true_series[idx]
        ukf_filter.update(z)
        # my_ukf.update(z)

        set_weights(params.hxw_model, ukf_filter.x)
        set_weights(ukf_ann, ukf_filter.x)

        # filter.x has shape Mx1
        ukf_w[:, i] = ukf_filter.x[:]
        my_ukf_w[:, i] = my_ukf.x[:]

    time_to_train = time.time() - t0
    print('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    # -------------------------------------------
    # Train SGD ANN (for comparison)
    print("Training neural net with SGD")
    info_tracker = EpochInfoTracker()
    callbacks = [info_tracker]
    history = sgd_ann.fit(params.X_data, params.y_data, batch_size=1, epochs=params.epochs, verbose=2, callbacks=callbacks)


    # -------------------------------------------
    # Results analysis

    # Visualize evolution of ANN weights
    sgd_ann_w = np.array(info_tracker.weights_history).T

    x_var = range(num_iter)
    utility.plot(x_var, ukf_w[0, :], xlabel='Iteration', title='UKF ANN weights', label='W_0', alpha=0.8)
    for j in range(1, ukf_w.shape[0]):
        utility.plot(x_var, ukf_w[j, :], new_figure=False, label='W_' + str(j), alpha=0.8)

    x_var = range(params.epochs)
    utility.plot(x_var, sgd_ann_w[0, :], xlabel='Iteration', title='SGD ANN weights', label='W_0', alpha=0.8)
    for j in range(1, sgd_ann_w.shape[0]):
        utility.plot(x_var, sgd_ann_w[j, :], new_figure=False, label='W_' + str(j), alpha=0.8)

    # Visualize evolution of true y vs. hxw(x,w)

    # Visualize error curve (SGD vs UKF)
    x_var = range(params.epochs)
    hist = history.history['loss']
    utility.plot(x_var, hist, xlabel='Epoch', label='SGD ANN training history (MSE)')
    utility.plot(x_var, ukf_train_mse, new_figure=False, label='UKF ANN training history (MSE)')

    # True test series vs. ANN pred vs, UKF pred
    evaluate_neural_nets(sgd_ann, ukf_ann, use_train_series=True, train_series=X_series)
    evaluate_neural_nets(sgd_ann, ukf_ann)

    utility.save_all_figures('output')
    plt.show()
