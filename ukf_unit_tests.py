from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
import ukf
import numpy as np


# Required params
M = 4
F = np.zeros((M, M))  # Process state transition matrix (MxM)
H = np.zeros((1, M))  # Measurement function matrix (DxM)
H[0][0] = 1.0   # Measure x_k
Q = 0.05 * np.eye(M)  # Process noise covariance matrix (MxM)
R = np.array([[0.1]])  # Measurement noise covariance matrix (DxD)
alpha, beta, kappa = 0.001, 2, 1
dt = 0.1

# Some non-linear function
def fx(x, dt=None):
    return x ** 2 + 1


def hx(x):
    return H @ x


def test_weight_computation():
    print('test_weight_computation: running')
    x = np.random.rand(M)
    P = np.diag(np.random.rand(M))  # Diagonal matrix with random no.s on the diagonal

    my_ukf = ukf.UnscentedKalmanFilter(fx, hx, R, Q, x, P, alpha, beta, kappa)
    points_obj = MerweScaledSigmaPoints(M, alpha, beta, kappa)

    my_Wm, my_Wc = my_ukf.Wm, my_ukf.Wc
    merwe_Wm, merwe_Wc = points_obj.Wm, points_obj.Wc

    assert np.array_equal(my_Wm, merwe_Wm)
    assert np.array_equal(my_Wc, merwe_Wc)
    print('test_weight_computation: passed')


def test_sigma_point_generation():
    print('test_sigma_point_generation: running')
    x = np.random.rand(M)
    P = np.diag(np.random.rand(M))  # Diagonal matrix with random no.s on the diagonal

    my_ukf = ukf.UnscentedKalmanFilter(fx, hx, R, Q, x, P, alpha, beta, kappa)
    points_obj = MerweScaledSigmaPoints(M, alpha, beta, kappa)

    merwe_sigmas = points_obj.sigma_points(x, P)
    my_sigmas = my_ukf.get_sigma_points(my_ukf.Wm, my_ukf.Wc, x, P)
    assert np.array_equal(my_sigmas, merwe_sigmas)
    print('test_sigma_point_generation: passed')


def test_unscented_transform():
    print('test_unscented_transform: running')
    x = np.random.rand(M)
    P = np.diag(np.random.rand(M))  # Diagonal matrix with random no.s on the diagonal

    my_ukf = ukf.UnscentedKalmanFilter(fx, hx, R, Q, x, P, alpha, beta, kappa)
    my_sigmas = my_ukf.get_sigma_points(my_ukf.Wm, my_ukf.Wc, x, P)
    my_mean, my_cov, my_transformed_sigmas = my_ukf.unscented_transform(fx, my_sigmas, my_ukf.Wm, my_ukf.Wc, P)

    fx_sigmas = [fx(point) for point in my_sigmas]
    ref_transformed_sigmas = np.atleast_2d(fx_sigmas)
    ref_mean, ref_cov = unscented_transform(ref_transformed_sigmas, my_ukf.Wm, my_ukf.Wc, P)

    assert np.array_equal(my_transformed_sigmas, ref_transformed_sigmas)
    assert np.array_equal(my_mean, ref_mean)
    assert np.allclose(my_cov, ref_cov)

    print('test_unscented_transform: passed')


def test_predict_update():
    print('test_predict_update: running')
    D = R.shape[0]
    x = np.random.rand(M)
    P = np.diag(np.random.rand(M))  # Diagonal matrix with random no.s on the diagonal

    my_ukf = ukf.UnscentedKalmanFilter(fx, hx, R, Q, x, P, alpha, beta, kappa)

    points_obj = MerweScaledSigmaPoints(M, alpha, beta, kappa)
    ref_ukf = UnscentedKalmanFilter(dim_x=M, dim_z=D, dt=dt, fx=fx, hx=hx, points=points_obj)
    ref_ukf.x, ref_ukf.P = x, P
    ref_ukf.R, ref_ukf.Q = R, Q

    # Time update
    ref_ukf.predict()
    my_ukf.predict()
    assert np.allclose(my_ukf.x, ref_ukf.x)

    # Measurement update
    z = np.random.randn(D)
    my_ukf.update(z)
    ref_ukf.update(z)
    assert np.allclose(my_ukf.x, ref_ukf.x)
    print('test_predict_update: passed')


if __name__ == "__main__":
    test_weight_computation()
    test_sigma_point_generation()
    test_unscented_transform()
    test_predict_update()
