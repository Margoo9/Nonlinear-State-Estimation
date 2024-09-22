import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
import scipy
from sklearn.metrics import mean_squared_error
import time


# np.random.seed(42)


def lv_ukf(observed_lv, true_state_lv, lv_equations, lv_solved, t_lv, theta_lv, init_point_lv):
    alpha, beta, gamma, delta = theta_lv
    def cholesky_func(x):
        try:
            result = scipy.linalg.cholesky(x)
        except scipy.linalg.LinAlgError:
            x = (x + x.T)/2
            result = scipy.linalg.cholesky(x)
        return result

    theta_values = true_state_lv[:, 0]
    omega_values = true_state_lv[:, 1]

    start_time = time.time()

    def f_cv(x, dt):
        sol = odeint(lv_equations, x, [0, dt], args=(theta_lv,))
        # sol = odeint(lv_equations, y0=init_point_lv, t=t_lv, rtol=0.01, args=(theta_lv,))
        return sol[-1]

    def h_cv(x):
        return x

    measurement_noise_std = 1

    # points = MerweScaledSigmaPoints(2, alpha=1e-3, beta=2., kappa=1.0, sqrt_method=cholesky_func)
    points = MerweScaledSigmaPoints(2, alpha=0.1, beta=2., kappa=0.0, sqrt_method=cholesky_func)
    ukf = UKF(dim_x=2, dim_z=2, fx=f_cv, hx=h_cv, dt=0.1, points=points)
    ukf.P = np.diag([10, 10])  # Initialize with larger values
    ukf.x = np.array(init_point_lv)
    ukf.R = np.diag([measurement_noise_std ** 2, measurement_noise_std ** 2])
    # ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=10, block_size=1)
    ukf.Q = np.eye(2)

    kf_theta_values = []
    kf_omega_values = []

    for i in range(len(t_lv)):
        measurement = observed_lv[i]

        ukf.predict()
        ukf.update(measurement)

        ukf.P += np.eye(2) * 1e-6

        kf_theta_values.append(ukf.x[0])
        kf_omega_values.append(ukf.x[1])

    predictions = list(zip(kf_theta_values, kf_omega_values))

    sampling_time = time.time() - start_time

    rms = mean_squared_error(true_state_lv, predictions, squared=False)
    print(f'RMSE of lv predictions using UKF: {rms}')
    print(f'Sampling time of lv using UKF is: {sampling_time}')

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(kf_theta_values, label='Estymata - algorytm UKF')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(kf_omega_values, label='Estymata - algorytm UKF')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_ukf_predictions.pdf')

    # plt.show()
    return predictions, sampling_time, None
