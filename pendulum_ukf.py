import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
import time
from sklearn.metrics import mean_squared_error


def pend_ukf(observed_pend, true_state_pend, pendulum_equations, pendulum_solved, t_pend, theta_pend,
             init_point_pendulum):
    dt = 0.1
    b, c = theta_pend
    start_time = time.time()

    def pendulum_eq(y, t, b, c):
        theta, omega = y
        dydt = [omega, -b * omega - c * np.sin(theta)]
        return dydt

    y0 = init_point_pendulum
    L = 3267

    theta_values = true_state_pend[:, 0]
    omega_values = true_state_pend[:, 1]

    def f_cv(x, dt):
        sol = odeint(pendulum_eq, x, [0, dt], args=(b, c))
        return sol[-1]

    def h_cv(x):
        theta_ = x[0]
        pos_x = L * np.sin(theta_)
        pos_y = L * np.cos(theta_)
        return [pos_x, pos_y]

    measurement_noise_std = 1

    points = MerweScaledSigmaPoints(2, alpha=1e-3, beta=2., kappa=1.0)
    ukf = UKF(dim_x=2, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=points)
    ukf.P = np.diag([1, 1])
    ukf.x = np.array(init_point_pendulum)
    ukf.R = np.diag([measurement_noise_std ** 2, measurement_noise_std ** 2])
    ukf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=10, block_size=1)

    kf_theta_values = []
    kf_omega_values = []

    for i in range(len(t_pend)):
        theta = theta_values[i]
        omega = omega_values[i]
        center = np.array((L * np.sin(theta), L * np.cos(theta)))

        ukf.predict()
        ukf.update(center)

        kf_theta_values.append(ukf.x[0])
        kf_omega_values.append(ukf.x[1])

    predictions = list(zip(kf_theta_values, kf_omega_values))

    sampling_time = time.time() - start_time

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using UKF: {rms}')
    print(f'Sampling time of pendulum using UKF is: {sampling_time}')

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_pend, true_state_pend[:, 0], 'b', label='Modelowe wartości')
    plt.plot(t_pend, kf_theta_values, label='Estymata - algorytm UKF')
    plt.plot(t_pend, observed_pend[:, 0], 'x', label='Pomiary')
    plt.xlabel('Czas')
    plt.ylabel('Kąt odchylenia')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_pend, true_state_pend[:, 1], 'g', label='Modelowe wartości')
    plt.plot(t_pend, kf_omega_values, label='Estymata - algorytm UKF')
    plt.plot(t_pend, observed_pend[:, 1], 'x', label='Pomiary')
    plt.xlabel('Czas')
    plt.ylabel('Prędkość kątowa')
    plt.legend()

    plt.tight_layout()
    plt.savefig('obj1_ukf_predictions.pdf')
    # plt.show()

    return predictions, sampling_time, None
