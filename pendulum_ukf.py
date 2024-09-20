import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from params import init_point_pendulum, number_of_points_pendulum, t_pend, b, c,\
                   init_point_lv, number_of_points_lv, t_lv, alpha, beta, gamma, delta

from objects import add_gaussian_noise, pendulum_equations, pendulum_solved, \
                    lv_equations, lv_solved, observed_pend, true_state_pend, theta_pend


def pend_ukf(observed_pend, true_state_pend, pendulum_equations, pendulum_solved, t_pend, theta_pend, init_point_pendulum):
    dt = 0.1

    def run_simulation():
        def pendulum_eq(y, t, b, c):
            theta, omega = y
            dydt = [omega, -b * omega - c * np.sin(theta)]
            return dydt


        y0 = init_point_pendulum

        theta_values = true_state_pend[:, 0]
        omega_values = true_state_pend[:, 1]

        def f_cv(x, dt):
            sol = odeint(pendulum_eq, x, [0, dt], args=(b, c))
            return sol[-1]

        def h_cv(x):
            L = 3267
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

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t_pend, sol[:, 0], 'b', label='Modelowe wartości')
        plt.plot(t_pend, kf_theta_values, label='Estymata - algorytm UKF')
        plt.plot(t_pend, data[:, 0], 'x', label='Pomiary')
        plt.xlabel('Czas')
        plt.ylabel('Kąt odchylenia')
        plt.legend()
    
        plt.subplot(2, 1, 2)
        plt.plot(t_pend, sol[:, 1], 'g', label='Modelowe wartości')
        plt.plot(t_pend, kf_omega_values, label='Estymata - algorytm UKF')
        plt.plot(t_pend, data[:, 1], 'x', label='Pomiary')
        plt.xlabel('Czas')
        plt.ylabel('Prędkość kątowa')
        plt.legend()
    
        plt.tight_layout()
        plt.savefig('obj1_ukf_predictions.pdf')
        # plt.show()
