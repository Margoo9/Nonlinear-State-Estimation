import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from params import init_point_pendulum, number_of_points_pendulum, t_pend, b, c, \
    init_point_lv, number_of_points_lv, t_lv, alpha, beta, gamma, delta


def add_gaussian_noise(data, number_of_points):
    gaussian_noise = np.random.normal(size=(number_of_points, 2))
    return data + gaussian_noise


# Pendulum
def pendulum_equations(X, t, theta_pend):
    b_pend = theta_pend[0]
    c_pend = theta_pend[1]
    return [X[1], -b_pend * X[1] - c_pend * np.sin(X[0])]


def pendulum_solved(rng, theta_pend, size=None):
    return odeint(pendulum_equations, y0=init_point_pendulum, t=t_pend, rtol=0.01, args=(theta_pend,))


theta_pend = [b, c]
true_state_pend = pendulum_solved(None, theta_pend)
observed_pend = add_gaussian_noise(true_state_pend, number_of_points_pendulum)

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
ax1.legend(loc='upper right')
ax1.set_ylabel('Kąt odchylenia')

ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')

plt.xlabel('Czas')
ax2.legend(loc='upper right')
ax2.set_ylabel('Prędkość kątowa')
plt.savefig('obj1_data.pdf')
# plt.show()


# Lotka - Volterra
def lv_equations(X, t, theta_lv):
    alpha_lv = theta_lv[0]
    beta_lv = theta_lv[1]
    gamma_lv = theta_lv[2]
    delta_lv = theta_lv[3]
    return np.array([alpha_lv * X[0] - beta_lv * X[0] * X[1], -gamma_lv * X[1] + delta_lv * beta_lv * X[0] * X[1]])


def lv_solved(rng, theta_lv, size=None):
    return odeint(lv_equations, y0=init_point_lv, t=t_lv, rtol=0.01, args=(theta_lv,))


theta_lv = [alpha, beta, gamma, delta]
true_state_lv = lv_solved(None, theta_lv)
observed_lv = add_gaussian_noise(true_state_lv, number_of_points_lv)

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
ax1.legend(loc='upper right')
ax1.set_ylabel('Liczba ofiar')

ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')

plt.xlabel('Czas')
ax2.legend(loc='upper right')
ax2.set_ylabel('Liczba drapieżników')
plt.savefig('obj2_data.pdf')
# plt.show()
