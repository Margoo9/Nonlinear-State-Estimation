from params import init_point_pendulum, number_of_points_pendulum, t_pend, b, c,\
                   init_point_lv, number_of_points_lv, t_lv, alpha, beta, gamma, delta, \
                   noise_count

from objects import add_gaussian_noise, pendulum_equations, pendulum_solved, \
                    lv_equations, lv_solved

import matplotlib.pyplot as plt
import numpy as np


# Pendulum
theta_pend = [b, c]
true_state_pend = pendulum_solved(None, theta_pend)

accumulated_noise_pend = np.zeros_like(true_state_pend)
for _ in range(noise_count):
    observed_pend = add_gaussian_noise(true_state_pend, number_of_points_pendulum)
    accumulated_noise_pend += (observed_pend - true_state_pend)

mean_noise_pend = accumulated_noise_pend / noise_count
mean_observed_pend = true_state_pend + mean_noise_pend

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
ax1.plot(mean_observed_pend[:, 0], 'x', label='Pomiary')
ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
ax1.legend(loc='upper right')
ax1.set_ylabel('Kąt odchylenia')

ax2.plot(mean_observed_pend[:, 1], 'x', label='Pomiary')
ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')

plt.xlabel('Czas')
ax2.legend(loc='upper right')
ax2.set_ylabel('Prędkość kątowa')
# plt.savefig('obj1_data.pdf')
plt.show()


# Lotka-Volterry
theta_lv = [alpha, beta, gamma, delta]
true_state_lv = lv_solved(None, theta_lv)

accumulated_noise_lv = np.zeros_like(true_state_lv)
for _ in range(noise_count):
    observed_lv = add_gaussian_noise(true_state_lv, number_of_points_lv)
    accumulated_noise_lv += (observed_lv - true_state_lv)

mean_noise_lv = accumulated_noise_lv / noise_count
mean_observed_lv = true_state_lv + mean_noise_lv

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
ax1.plot(mean_observed_lv[:, 0], 'x', label='Pomiary')
ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
ax1.legend(loc='upper right')
ax1.set_ylabel('Liczba ofiar')

ax2.plot(mean_observed_lv[:, 1], 'x', label='Pomiary')
ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')

plt.xlabel('Czas')
ax2.legend(loc='upper right')
ax2.set_ylabel('Liczba drapieżników')
# plt.savefig('obj2_data.pdf')
plt.show()
