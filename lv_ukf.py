import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
import scipy
from sklearn.metrics import mean_squared_error
import time


np.random.seed(42)


def sqrt_func(x):
    try:
        result = scipy.linalg.cholesky(x)
    except scipy.linalg.LinAlgError:
        x = (x + x.T)/2
        result = scipy.linalg.cholesky(x)
    return result


# Pendulum parameters and variables
theta0 = 15
omega0 = 7
a = 0.9
b = 0.5
c = 0.75
d = 0.25

# Time parameters
size = 100
time_lv = 32
dt = 0.1
t = np.linspace(0, time_lv, size)

# Define the system of differential equations
def pendulum_eq(y, t, a, b, c, d):
    theta, omega = y
    dydt = [a * theta - b * theta * omega, -c * omega + d * b * theta * omega]
    return dydt

# Initial conditions
y0 = [theta0, omega0]

# Solve the differential equations
sol = odeint(pendulum_eq, y0, t, args=(a, b, c, d))

def add_noise():
    noise = np.random.normal(size=(size, 2))
    simulated = sol + noise
    return simulated

# Noisy measurements
data = add_noise()

# Extract the results
theta_values = sol[:, 0]
omega_values = sol[:, 1]

start_time = time.time()

# State transition function using odeint
def f_cv(x, dt):
    sol = odeint(pendulum_eq, x, [0, dt], args=(a, b, c, d))
    return sol[-1]

# Measurement function
def h_cv(x):
    # theta_ = x[0]
    # omega_ = x[1]

    # pos_x = theta_ + theta_(a + b*omega_)*dt
    # pos_y = omega_ + omega_(-c + d*theta_)*dt

    # pos_x = np.sin(theta_)
    # pos_y = np.cos(theta_)
    # return [pos_x, pos_y]
    return x
    # return [theta_, omega_]

measurement_noise_std = 1

# UKF Model
points = MerweScaledSigmaPoints(2, alpha=1e-3, beta=2., kappa=1.0, sqrt_method=sqrt_func)
# points = MerweScaledSigmaPoints(2, alpha=0.1, beta=2., kappa=0.0, sqrt_method=sqrt_func)
ukf = UKF(dim_x=2, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=points)
ukf.P = np.diag([10, 10])  # Initialize with larger values
ukf.x = np.array([theta0, omega0])
ukf.R = np.diag([measurement_noise_std ** 2, measurement_noise_std ** 2])
ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=10, block_size=1)
# ukf.Q = np.eye(2)

# Simulation loop
kf_theta_values = []
kf_omega_values = []

for i in range(len(t)):
    # Get the noisy measurement
    measurement = data[i]

    # Kalman filter update
    ukf.predict()
    ukf.update(measurement)

    # Add jitter to the diagonal of P to maintain positive definiteness
    ukf.P += np.eye(2) * 1e-6

    # Store values for plotting
    kf_theta_values.append(ukf.x[0])
    kf_omega_values.append(ukf.x[1])

predictions = list(zip(kf_theta_values, kf_omega_values))

# Plotting
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(t, theta_values, label='True Theta')
# plt.plot(t, kf_theta_values, label='Kalman Filter Theta')
# plt.scatter(t, data[:, 0], label='Measurements')
# plt.xlabel('Time (s)')
# plt.ylabel('Theta (rad)')
# plt.legend()
#
# plt.subplot(2, 1, 2)
# plt.plot(t, omega_values, label='True Omega')
# plt.plot(t, kf_omega_values, label='Kalman Filter Omega')
# plt.scatter(t, data[:, 1], label='Measurements')
# plt.xlabel('Time (s)')
# plt.ylabel('Omega (rad/s)')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


sampling_time = time.time() - start_time

# print(sol)

rms = mean_squared_error(sol, predictions, squared=False)
print(f'RMSE of pendulum predictions using SMC: {rms}')
print(f'Sampling time of pendulum using SMC is: {sampling_time}')


fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
ax1.plot(data[:, 0], 'x', label='Pomiary')
ax1.plot(sol[:, 0], 'b', label='Modelowe wartości')
ax1.plot(kf_theta_values, label='Estymata - algorytm UKF')
ax1.legend(loc='upper right')
ax1.set_ylabel('Liczba ofiar')

ax2.plot(data[:, 1], 'x', label='Pomiary')
ax2.plot(sol[:, 1], 'g', label='Modelowe wartości')
ax2.plot(kf_omega_values, label='Estymata - algorytm UKF')

plt.xlabel('Czas')
ax2.legend(loc='upper right')
ax2.set_ylabel('Liczba drapieżników')
plt.savefig('obj2_ukf_predictions.pdf')

plt.show()
