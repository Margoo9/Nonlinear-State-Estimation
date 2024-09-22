import numpy as np


num_runs = 10

# pendulum params
init_point_pendulum = [0.75 * np.pi, 0.0]
number_of_points_pendulum = 100
t_pend = np.linspace(0, 15, number_of_points_pendulum)
b = 0.3    # damping coefficient -> effect of friction
c = 4.0     # const, depends on the length and mass of the pendulum
# b*omega - damping or frictional force
# c * sin(theta) - gravitational force

# lotka-volterra params
init_point_lv = [15, 7]
number_of_points_lv = 100
t_lv = np.linspace(0, 32, number_of_points_lv)
alpha = 0.9
beta = 0.5
gamma = 0.75
delta = 0.25
