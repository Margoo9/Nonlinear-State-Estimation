import numpy as np


number_of_points = 128

# pendulum params
init_point_pendulum = [0.75 * np.pi, 0.0]
t_pend = np.linspace(0, 10, number_of_points)
b = 0.3    # damping coefficient -> effect of friction
c = 4.0     # const, depends on the length and mass of the pendulum
# b*omega - damping or frictional force
# c * sin(theta) - gravitational force

# lotka-volterra params
init_point_lv = [15, 7]
t_lv = np.linspace(0, 100, number_of_points)
alpha = 1.5
beta = 1.2
gamma = 1.3
delta = 0.9
