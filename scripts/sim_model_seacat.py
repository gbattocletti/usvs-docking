import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seacat_dp.model import disturbances, model_seacat, parameters_seacat
from seacat_dp.utils.wrappers import progress_sim
from seacat_dp.visualization import plot

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
t = 0.0  # time [s]
t_end = 60.0  # simulation duration [s]
sim_dt = 0.001  # simulation time step [s]
sim_n = int(t_end / sim_dt)  # number of time steps
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]

# Initialize the model
params = parameters_seacat.SeaCatParameters()
model = model_seacat.SeaCatModel(params)
model.set_time_step(sim_dt)  # set the time step for the model
model.set_integration_method("euler")  # set the integration method for the model
model.set_initial_conditions(np.zeros(6))  # set the initial conditions for
dist = disturbances.Disturbances()
dist.set_current_direction(np.pi / 2)  # set the current direction [rad]
dist.set_current_speed(0.3)  # set the current speed [m/s] (max speed is 1.0 m/s)
v_current = dist.current()  # current exogenous input (stationary, measured)
v_wind = dist.wind()  # wind exogenous input (stationary, measured)

# Initialize variables
q = np.zeros(6)  # state
w = np.zeros(6)  # disturbance
q_meas = np.zeros(6)  # measured state
u = np.zeros(4)  # control input
u[0] = 100  # stern left
u[1] = 100  # stern right
u[2] = 0  # bow left
u[3] = 0  # bow right

# Initialize time series
q_mat = np.zeros((6, sim_n + 1))  # state time series
w_mat = np.zeros((6, sim_n))  # disturbance time series
q_meas_mat = np.zeros((6, sim_n))  # state time series
u_mat = np.zeros((4, sim_n))  # control input time series

# Run the simulation
for i in progress_sim(range(sim_n), dt=sim_dt):

    # plant
    q = model(u, v_current, v_wind)  # update the model state

    # store step data
    q_mat[:, i + 1] = q
    w_mat[:, i] = w  # noise
    q_meas_mat[:, i] = q_meas  # measured state
    u_mat[:, i] = u  # control input

    # update time
    t += sim_dt

# Plot the simulation data
plot.plot_variables(t_vec, u_mat, q_mat[:, :-1])
plot.phase_plot(q_mat[:, :-1], v_current, v_wind)
plt.show()
