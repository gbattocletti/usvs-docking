# import matplotlib.pyplot as plt
import os
from pathlib import Path

import numpy as np

from seacat_dp.models import disturbances, nonlinear_model, parameters
from seacat_dp.util import io

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
sim_t_end = 60.0  # simulation duration [s]
sim_dt = 0.001  # simulation time step [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]
ctrl_dt = 0.1  # control time step [s]
ctrl_n = int(ctrl_dt / sim_dt)  # number of time steps for control input
ctrl_elapsed_t = 1.0  # time from the last control input [s] (used to trigger control).
# The value is initialized to 1.0 to trigger the first control input update at t = 0.

# Initialize the model
params = parameters.Parameters()
plant = nonlinear_model.NonlinearModel(params)
plant.set_time_step(sim_dt)  # set the time step for the model
plant.set_integration_method("euler")  # set the integration method for the model
plant.set_initial_conditions(np.zeros(6))  # set the initial conditions for
dist = disturbances.Disturbances()
dist.set_current_direction(-np.pi / 4)  # set the current direction [rad]
dist.set_current_speed(10.0)  # set the current speed [m/s]
dist.set_wind_direction(0.0)  # set the wind direction [rad]
dist.set_wind_speed(0.0)  # set the wind speed [m/s]
b_current = dist.current()  # current exogenous input (stationary, measured)
b_wind = dist.wind()  # wind exogenous input (stationary, measured)

# Initialize variables
w = np.zeros(6)  # disturbance
q_meas = np.zeros(6)  # measured state
u = np.zeros(4)  # control input

# Initialize time series
q_mat = np.zeros((6, sim_n + 1))  # state time series
u_mat = np.zeros((4, sim_n))  # control input time series
w_mat = np.zeros((6, sim_n))  # disturbance time series
q_meas_mat = np.zeros((6, sim_n))  # state time series

# Run the simulation
t = 0.0  # current time [s]
for i in range(sim_n):

    # controller
    if ctrl_elapsed_t >= ctrl_dt:

        # measure state
        w = dist.measurement_noise()
        q_meas = q_mat[:, i] + w

        # compute control input
        u = np.zeros(4)  # control input [N]

        # reset the elapsed time counter
        ctrl_elapsed_t = 0.0

    # plant
    q_mat[:, i + 1] = plant(u, b_current, b_wind)  # update the model state

    # store time step data
    w_mat[:, i] = w  # noise
    q_meas_mat[:, i] = q_meas  # measured state
    u_mat[:, i] = u  # control input

    # update time
    t += sim_dt
    ctrl_elapsed_t += sim_dt

    # print progress
    # if i % 100 == 0:
    print(f"Simulation progress: {i+1}/{sim_n} [{(i+1) / sim_n * 100:.2f}%]", end="\r")

# Save the simulation data
io.save_sim_data(q_mat, w_mat, q_meas_mat, u_mat, b_current, b_wind)
