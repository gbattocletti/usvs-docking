import os
from pathlib import Path

import numpy as np

from seacat_dp.models import disturbances, nonlinear_model, parameters
from seacat_dp.utils import io

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
t = 0.0  # time [s]
t_end = 60.0  # simulation duration [s]
dt = 0.001  # simulation time step [s]
n = int(t_end / dt)  # number of time steps
t_vec = dt * np.arange(n)  # time vector [s]

# Initialize the model
params = parameters.Parameters()
model = nonlinear_model.NonlinearModel(params)
model.set_time_step(dt)  # set the time step for the model
model.set_integration_method("euler")  # set the integration method for the model
model.set_initial_conditions(np.zeros(6))  # set the initial conditions for
dist = disturbances.Disturbances()
dist.set_current_direction(-np.pi / 4)  # set the current direction [rad]
dist.set_current_speed(10.0)  # set the current speed [m/s]
dist.set_wind_direction(0.0)  # set the wind direction [rad]
dist.set_wind_speed(0.0)  # set the wind speed [m/s]
b_current = dist.current()  # current exogenous input (stationary, measured)
b_wind = dist.wind()  # wind exogenous input (stationary, measured)

# Initialize variables
q = np.zeros(6)  # state
w = np.zeros(6)  # disturbance (CONSTANT)
q_meas = np.zeros(6)  # measured state (CONSTANT)
u = np.zeros(4)  # control input (CONSTANT)

# Initialize time series
q_mat = np.zeros((6, n + 1))  # state time series
w_mat = np.zeros((6, n))  # disturbance time series
q_meas_mat = np.zeros((6, n))  # state time series
u_mat = np.zeros((4, n))  # control input time series

# Run the simulation
print("\nSimulation started...")
for i in range(n):

    # plant
    q = model(u, b_current, b_wind)  # update the model state

    # store step data
    q_mat[:, i + 1] = q
    w_mat[:, i] = w  # noise
    q_meas_mat[:, i] = q_meas  # measured state
    u_mat[:, i] = u  # control input

    # update time
    t += dt

    # print progress
    print(f"Simulation progress: {i+1}/{n} [{(i+1) / n * 100:.2f}%]", end="\r")

print("\nSimulation completed.")

# Save the simulation data
io.save_sim_data(
    params, dist, t_vec, q_mat, w_mat, q_meas_mat, u_mat, b_current, b_wind
)
