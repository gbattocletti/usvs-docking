import os
from pathlib import Path

import numpy as np

from seacat_dp.control import linear_mpc, linearize
from seacat_dp.model import disturbances, nonlinear_model, parameters
from seacat_dp.utils import io

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
t = 0.0  # initial time [s]
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
H = plant.get_H()  # get the dynamic matrix H
dist = disturbances.Disturbances()
dist.set_current_direction(-np.pi / 4)  # set the current direction [rad]
dist.set_current_speed(10.0)  # set the current speed [m/s]
dist.set_wind_direction(0.0)  # set the wind direction [rad]
dist.set_wind_speed(0.0)  # set the wind speed [m/s]
mpc = linear_mpc.Mpc()  # initialize the MPC controller # TODO add parameters
mpc.set_horizon(10)  # set the prediction horizon for the MPC
mpc.set_model(np.zeros((6, 6)), np.zeros((6, 4)), np.zeros((6, 6)), np.zeros((6, 4)))
mpc.set_weights(np.eye(6), np.eye(4), np.eye(6))  # set the weights for the MPC
b_curr = dist.current()  # current exogenous input (stationary, measured)
b_wind = dist.wind()  # wind exogenous input (stationary, measured)

# Initialize variables
q = np.zeros(6)  # state
w = np.zeros(6)  # disturbance
q_meas = np.zeros(6)  # measured state
u = np.zeros(4)  # control input

# Initialize time series
q_mat = np.zeros((6, sim_n + 1))  # state time series
w_mat = np.zeros((6, sim_n))  # disturbance time series
q_meas_mat = np.zeros((6, sim_n))  # state time series
u_mat = np.zeros((4, sim_n))  # control input time series

# Run the simulation
print("\nSimulation started...")
for i in range(sim_n):

    # update control input
    if ctrl_elapsed_t >= ctrl_dt:

        w = np.zeros(6)
        q_meas = q + w
        mpc.set_A(linearize.linearize(q[2], H))  # linearize the model
        u = np.zeros(4)  # update control input # TODO
        ctrl_elapsed_t = 0.0  # reset the elapsed time

    # plant time step
    q = plant(u, b_curr, b_wind)  # update the model state

    # store step data
    q_mat[:, i + 1] = q
    w_mat[:, i] = w  # noise
    q_meas_mat[:, i] = q_meas  # measured state
    u_mat[:, i] = u  # control input

    # update time
    ctrl_elapsed_t += sim_dt
    t += sim_dt

    # print progress
    print(f"Simulation progress: {i+1}/{sim_n} [{(i+1) / sim_n * 100:.2f}%]", end="\r")

print("\nSimulation completed.")

# Save the simulation data
io.save_sim_data(params, dist, t_vec, q_mat, w_mat, q_meas_mat, u_mat, b_curr, b_wind)
