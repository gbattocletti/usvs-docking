import os
from pathlib import Path

import numpy as np
import scipy.linalg

from seacat_dp.control import linear_mpc
from seacat_dp.model import disturbances, nonlinear_model, parameters
from seacat_dp.utils import io

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
sim_t = 0.0  # simulation time [s]
sim_t_end = 60.0  # simulation duration [s]
sim_dt = 0.001  # simulation time step [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]
ctrl_dt = 0.1  # control time step [s]
ctrl_t = 0.0  # time from the last control input [s] (used to trigger control).

# Initialize model
params = parameters.Parameters()
plant = nonlinear_model.NonlinearModel(params)
plant.set_time_step(sim_dt)  # set the time step for the model
plant.set_integration_method("euler")  # set the integration method for the model
plant.set_initial_conditions(np.zeros(6))  # set the initial conditions for

# Initialize disturbances
dist = disturbances.Disturbances()
dist.set_current_direction(-np.pi / 4)  # set the current direction [rad]
dist.set_current_speed(10.0)  # set the current speed [m/s]
dist.set_wind_direction(0.0)  # set the wind direction [rad]
dist.set_wind_speed(0.0)  # set the wind speed [m/s]
b_curr = dist.current()  # current exogenous input (stationary, measured)
b_wind = dist.wind()  # wind exogenous input (stationary, measured)

# Initialize controller
mpc = linear_mpc.Mpc()  # initialize the MPC controller
mpc.set_horizon(10)  # set the prediction horizon for the MPC
H = plant.get_H()  # get the dynamic matrix H = -M^-1 * D_L (to build the A matrix)
mpc.set_A(plant.q[2], H)  # set the A matrix for the MPC
mpc.set_B(np.vstack((np.zeros((3, 4)), plant.T)))  # set the B matrix for the MPC
Q = scipy.linalg.block_diag(10 * np.eye(3), np.eye(3))  # set the Q matrix for the MPC
R = scipy.linalg.block_diag(np.eye(2), 10 * np.eye(2))  # set the R matrix for the MPC
P = scipy.linalg.block_diag(100 * np.eye(3), np.eye(3))  # set the P matrix for the MPC
mpc.set_weights(Q, R, P)  # set the matrices for the MPC
u_max = np.array(
    [
        params.max_stern_thrust_forward,
        params.max_stern_thrust_forward,
        params.max_bow_thrust_forward,
        params.max_bow_thrust_forward,
    ]
)
u_min = np.array(
    [
        params.max_stern_thrust_backward,
        params.max_stern_thrust_backward,
        params.max_bow_thrust_backward,
        params.max_bow_thrust_backward,
    ]
)
mpc.set_input_bounds(u_min, u_max)  # set input bounds for the MPC

# Initialize variables
q = plant.q  # initial state
q_ref = np.zeros(6)  # initialize reference state variable
w = np.zeros(6)  # initialize disturbance variable
q_meas = np.zeros(6)  # initialize measured state variable
u = np.zeros(4)  # initialize control input variable

# Initialize time series
# Note: time corresponds to the column, the matrices time step is set to sim_dt
q_mat = np.zeros((6, sim_n + 1))  # state time series
q_mat[:, 0] = plant.q  # save initial state
q_ref = np.zeros(6)  # reference state variable
q_ref[2] = np.pi / 4  # set reference heading angle [rad] (constant)
q_ref_mat = np.zeros((6, sim_n))  # reference state time series
w_mat = np.zeros((6, sim_n))  # disturbance time series
q_meas_mat = np.zeros((6, sim_n))  # state time series
u_mat = np.zeros((4, sim_n))  # control input time series

# Run the simulation
print("\nSimulation started...")
for i in range(sim_n):

    # update control input
    if ctrl_t == 0.0 or ctrl_t >= ctrl_dt:

        w = dist.measurement_noise()  # generate measurement noise
        q_meas = q + w  # measure the state (with noise)
        mpc.set_A(q_meas[2], H)  # linearize the model
        u_vec, q_pred = mpc.solve(q_meas, q_ref, b_curr, b_wind)  # solve mpc problem
        u = u_vec[:, 0]
        u = np.clip(u, u_min, u_max)  # enforce input bounds (redundant?)
        ctrl_t = 0.0  # reset the elapsed time

    # plant time step
    q = plant(u, b_curr, b_wind)  # update the model state

    # store step data
    q_mat[:, i + 1] = q  # state
    q_ref_mat[:, i] = q_ref  # reference state
    w_mat[:, i] = w  # noise
    q_meas_mat[:, i] = q_meas  # measured state
    u_mat[:, i] = u  # control input

    # update time
    ctrl_t += sim_dt
    sim_t += sim_dt

    # print progress
    print(f"Simulation progress: {i+1}/{sim_n} [{(i+1) / sim_n * 100:.2f}%]", end="\r")

print("\nSimulation completed.")

# Save the simulation data
io.save_sim_data(params, dist, t_vec, q_mat, w_mat, q_meas_mat, u_mat, b_curr, b_wind)
