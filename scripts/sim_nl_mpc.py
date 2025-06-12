# import datetime
import os
from pathlib import Path

import numpy as np
import scipy.linalg

from seacat_dp.control import nonlinear_mpc
from seacat_dp.model import disturbances, nonlinear_model, parameters

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
sim_t = 0.0  # simulation time [s]
sim_dt = 0.001  # simulation time step [s]
sim_t_end = 100.0  # simulation duration [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps []
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]
ctrl_t = 0.0  # time from the last control input [s] (used to trigger control).
ctrl_dt = 0.5  # control time step [s]
ctrl_n = int(ctrl_dt / sim_dt)  # time steps per control step
ctrl_N = 10  # prediction horizon

# Initialize model
params = parameters.Parameters()
plant = nonlinear_model.NonlinearModel(params)
plant.set_time_step(sim_dt)
plant.set_integration_method("rk4")
plant.set_initial_conditions(np.zeros(6))

# Initialize disturbances
dist = disturbances.Disturbances()
dist.set_current_direction(0.0)  # [rad]
dist.set_current_speed(0.0)  # [m/s]
dist.set_wind_direction(0.0)  # [rad]
dist.set_wind_speed(0.0)  # [m/s]
v_curr = dist.current()  # (3, ) current speed in inertial frame
v_wind = dist.wind()  # (3, ) wind speed in inertial frame
b_curr = plant.crossflow_drag(v_curr)  # (3, ) current force in body frame
b_wind = plant.wind_load(v_wind)  # (3, ) wind force in body frame

# Initialize MPC controller
mpc = nonlinear_mpc.NonlinearMpc()
mpc.set_dt(ctrl_dt)
mpc.set_horizon(ctrl_N)
mpc.set_discretization_method("rk4")
mpc.set_model(plant.M_inv, plant.D_L, plant.T)
Q = scipy.linalg.block_diag(10e3 * np.eye(2), 10e0, np.eye(2), 0.01)  # pos, vel
R = scipy.linalg.block_diag(10e-3 * np.eye(2), 10e-1 * np.eye(2))  # stern, bow
P = Q
mpc.set_weights(Q, R, P)
mpc.init_ocp()
