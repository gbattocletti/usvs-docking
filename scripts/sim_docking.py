"""
Script to simulate the multi-agent docking NMPC. The script simulates the docking
of a SeaCat2 USV and a SeaDragon USV using a centralized multi-agent nonlinear MPC.

Acronyms:
- USV: Unmanned Surface Vehicle
- MPC: Model Predictive Control
- NMPC: Nonlinear Model Predictive Control
- SC: SeaCat2
- SD: SeaDragon
"""

import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seacat_dp.control import ma_mpc
from seacat_dp.model import (
    disturbances,
    hydrodynamics,
    model_seacat,
    model_seadragon,
    parameters_seacat,
    parameters_seadragon,
    wind_dynamics,
)
from seacat_dp.utils import io, settings_ma
from seacat_dp.utils.wrappers import progress_sim
from seacat_dp.visualization import plot_ma

# Load simulation settings
sim_settings = settings_ma.SimSettings()
np.random.seed(sim_settings.seed)  # for reproducibility

### MANUAL CUSTOMIZATION OF SIMULATION SETTINGS ########################################

# Simulation duration [s]
sim_settings.sim_t_end = 60.0

# Controller settings
sim_settings.ctrl_N = 20  # Prediction horizon
sim_settings.ctrl_dt = 0.5  # control time step [s]

# Initial state (joint state (12, ) of SeaCat and SeaDragon)
sim_settings.q_0 = np.array(
    [
        0.0,  # x position SeaCat [m]
        0.0,  # y position SeaCat [m]
        0.0,  # yaw angle SeaCat [rad]
        0.0,  # x velocity SeaCat [m/s]
        0.0,  # y velocity SeaCat [m/s]
        0.0,  # yaw rate SeaCat [rad/s]
        10.0,  # x position SeaDragon [m]
        10.0,  # y position SeaDragon [m]
        0.0,  # yaw angle SeaDragon [rad]
        0.0,  # x velocity SeaDragon [m/s]
        0.0,  # y velocity SeaDragon [m/s]
        0.0,  # yaw rate SeaDragon [rad/s]
    ]
)

# Exogenous disturbances
sim_settings.v_wind = 0.0  # wind speed [m/s]
sim_settings.h_wind = 0.0  # wind direction [rad]
sim_settings.v_curr = 0.0  # current speed [m/s]
sim_settings.h_curr = 0.0  # current direction [rad]
sim_settings.enable_measurement_noise = False  # enable/disable measurement noise
sim_settings.enable_actuation_noise = False  # enable/disable actuation noise

# Plot settings
sim_settings.save_anim = True  # enable/disable animation saving

########################################################################################

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Print initial conditions to the console
q_0_str = "["
for i, q in enumerate(sim_settings.q_0):
    if i == len(sim_settings.q_0) - 1:
        q_0_str += f"{q:.2f}"
    else:
        q_0_str += f"{q:.2f}, "
q_0_str += "]"
print(
    "=" * 53 + "\n"
    "USV type: SeaCat + SeaDragon\n"
    f"(ctrl_dt: {sim_settings.ctrl_dt:.2f}, "
    f"ctrl_N: {sim_settings.ctrl_N})\n"
    f"Initial state:\t {q_0_str}\n"
)

# SETUP SIMULATION #####################################################################
# Simulation parameters
sim_t = 0.0  # simulation time [s]
sim_dt = sim_settings.sim_dt  # simulation time step [s]
sim_t_end = sim_settings.sim_t_end  # simulation duration [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps []
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]

# Controller parameters
ctrl_t = 0.0  # time from the last control input [s] (used to trigger control).
ctrl_dt = sim_settings.ctrl_dt  # control time step [s]
ctrl_n = int(ctrl_dt / sim_dt)  # time steps per control step
ctrl_N = sim_settings.ctrl_N  # prediction horizon [steps]

# Initialize plants
params_sc = parameters_seacat.SeaCatParameters()
plant_sc = model_seacat.SeaCatModel(params_sc)
plant_sc.set_time_step(sim_dt)
plant_sc.set_integration_method("rk4")
plant_sc.set_initial_conditions(sim_settings.q_0[0:6])
params_sd = parameters_seadragon.SeaDragonParameters()
plant_sd = model_seadragon.SeaDragonModel(params_sd)
plant_sd.set_time_step(sim_dt)
plant_sd.set_integration_method("rk4")
plant_sd.set_initial_conditions(sim_settings.q_0[6:12])

# Initialize exogenous inputs
dist = disturbances.Disturbances()
dist.set_current_direction(sim_settings.h_curr)  # [rad]
dist.set_current_speed(sim_settings.v_curr)  # [m/s]
dist.set_wind_direction(sim_settings.h_wind)  # [rad]
dist.set_wind_speed(sim_settings.v_wind)  # [m/s]
v_curr = dist.current()  # (3, ) current speed in inertial frame
v_wind = dist.wind()  # (3, ) wind speed in inertial frame
b_curr_sc = hydrodynamics.crossflow_drag(
    plant_sc.q, params_sc, v_curr
)  # (3, ) current force in body RF for SeaCat
b_wind_sc = wind_dynamics.wind_load(
    plant_sc.q, v_wind
)  # (3, ) wind force in body RF for SeaCat
b_curr_sd = hydrodynamics.crossflow_drag(
    plant_sd.q, params_sd, v_curr
)  # (3, ) current force in body RF for SeaDragon
b_wind_sd = wind_dynamics.wind_load(
    plant_sd.q, v_wind
)  # (3, ) wind force in body RF for SeaDragon

# Initialize MPC controller
mpc = ma_mpc.DockingMpc()
mpc.set_dt(ctrl_dt)
mpc.set_horizon(ctrl_N)
mpc.set_discretization_method(sim_settings.discretization)
mpc.set_model(plant_sc, plant_sd)
mpc.set_weights(sim_settings.Q, sim_settings.R, sim_settings.P)
mpc.set_input_bounds(sim_settings.u_min, sim_settings.u_max)
mpc.set_input_rate_bounds(sim_settings.delta_u_min / 5, sim_settings.delta_u_max / 5)
mpc.init_ocp()

# Initialize variables
q_meas = np.zeros(12)  # joint measured state
q_pred = np.zeros((12, mpc.N + 1))  # joint predicted state (MPC solution)
u = np.zeros(8)  # joint control input
w_q = np.zeros(12)  # joint measurement noise
w_u = np.zeros(8)  # joint actuation noise
cost = 0.0  # mpc cost
t_sol = 0.0

# Initialize time series to store simulation data
# NOTE: time increases along the column direction, i.e., q_mat[:, i] is q at time i*dt
q_mat = np.zeros((12, sim_n + 1))
q_mat[:, 0] = np.hstack((plant_sc.q, plant_sd.q))
q_ref_mat = np.zeros((12, sim_n))
q_meas_mat = np.zeros((12, sim_n))
u_mat = np.zeros((8, sim_n))
w_q_mat = np.zeros((12, sim_n))
w_u_mat = np.zeros((8, sim_n))
cost_mat = np.zeros(sim_n)
sol_t_mat = np.zeros(sim_n)

# RUN THE SIMULATION ###################################################################
for i in progress_sim(range(sim_n), dt=sim_dt):

    # update control input
    if ctrl_t == 0.0 or ctrl_t >= ctrl_dt:

        # measure the state (possibly with noise)
        if sim_settings.enable_measurement_noise is True:
            w_q_sc = dist.measurement_noise()  # generate measurement noise for SC
            w_q_sd = dist.measurement_noise()  # generate measurement noise for SD
            q_meas = np.hstack((plant_sc.q + w_q_sc, plant_sd.q + w_q_sd))
        else:
            w_q = np.zeros(12)
            q_meas = np.hstack((plant_sc.q, plant_sd.q))

        # estimate disturbances
        b_curr_sc = hydrodynamics.crossflow_drag(plant_sc.q, params_sc, v_curr)
        b_wind_sc = wind_dynamics.wind_load(plant_sc.q, v_wind)
        b_curr_sd = hydrodynamics.crossflow_drag(plant_sd.q, params_sd, v_curr)
        b_wind_sd = wind_dynamics.wind_load(plant_sd.q, v_wind)

        # solve mpc
        u_vec, q_pred, cost, t_sol = mpc.solve(
            q_meas,
            b_curr_sc,
            b_wind_sc,
            b_curr_sd,
            b_wind_sd,
            use_warm_start=True,
        )

        # actuation noise and clipping
        u = u_vec[:, 0]  # extract the first control input from the solution
        if sim_settings.enable_actuation_noise is True:
            w_u_sc = dist.actuation_noise_seacat()  # SeaCat actuation noise
            w_u_sd = dist.actuation_noise_seadragon()  # SeaDragon actuation noise
            u = u + np.hstack((w_u_sc, w_u_sd))  # add actuation noise
        else:
            w_u = np.zeros(8)  # only for logging purposes
        u = np.clip(u, sim_settings.u_min, sim_settings.u_max)  # enforce input bounds
        ctrl_t = 0.0  # reset the elapsed time

    # perform one plant time step (with time step sim_dt < ctrl_dt)
    plant_sc.step(u[:4], v_curr, v_wind)
    plant_sd.step(u[4:], v_curr, v_wind)

    # store step data
    q_mat[:, i + 1] = np.hstack((plant_sc.q, plant_sd.q))
    w_q_mat[:, i] = w_q
    w_u_mat[:, i] = w_u
    q_meas_mat[:, i] = q_meas
    u_mat[:, i] = u
    cost_mat[i] = cost
    sol_t_mat[i] = t_sol

    # update time
    ctrl_t += sim_dt

final_time = datetime.datetime.now()
print(f"\nSimulation completed. [{final_time.strftime('%H:%M:%S')}]")


# SAVE RESULTS #########################################################################
# Generate filename to save data
sim_name, _ = io.generate_filename()

# Set show/save plot settings
sim_settings.show_plots = True  # enable/disable plot showing
sim_settings.save_plots = False  # enable/disable plot saving
sim_settings.save_anim = False  # enable/disable animation saving

# Save simulation data
fig_variables, _ = plot_ma.plot_variables(
    t_vec,
    u_mat,
    q_mat[:, :-1],
    q_ref_mat,
    cost_mat,
)
idx_list = list(np.linspace(0, sim_n, num=int(sim_n / 10000) + 1, dtype=int))
fig_phase, _ = plot_ma.phase_plot(
    q_mat[:, :-1],
    v_curr,
    v_wind,
    # idx=idx_list,
)

plt.show(block=True)

# Generate and save plots
# TODO: update plot functions to handle multi-agent simulations
