"""
Script to simulate the single-agent NMPC for reference tracking.
"""

import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seacat_dp.control import linear_mpc, nonlinear_mpc
from seacat_dp.model import (
    disturbances,
    hydrodynamics,
    model_seacat,
    model_seadragon,
    parameters_seacat,
    parameters_seadragon,
    wind_dynamics,
)
from seacat_dp.utils import io, settings_sa
from seacat_dp.utils.wrappers import progress_sim
from seacat_dp.visualization import animate, plot

# Load simulation settings
sim_settings = settings_sa.SimSettings()
np.random.seed(sim_settings.seed)  # for reproducibility

### MANUAL CUSTOMIZATION OF SIMULATION SETTINGS ########################################

sim_settings.save_anim = True  # enable/disable animation saving

# Simulation duration [s]
sim_settings.sim_t_end = 60.0

# Controller settings
# NOTE: specifying the usv_type also loads the appropriate controller options. If you
# want to manually set the controller options, do it after specifying the usv_type.
sim_settings.usv_type = "seadragon"  # USV model type: {"seacat", "seadragon"}
sim_settings.controller = "nonlinear_mpc"  # Controller: {"linear_mpc", "nonlinear_mpc"}
sim_settings.ctrl_N = 20  # Prediction horizon
sim_settings.ctrl_dt = 0.5  # control time step [s]

# Initial state
sim_settings.q_0 = np.array(
    [
        0.0,  # x position [m]
        0.0,  # y position [m]
        0.0,  # yaw angle [rad]
        0.0,  # x velocity [m/s]
        0.0,  # y velocity [m/s]
        0.0,  # yaw rate [rad/s]
    ]
)

# Target state
sim_settings.q_ref = np.array(
    [
        8.0,  # x position [m]
        8.0,  # y position [m]
        0.0,  # yaw angle [rad]
        0.0,  # x velocity [m/s]
        0.0,  # y velocity [m/s]
        0.0,  # yaw rate [rad/s]
    ]
)

# Exogenous disturbances
sim_settings.v_wind = 0.0  # wind speed [m/s]
sim_settings.h_wind = 0.0  # wind direction [rad]
sim_settings.v_curr = 0.0  # current speed [m/s]
sim_settings.h_curr = 0.0  # current direction [rad]
enable_measurement_noise: bool = False  # enable/disable measurement noise
enable_actuation_noise: bool = False  # enable/disable actuation noise

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

q_ref_str = "["
for i, q in enumerate(sim_settings.q_ref):
    if i == len(sim_settings.q_ref) - 1:
        q_ref_str += f"{q:.2f}"
    else:
        q_ref_str += f"{q:.2f}, "
q_ref_str += "]"
print(
    "=" * 53 + "\n"
    f"USV type: {sim_settings.usv_type}\n"
    f"Controller: {sim_settings.controller} "
    f"(ctrl_dt: {sim_settings.ctrl_dt:.2f}, "
    f"ctrl_N: {sim_settings.ctrl_N})\n"
    f"Initial state:\t {q_0_str}\n"
    f"Reference state: {q_ref_str}\n" + "=" * 53 + "\n",
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

# Initialize model
match sim_settings.usv_type:
    case "seacat":
        params = parameters_seacat.SeaCatParameters()
        plant = model_seacat.SeaCatModel(params)
    case "seadragon":
        params = parameters_seadragon.SeaDragonParameters()
        plant = model_seadragon.SeaDragonModel(params)
plant.set_time_step(sim_dt)  # use finer time step for plant simulation
plant.set_integration_method("rk4")
plant.set_initial_conditions(sim_settings.q_0)

# Initialize exogenous inputs
dist = disturbances.Disturbances()
dist.set_current_direction(sim_settings.h_curr)  # [rad]
dist.set_current_speed(sim_settings.v_curr)  # [m/s]
dist.set_wind_direction(sim_settings.h_wind)  # [rad]
dist.set_wind_speed(sim_settings.v_wind)  # [m/s]
v_curr = dist.current()  # (3, ) current speed in inertial frame
v_wind = dist.wind()  # (3, ) wind speed in inertial frame
b_curr = hydrodynamics.crossflow_drag(plant.q, params, v_curr)  # (3, ) force in body RF
b_wind = wind_dynamics.wind_load(plant.q, v_wind)  # (3, ) wind force in body RF

# Initialize MPC controller
match sim_settings.controller:
    case "linear_mpc":
        mpc = linear_mpc.LinearMpc()  # only if usv_type is "seacat"
        mpc.solver = "cplex"  # can be modified if needed
    case "nonlinear_mpc":
        mpc = nonlinear_mpc.NonlinearMpc()
mpc.set_dt(ctrl_dt)
mpc.set_horizon(ctrl_N)
mpc.set_discretization_method(sim_settings.discretization)
match sim_settings.usv_type:
    case "seacat":
        match sim_settings.controller:
            case "linear_mpc":
                mpc.set_model(plant.M_inv, plant.D_L, plant.T, phi=plant.q[2])
            case "nonlinear_mpc":
                mpc.set_model(plant.M_inv, plant.D_L, plant.T)
    case "seadragon":
        mpc.set_model(
            plant.M_inv,
            plant.D_L,
            azimuth_thrusters=True,
            pars_b_thrusters=0.85,
            pars_l_thrusters=1.52,
        )
mpc.set_weights(sim_settings.Q, sim_settings.R, sim_settings.P)
mpc.set_input_bounds(sim_settings.u_min, sim_settings.u_max)
mpc.set_input_rate_bounds(sim_settings.delta_u_min / 5, sim_settings.delta_u_max / 5)
mpc.init_ocp()

# Initialize variables
q_ref = sim_settings.q_ref  # reference state
q_meas = np.zeros(6)  # measured state
q_pred = np.zeros((6, mpc.N + 1))  # predicted state (MPC solution)
u = np.zeros(4)  # control input
w_q = np.zeros(6)  # measurement noise
w_u = np.zeros(4)  # actuation noise
cost = 0.0  # mpc cost
t_sol = 0.0

# Initialize time series to store simulation data
# NOTE: time increases along the column direction, i.e., q_mat[:, i] is q at time i*dt
q_mat = np.zeros((6, sim_n + 1))
q_mat[:, 0] = plant.q
q_ref_mat = np.zeros((6, sim_n))
q_meas_mat = np.zeros((6, sim_n))
u_mat = np.zeros((4, sim_n))
w_q_mat = np.zeros((6, sim_n))
w_u_mat = np.zeros((4, sim_n))
cost_mat = np.zeros(sim_n)
sol_t_mat = np.zeros(sim_n)

# RUN THE SIMULATION ###################################################################
for i in progress_sim(range(sim_n), dt=sim_dt):

    # update control input
    if ctrl_t == 0.0 or ctrl_t >= ctrl_dt:

        if enable_measurement_noise is True:
            w_q = dist.measurement_noise()  # generate measurement noise
            q_meas = plant.q + w_q  # measure the state (with noise)
        else:
            w_q = np.zeros(6)
            q_meas = plant.q

        # estimate disturbances
        # TODO: implement disturbance estimator (currently uses exact measurement)
        b_curr = hydrodynamics.crossflow_drag(plant.q, params, v_curr)  # force body RF
        b_wind = wind_dynamics.wind_load(plant.q, v_wind)  # wind force in body RF

        # solve mpc
        u_vec, q_pred, cost, t_sol = mpc.solve(
            q_meas,
            q_ref,
            b_curr,
            b_wind,
            use_warm_start=True,
        )

        # actuation noise and clipping
        u = u_vec[:, 0]  # extract the first control input from the solution
        if enable_actuation_noise is True:
            match sim_settings.usv_type:
                case "seacat":
                    w_u = dist.actuation_noise_seacat()  # generate actuation noise
                case "seadragon":
                    w_u = dist.actuation_noise_seadragon()  # generate actuation noise
            u = u + w_u  # add actuation noise
        else:
            w_u = np.zeros(4)
        u = np.clip(u, sim_settings.u_min, sim_settings.u_max)  # enforce input bounds
        ctrl_t = 0.0  # reset the elapsed time

    # perform one plant time step (with time step sim_dt < ctrl_dt)
    plant.step(u, v_curr, v_wind)

    # store step data
    q_mat[:, i + 1] = plant.q
    q_ref_mat[:, i] = q_ref
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

# Save simulation data
io.save_sim_data(
    sim_name,
    params,
    sim_settings,
    plant,
    mpc,
    dist,
    t_vec,
    q_ref_mat,
    q_mat,
    q_meas_mat,
    u_mat,
    w_q_mat,
    w_u_mat,
    v_curr,
    v_wind,
    b_curr,
    b_wind,
    cost_mat,
    sol_t_mat,
)

# Generate and save plots
fig_variables = None  # Initialize to None to avoid errors
fig_phase = None

if sim_settings.save_plots is True or sim_settings.show_plots is True:
    fig_variables, _ = plot.plot_variables(
        t_vec,
        u_mat,
        q_mat[:, :-1],
        q_ref_mat,
        cost_mat,
    )
    idx_list = list(np.linspace(0, sim_n, num=int(sim_n / 10000) + 1, dtype=int))
    fig_phase, _ = plot.phase_plot(
        q_mat[:, :-1],
        v_curr,
        v_wind,
        # idx=idx_list,
    )

if sim_settings.save_plots is True:
    io.save_figure(fig_variables, sim_name, "variables")
    io.save_figure(fig_phase, sim_name, "phase-plot")

if sim_settings.show_plots is True:
    plt.show(block=False)

if sim_settings.save_anim is True:
    speed_up_factor = 500
    anim = animate.generate_animation(
        t_vec,
        q_mat[:, :-1],
        q_ref_mat,
        u_mat,
        v_curr,
        v_wind,
        speed_up_factor,
    )  # Generate and save animation
    io.save_animation(anim, sim_name)
