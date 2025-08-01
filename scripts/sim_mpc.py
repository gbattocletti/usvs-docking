# import datetime
import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seacat_dp.control import linear_mpc, nonlinear_mpc
from seacat_dp.model import disturbances, seacat_model, seacat_pars
from seacat_dp.utils import io, settings
from seacat_dp.visualization import plot_functions

# Load simulation settings
sim_settings = settings.SimSettings()
# sim_settings.controller = "nonlinear_mpc"
sim_settings.controller = "linear_mpc"

# CHANGE SETTINGS HERE
sim_settings.sim_t_end = 180
sim_settings.q_ref = np.array(
    [
        10.0,  # x position [m]
        6.0,  # y position [m]
        0.0,  # yaw angle [rad]
        0.0,  # x velocity [m/s]
        0.0,  # y velocity [m/s]
        0.0,  # yaw rate [rad/s]
    ]
)
sim_settings.v_curr = 0.15  # current speed [m/s]
sim_settings.h_curr = -np.pi / 2  # current direction [rad]

########################################################################################

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Preprocess info for printing
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

# Print some info to the console
print(
    f"Simulation settings:\n"
    f"\tController: {sim_settings.controller} "
    f"(ctrl_dt: {sim_settings.ctrl_dt:.4f}, "
    f"ctrl_N: {sim_settings.ctrl_N})\n"
    f"\tInitial state: {q_0_str}\n"
    f"\tReference state: {q_ref_str}\n"
)

# Seed
np.random.seed(sim_settings.seed)  # for reproducibility

# Simulation parameters
sim_t = 0.0  # simulation time [s]
sim_dt = sim_settings.sim_dt  # simulation time step [s]
sim_t_end = sim_settings.sim_t_end  # simulation duration [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps []
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]
ctrl_t = 0.0  # time from the last control input [s] (used to trigger control).
ctrl_dt = sim_settings.ctrl_dt  # control time step [s]
ctrl_n = int(ctrl_dt / sim_dt)  # time steps per control step
ctrl_N = sim_settings.ctrl_N  # prediction horizon [steps]
controller = sim_settings.controller  # controller type

# Initialize model
params = seacat_pars.SeaCatParameters()
plant = seacat_model.SeaCatModel(params)
plant.set_time_step(sim_dt)
plant.set_integration_method("rk4")
plant.set_initial_conditions(sim_settings.q_0)

# Initialize disturbances
dist = disturbances.Disturbances()
dist.set_current_direction(sim_settings.h_curr)  # [rad]
dist.set_current_speed(sim_settings.v_curr)  # [m/s]
dist.set_wind_direction(sim_settings.h_wind)  # [rad]
dist.set_wind_speed(sim_settings.v_wind)  # [m/s]
v_curr = dist.current()  # (3, ) current speed in inertial frame
v_wind = dist.wind()  # (3, ) wind speed in inertial frame
b_curr = plant.crossflow_drag(v_curr)  # (3, ) current force in body frame
b_wind = plant.wind_load(v_wind)  # (3, ) wind force in body frame

# Initialize MPC controller
if controller == "linear_mpc":
    mpc = linear_mpc.LinearMpc()
    mpc.solver = "gurobi"  # modify this to change solver
elif controller == "nonlinear_mpc":
    mpc = nonlinear_mpc.NonlinearMpc()
elif controller == "pid":
    raise NotImplementedError("PID controller is not implemented in this script.")
else:
    raise ValueError(f"Unknown controller type: {controller}.")

mpc.set_dt(ctrl_dt)
mpc.set_horizon(ctrl_N)
mpc.set_discretization_method(sim_settings.discretization)
if controller == "linear_mpc":
    mpc.set_model(plant.M_inv, plant.D_L, plant.T, phi=plant.q[2])
elif controller == "nonlinear_mpc":
    mpc.set_model(plant.M_inv, plant.D_L, plant.T)
mpc.set_weights(sim_settings.Q, sim_settings.R, sim_settings.P)
mpc.set_input_bounds(sim_settings.u_min, sim_settings.u_max)
mpc.set_input_rate_bounds(sim_settings.delta_u_min / 5, sim_settings.delta_u_max / 5)
mpc.init_ocp()

# Initialize variables
q_ref = sim_settings.q_ref  # state reference
w_q = np.zeros(6)  # measurement noise
w_u = np.zeros(4)  # actuation noise
q_meas = np.zeros(6)  # measured state
u = np.zeros(4)  # control input
cost = 0.0  # mpc cost
t_sol = 0.0

# Helper variables
q_0 = np.zeros(6)  # copy of plant.q at the start of the MPC control step
q_pred = np.zeros((6, mpc.N + 1))  # predicted state (MPC solution)

# Initialize time series
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

# Run the simulation
initial_time = datetime.datetime.now()
print(f"\nSimulation started... [{initial_time.strftime('%H:%M:%S')}]")
for i in range(sim_n):

    # update control input
    if ctrl_t == 0.0 or ctrl_t >= ctrl_dt:

        # Print information if verbose mode is enabled
        # NOTE: to be able to display the updated plant state, the information is
        # printed at the beginning of the next control step, when the plant dynamics
        # has been rolled out for ctrl_n time steps.
        if sim_settings.verbose and i != 0:
            i_prev = max(0, i - ctrl_n)
            print(
                f"\nt = {i}/{sim_n} [{i / sim_n * 100:.4f}%]:"
                f"\n\tq plant =\t[{q_0[0]:.4f}, {q_0[1]:.4f}, {q_0[2]:.4f}, "
                f"{q_0[3]:.4f}, {q_0[4]:.4f}, {q_0[5]:.4f}]"
                f"\n\tq+ plant =\t[{plant.q[0]:.4f}, {plant.q[1]:.4f}, "
                f"{plant.q[2]:.4f}, {plant.q[3]:.4f}, {plant.q[4]:.4f}, "
                f"{plant.q[5]:.4f}]"
                f"\n\tw =\t\t[{w_q[0]:.4f}, {w_q[1]:.4f}, {w_q[2]:.4f}, "
                f"{w_q[3]:.4f}, {w_q[4]:.4f}, {w_q[5]:.4f}]"
                f"\n\tq mpc =\t\t[{q_pred[0, 0]:.4f}, {q_pred[1, 0]:.4f}, "
                f"{q_pred[2, 0]:.4f}, {q_pred[3, 0]:.4f}, {q_pred[4, 0]:.4f}, "
                f"{q_pred[5, 0]:.4f}]"
                f"\n\tq+ mpc =\t[{q_pred[0, 1]:.4f}, {q_pred[1, 1]:.4f}, "
                f"{q_pred[2, 1]:.4f}, {q_pred[3, 1]:.4f}, {q_pred[4, 1]:.4f}, "
                f"{q_pred[5, 1]:.4f}]"
                f"\n\tu =\t\t[{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}, {u[3]:.4f}]"
                f"\n\tw_u =\t\t[{w_u[0]:.4f}, {w_u[1]:.4f}, {w_u[2]:.4f}, {w_u[3]:.4f}]"
                f"\n\tb_curr =\t[{b_curr[0]:.4f}, {b_curr[1]:.4f}, {b_curr[2]:.4f}]"
                f"\n\tb_wind =\t[{b_wind[0]:.4f}, {b_wind[1]:.4f}, {b_wind[2]:.4f}]"
                f"\n\tcost mpc =\t{cost:.4f}"
            )

        q_0 = plant.q  # save the current state (used for debug when VERBOSE)
        w_q = dist.measurement_noise()  # generate measurement noise
        q_meas = plant.q + w_q  # measure the state (with noise)
        b_curr = plant.crossflow_drag(v_curr)  # (3, ) measure current force in body RF
        b_wind = plant.wind_load(v_wind)  # (3, ) measure wind force in body RF

        if controller == "linear_mpc":
            mpc.update_model(q_meas[2])  # linearize the model around current heading

        # solve mpc
        u_vec, q_pred, cost, t_sol = mpc.solve(q_meas, q_ref, b_curr, b_wind)

        # actuation noise and clipping
        u = u_vec[:, 0]  # extract the first control input from the solution
        w_u = dist.actuation_noise()  # generate actuation noise
        u = u + w_u  # add actuation noise
        u = np.clip(u, sim_settings.u_min, sim_settings.u_max)  # enforce input bounds
        ctrl_t = 0.0  # reset the elapsed time

    # perform one plant time step
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

    # print progress
    if not sim_settings.verbose:
        print(
            f"Simulation progress: {i+1}/{sim_n} [{(i+1) / sim_n * 100:.4f}%]", end="\r"
        )

final_time = datetime.datetime.now()
elapsed_time = final_time - initial_time
print(f"\nSimulation completed. [{final_time.strftime('%H:%M:%S')}]")
print(f"Elapsed time: {str(elapsed_time)}")

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
    fig_variables, _ = plot_functions.plot_variables(
        t_vec,
        u_mat,
        q_mat[:, :-1],
        q_ref_mat,
        cost_mat,
    )
    idx_list = list(np.linspace(0, sim_n, num=int(sim_n / 10000) + 1, dtype=int))
    fig_phase, _ = plot_functions.phase_plot(
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

# Generate and save animation
if sim_settings.save_anim is True:
    speed_up_factor = 500
    anim = plot_functions.generate_animation(
        t_vec,
        q_mat[:, :-1],
        q_ref_mat,
        u_mat,
        v_curr,
        v_wind,
        speed_up_factor,
    )
    io.save_animation(anim, sim_name)
