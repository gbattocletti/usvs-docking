import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from seacat_dp.control import linear_mpc
from seacat_dp.model import disturbances, nonlinear_model, parameters
from seacat_dp.utils import io
from seacat_dp.visualization import plot_functions

# TogSimulations settings
VERBOSE = True  # set to True to print detailed information
SHOW_PLOTS = False  # set to True to show plots at the end of the simulation

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Simulation parameters
sim_t = 0.0  # simulation time [s]
sim_t_end = 120.0  # simulation duration [s]
sim_dt = 0.001  # simulation time step [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps []
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]
ctrl_t = 0.0  # time from the last control input [s] (used to trigger control).
ctrl_dt = 0.5  # control time step [s]
ctrl_n = int(ctrl_dt / sim_dt)  # time steps per control step
ctrl_N = 30  # prediction horizon

# Initialize model
params = parameters.Parameters()
plant = nonlinear_model.NonlinearModel(params)
plant.set_time_step(sim_dt)
plant.set_integration_method("rk4")
plant.set_initial_conditions(np.zeros(6))

# Initialize disturbances
dist = disturbances.Disturbances()
dist.set_current_direction(np.pi + np.pi / 3.0)  # [rad]
dist.set_current_speed(0.2)  # [m/s]
dist.set_wind_direction(0.0)  # [rad]
dist.set_wind_speed(0.0)  # [m/s]
v_curr = dist.current()  # (3, ) current speed in inertial frame
v_wind = dist.wind()  # (3, ) wind speed in inertial frame
b_curr = plant.crossflow_drag(v_curr)  # (3, ) current force in body frame
b_wind = plant.wind_load(v_wind)  # (3, ) wind force in body frame

# Initialize MPC controller
mpc = linear_mpc.LinearMpc()
mpc.set_dt(ctrl_dt)
mpc.set_horizon(ctrl_N)
mpc.set_discretization_method("zoh")
mpc.set_model(plant.M_inv, plant.D_L, plant.T, plant.q[2])
Q = scipy.linalg.block_diag(10e3 * np.eye(2), 10e1, np.eye(2), 0.01)  # pos, vel
R = scipy.linalg.block_diag(10e-3 * np.eye(2), 5 * 10e-1 * np.eye(2))  # stern, bow
P = Q
mpc.set_weights(Q, R, P)
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
mpc.set_input_bounds(u_min, u_max)
delta_u_max = np.array(
    [
        params.max_stern_thrust_forward / params.t_stern * ctrl_dt,
        params.max_stern_thrust_forward / params.t_stern * ctrl_dt,
        params.max_bow_thrust_forward / params.t_bow * ctrl_dt,
        params.max_bow_thrust_forward / params.t_bow * ctrl_dt,
    ]
)  # [N/s]
delta_u_min = np.array(
    [
        params.max_stern_thrust_backward / params.t_stern * ctrl_dt,
        params.max_stern_thrust_backward / params.t_stern * ctrl_dt,
        params.max_bow_thrust_backward / params.t_bow * ctrl_dt,
        params.max_bow_thrust_backward / params.t_bow * ctrl_dt,
    ]
)  # [N/s]
mpc.set_input_rate_bounds(delta_u_min / 5, delta_u_max / 5)
mpc.init_ocp()

# Initialize variables
q_ref = np.zeros(6)  # state reference
q_ref[0] = 6.0  # [m]
q_ref[1] = -3.0  # [m]
q_ref[2] = np.pi / 3  # [rad]
w_q = np.zeros(6)  # measurement noise
w_u = np.zeros(4)  # actuation noise
q_meas = np.zeros(6)  # measured state
u = np.zeros(4)  # control input
cost = 0.0  # mpc cost

# Helper variables
q0 = np.zeros(6)  # copy of plant.q at the start of the MPC control step
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


# Run the simulation
print(f"\nSimulation started... [{datetime.datetime.now().strftime('%H:%M:%S')}]")
for i in range(sim_n):

    # update control input
    if ctrl_t == 0.0 or ctrl_t >= ctrl_dt:

        # Print information if verbose mode is enabled
        # NOTE: to be able to display the updated plant state, the information is
        # printed at the beginning of the next control step, when the plant dynamics
        # has been rolled out for ctrl_n time steps.
        if VERBOSE and i != 0:
            i_prev = max(0, i - ctrl_n)
            print(
                f"\nt = {i}/{sim_n} [{i / sim_n * 100:.4f}%]:"
                f"\n\tq plant =\t[{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, "
                f"{q0[3]:.4f}, {q0[4]:.4f}, {q0[5]:.4f}]"
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

        q0 = plant.q  # save the current state (used for debug when VERBOSE)
        w_q = dist.measurement_noise()  # generate measurement noise
        q_meas = plant.q + w_q  # measure the state (with noise)
        mpc.update_model(q_meas[2])  # linearize the model around current heading
        b_curr = plant.crossflow_drag(v_curr)  # (3, ) measure current force in body RF
        b_wind = plant.wind_load(v_wind)  # (3, ) measure wind force in body RF
        u_vec, q_pred, cost = mpc.solve(q_meas, q_ref, b_curr, b_wind)  # solve mpc
        u = u_vec[:, 0]  # extract the first control input from the solution
        w_u = dist.actuation_noise()  # generate actuation noise
        u = u + w_u  # add actuation noise
        u = np.clip(u, u_min, u_max)  # enforce input bounds
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

    # update time
    ctrl_t += sim_dt
    sim_t += sim_dt

    # print progress
    if not VERBOSE:
        print(
            f"Simulation progress: {i+1}/{sim_n} [{(i+1) / sim_n * 100:.4f}%]", end="\r"
        )

print(f"\nSimulation completed. [{datetime.datetime.now().strftime('%H:%M:%S')}]")

# Generate filename to save data
filename = io.generate_filename()

# Save simulation data
io.save_sim_data(
    filename,
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
    b_curr,
    b_wind,
)

# Generate and save plots
fig_variables, ax_variables = plot_functions.plot_variables(
    t_vec, u_mat, q_mat[:, :-1], q_ref_mat, cost_mat
)
fig_phase, ax_phase = plot_functions.phase_plot(q_mat[:, :-1], v_curr, v_wind)
io.save_figure(fig_variables, filename, "variables")
io.save_figure(fig_phase, filename, "phase-plot")

if SHOW_PLOTS == True:
    plt.show(block=False)

# Generate and save animation
speed_up_factor = 100
anim = plot_functions.generate_animation(
    t_vec, q_mat[:, :-1], q_ref_mat, u_mat, v_curr, v_wind, speed_up_factor
)
io.save_animation(anim, filename)
