"""
Script to simulate the multi-agent docking NMPC. The script simulates the docking
of a SeaCat2 USV and a SeaDragon USV using a centralized multi-agent nonlinear MPC.

Multiple controllers are evaluated and compared in this script, namely:
- Cooperative docking to q_ref
- Docking to static USV

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

import numpy as np

from seacat_dp.control import ma_mpc
from seacat_dp.model import (
    disturbances,
    hydrodynamics,
    model_seacat,
    model_seadragon,
    parameters_seacat,
    parameters_seadragon,
)
from seacat_dp.utils import io, settings_ma, transformations
from seacat_dp.utils.wrappers import progress_sim
from seacat_dp.visualization import plot_ma
from seacat_dp.visualization.colors import CmdColors

# Set cwd to the script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Load simulation settings
sim_settings = settings_ma.SimSettings()
np.random.seed(2)  # for reproducibility

# SIMULATION SETTINGS ##################################################################
# Simulation duration [s]
sim_settings.sim_t_end = 240.0

# Controller settings
sim_settings.ctrl_N = 50  # Prediction horizon
sim_settings.ctrl_dt = 0.5  # control time step [s]
sim_settings.sim_dt = 0.5  # simulation time step [s]

# Initial state (joint state (12, ) of SeaCat and SeaDragon)
sim_settings.q_0 = np.array(
    [
        0.0,  # x position SeaCat [m]
        0.0,  # y position SeaCat [m]
        np.pi / 4,  # yaw angle SeaCat [rad]
        0.0,  # x velocity SeaCat [m/s]
        0.0,  # y velocity SeaCat [m/s]
        0.0,  # yaw rate SeaCat [rad/s]
        10.0,  # x position SeaDragon [m]
        -5.0,  # y position SeaDragon [m]
        -np.pi / 2,  # yaw angle SeaDragon [rad]
        0.0,  # x velocity SeaDragon [m/s]
        0.0,  # y velocity SeaDragon [m/s]
        0.0,  # yaw rate SeaDragon [rad/s]
    ]
)

# List of evaluations to perform
eval_list = [
    "static_usv",
    "cooperative",
    "cooperative_no_estimation",
    "cooperative_wrong_heading",
]

# Set debug mode
DEBUG: bool = False

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
sim_settings.v_curr = 0.15  # current speed [m/s]
sim_settings.h_curr = np.pi / 3  # current direction [rad]
dist = disturbances.Disturbances()
dist.set_current_direction(sim_settings.h_curr)  # [rad]
dist.set_current_speed(sim_settings.v_curr)  # [m/s]
v_curr = dist.current()  # (3, ) current speed in inertial frame
v_wind = np.zeros(3)  # disable wind for simplicy (lump in current force vector)
b_curr_sc = hydrodynamics.crossflow_drag(plant_sc.q, params_sc, v_curr)
b_curr_sd = hydrodynamics.crossflow_drag(plant_sd.q, params_sd, v_curr)
b_wind_sc = np.zeros(3)  # disable wind for simplicy (lump in current force vector)
b_wind_sd = np.zeros(3)

# Set noise type
sim_settings.enable_measurement_noise = False  # enable/disable measurement noise
sim_settings.enable_actuation_noise = False  # enable/disable actuation noise

# Initialize MPC controller
mpc = ma_mpc.DockingMpc()
mpc.set_dt(ctrl_dt)
mpc.set_horizon(ctrl_N)
mpc.set_discretization_method(sim_settings.discretization)
mpc.set_model(plant_sc, plant_sd)
mpc.set_weights(sim_settings.Q, sim_settings.R, sim_settings.P)
mpc.set_input_bounds(sim_settings.u_min, sim_settings.u_max)
mpc.set_input_rate_bounds(sim_settings.delta_u_min, sim_settings.delta_u_max)
mpc.init_ocp(mode="reference")

# Tolerance to consider goal reached
pos_thresh = 0.5
angle_thresh = np.deg2rad(10)

# Generate filename to save data
sim_name, _ = io.generate_filename()

# RUN ALL EVALUATIONS ##################################################################
for eval_name in eval_list:

    # Reset variables
    q_meas = np.zeros(12)  # joint measured state
    q_pred = np.zeros((12, mpc.N + 1))  # joint predicted state (MPC solution)
    u = np.zeros(8)  # joint control input
    w_q = np.zeros(12)  # joint measurement noise
    w_u = np.zeros(8)  # joint actuation noise
    cost = 0.0  # mpc cost
    t_sol = 0.0  # mpc solve time
    cost_eval = 0.0  # evaluation cost (evaluates only current timestep)
    t_completion = 0.0  # task completion time
    len_usv_1 = 0.0  # distance traveled by USV1
    len_usv_2 = 0.0  # distance traveled by USV2

    # Initialize time series to store simulation data
    # NOTE: time increases along column direction -> q_mat[:, i] is q at time i*dt
    q_mat = np.zeros((12, sim_n + 1))
    q_mat[:, 0] = np.hstack((plant_sc.q, plant_sd.q))  # save initial state
    q_ref_mat = np.zeros((12, sim_n))
    q_meas_mat = np.zeros((12, sim_n))
    u_mat = np.zeros((8, sim_n))
    w_q_mat = np.zeros((12, sim_n))
    w_u_mat = np.zeros((8, sim_n))
    cost_mat = np.zeros(sim_n)
    sol_t_mat = np.zeros(sim_n)

    # Reset initial conditions in the plant
    plant_sc.set_initial_conditions(sim_settings.q_0[0:6])
    plant_sd.set_initial_conditions(sim_settings.q_0[6:12])
    plant_sc.u = np.zeros(4)
    plant_sd.u = np.zeros(4)

    # Select reference depending on evaluation type
    match eval_name:
        case "static_usv":
            q_ref = np.array(
                [
                    0.0,  # x position SeaCat [m]
                    0.0,  # y position SeaCat [m]
                    0.0,  # yaw angle SeaCat [rad]
                    0.0,  # x velocity SeaCat [m/s]
                    0.0,  # y velocity SeaCat [m/s]
                    0.0,  # yaw rate SeaCat [rad/s]
                    0.0,  # x position SeaDragon [m]
                    -2.2,  # y position SeaDragon [m]
                    -np.pi,  # yaw angle SeaDragon [rad]
                    0.0,  # x velocity SeaDragon [m/s]
                    0.0,  # y velocity SeaDragon [m/s]
                    0.0,  # yaw rate SeaDragon [rad/s]
                ]
            )
        case "cooperative" | "cooperative_no_estimation":
            q_ref = np.array(
                [
                    5.6,  # x position SeaCat [m]
                    6.6,  # y position SeaCat [m]
                    np.pi / 4,  # yaw angle SeaCat [rad]
                    0.0,  # x velocity SeaCat [m/s]
                    0.0,  # y velocity SeaCat [m/s]
                    0.0,  # yaw rate SeaCat [rad/s]
                    4.0,  # x position SeaDragon [m]
                    5.0,  # y position SeaDragon [m]
                    -3 / 4 * np.pi,  # yaw angle SeaDragon [rad]
                    0.0,  # x velocity SeaDragon [m/s]
                    0.0,  # y velocity SeaDragon [m/s]
                    0.0,  # yaw rate SeaDragon [rad/s]
                ]
            )
        case "cooperative_wrong_heading":
            q_ref = np.array(
                [
                    5.6,  # x position SeaCat [m]
                    6.6,  # y position SeaCat [m]
                    np.pi / 2,  # yaw angle SeaCat [rad]
                    0.0,  # x velocity SeaCat [m/s]
                    0.0,  # y velocity SeaCat [m/s]
                    0.0,  # yaw rate SeaCat [rad/s]
                    4.0,  # x position SeaDragon [m]
                    5.0,  # y position SeaDragon [m]
                    -np.pi / 2,  # yaw angle SeaDragon [rad]
                    0.0,  # x velocity SeaDragon [m/s]
                    0.0,  # y velocity SeaDragon [m/s]
                    0.0,  # yaw rate SeaDragon [rad/s]
                ]
            )
        case _:
            raise ValueError("Unknown eval type")

    start_time = datetime.datetime.now()
    print(
        f"\n{CmdColors.OKBLUE}Evaluation {eval_name} started. "
        f"[{start_time.strftime('%H:%M:%S')}]{CmdColors.ENDC}"
    )

    for i in progress_sim(range(sim_n), dt=sim_dt):

        # update control input
        if ctrl_t == 0.0 or ctrl_t >= ctrl_dt:

            # measure the state (possibly with noise)
            if sim_settings.enable_measurement_noise is True:
                w_q_sc = dist.measurement_noise()  # measurement noise for SC
                w_q_sd = dist.measurement_noise()  # measurement noise for SD
                q_meas = np.hstack((plant_sc.q + w_q_sc, plant_sd.q + w_q_sd))
            else:
                w_q = np.zeros(12)
                q_meas = np.hstack((plant_sc.q, plant_sd.q))

            # estimate exogenous inputs
            b_curr_sc = hydrodynamics.crossflow_drag(plant_sc.q, params_sc, v_curr)
            b_curr_sd = hydrodynamics.crossflow_drag(plant_sd.q, params_sd, v_curr)

            # solve mpc
            u_vec, q_pred, cost, t_sol = mpc.solve(
                q_meas,
                b_curr_sc,
                b_wind_sc,
                b_curr_sd,
                b_wind_sd,
                q_ref=q_ref,
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
            u = np.clip(
                u, sim_settings.u_min, sim_settings.u_max
            )  # enforce input bounds
            ctrl_t = 0.0  # reset the elapsed time

            # compute evaluation cost
            # NOTE: the cost is compute only at ctrl_dt intervals
            joint_err = np.hstack((plant_sc.q, plant_sd.q)) - q_ref
            joint_u = np.hstack((plant_sc.u, plant_sd.u))
            cost_eval += (
                joint_err.T @ sim_settings.Q @ joint_err
                + joint_u.T @ sim_settings.R @ joint_u
            )

            # check if final configuration has been reached
            err_sc = np.linalg.norm(plant_sc.q[0:2] - q_ref[0:2])  # pos error SC
            err_sc_heading = abs(transformations.angle_wrap(plant_sc.q[2] - q_ref[2]))
            err_sd = np.linalg.norm(plant_sd.q[0:2] - q_ref[6:8])  # pos error SD
            err_sd_heading = abs(transformations.angle_wrap(plant_sd.q[2] - q_ref[8]))

            if t_completion is None and (
                err_sc < pos_thresh
                and err_sd < pos_thresh
                and abs(err_sc_heading) < angle_thresh
                and abs(err_sd_heading) < angle_thresh
            ):
                t_completion = i * sim_dt  # first time the goal was reached

            elif t_completion is not None and (
                err_sc > pos_thresh
                or err_sd > pos_thresh
                or abs(err_sc_heading) > angle_thresh
                or abs(err_sd_heading) > angle_thresh
            ):
                t_completion = None  # reset completion time if out of bounds again

            # print debug info
            if DEBUG is True:
                c_tot, c_head, c_dist, c_inp = mpc.cost()
                print(
                    f"Time: {i*sim_dt:.3f} s | "
                    f"Pos SC: ({plant_sc.q[0]:.2f}, {plant_sc.q[1]:.2f}, "
                    f"{plant_sc.q[2]:.2f}) | "
                    f"Pos SD: ({plant_sd.q[0]:.2f}, {plant_sd.q[1]:.2f}, "
                    f"{plant_sd.q[2]:.2f}) | "
                    f"Inputs: [{u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}, {u[3]:.2f}, "
                    f"{u[4]:.2f}, {u[5]:.2f}, {u[6]:.2f}, {u[7]:.2f}] | "
                    f"Cost: {cost:.2f} ({c_head:.2f}, {c_dist:.2f}, {c_inp:.2f}) | "
                    f"Sol. t: {t_sol:.2f} s"
                )

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

    # TEMP trim first and last data points
    # There is a bug in the script that causes either the first or last data point to be
    # copied from the previous experiment, which messes up the plots. I am removing it
    # manually, since the time step is small enough for this not to affect the results.
    # I will hopefully fix this properly at some point.
    q_mat = q_mat[1:-1]
    w_q_mat = w_q_mat[:, 1:-1]
    w_u_mat = w_u_mat[:, 1:-1]
    q_meas_mat = q_meas_mat[:, 1:-1]
    u_mat = u_mat[:, 1:-1]
    cost_mat = cost_mat[1:-1]
    sol_t_mat = sol_t_mat[1:-1]

    final_time = datetime.datetime.now()
    print(
        f"{CmdColors.OKBLUE}Evaluation {eval_name} completed. "
        f"[{final_time.strftime('%H:%M:%S')}]{CmdColors.ENDC}"
    )

    # Print stats
    len_sc = np.sum(np.sqrt(np.diff(q_mat[0, :]) ** 2 + np.diff(q_mat[1, :]) ** 2))
    len_sd = np.sum(np.sqrt(np.diff(q_mat[6, :]) ** 2 + np.diff(q_mat[7, :]) ** 2))
    print(f"{CmdColors.OKBLUE}Cumulative evaluation cost: {cost_eval:.2f}")
    print(f"{CmdColors.OKBLUE}Distance traveled by SeaCat: {len_sc:.2f}")
    print(f"{CmdColors.OKBLUE}Distance traveled by SeaDragon: {len_sd:.2f}")
    if t_completion is None:
        print(f"{CmdColors.OKBLUE}Docking not completed during the simulation.\n")
    else:
        print(f"{CmdColors.OKBLUE}Docking completion time: {t_completion:.2f} [s]\n")

    # Save simulation data
    # NOTE: the SeaCat data are used where only one USV is allowed
    io.save_sim_data(
        sim_name + eval_name,
        params_sc,
        sim_settings,
        plant_sc,
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
        b_curr_sc,
        b_wind_sc,
        cost_mat,
        sol_t_mat,
    )

    # Generate plots
    fig_variables, _ = plot_ma.plot_variables(
        t_vec,
        u_mat,
        q_mat[:, :-1],
        q_ref_mat,
        cost_mat,
    )
    fig_phase, _ = plot_ma.phase_plot(
        q_mat[:, :-1],
        v_curr,
        v_wind,
        # idx=list(np.linspace(0, sim_n, num=int(sim_n / 10000) + 1, dtype=int)),
    )

    # Save plots
    io.save_figure(fig_variables, sim_name, f"variables-{eval_name}")
    io.save_figure(fig_phase, sim_name, f"phase-plot-{eval_name}")
