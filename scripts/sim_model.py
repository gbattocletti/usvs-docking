"""
Script to simulate the USV dynamic model (SeaCat or SeaDragon) with disturbances. The
controller is disabled and a predefined constant control input is applied. The
simulation results are plotted at the end of the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

from usvs_control.model import (
    disturbances,
    model_seacat,
    model_seadragon,
    parameters_seacat,
    parameters_seadragon,
)
from usvs_control.utils.wrappers import progress_sim
from usvs_control.visualization import plot

### SIMULATION SETTINGS - USER DEFINABLE ###############################################

T_END = 60.0  # simulation duration [s]
USV_TYPE = "seadragon"  # USV model type: {"seacat", "seadragon"}

# Initial state
Q0 = np.array(
    [
        0,  # surge position [m]
        0,  # sway position [m]
        0,  # yaw position [rad]
        0,  # surge velocity [m/s]
        0,  # sway velocity [m/s]
        0,  # yaw velocity [rad/s]
    ]
)

# Disturbances
CURRENT_ANGLE = -3 / 4 * np.pi  # current direction [rad]
CURRENT_SPEED = 0.3  # current speed [m/s]  (max speed is 1.0 m/s)

# Control input:
# TODO: extend to time-varying inputs
match USV_TYPE:
    case "seacat":
        U = np.array(
            [
                100,  # stern left [N]
                100,  # stern right [N]
                0,  # bow left [N]
                0,  # bow right [N]
            ]
        )
    case "seadragon":
        U = np.array(
            [
                100,  # left actuator [N]
                100,  # right actuator [N]
                0,  # angle left [rad]
                0,  # angle right [rad]
            ]
        )

########################################################################################

# Simulation parameters
t = 0.0  # time [s]
t_end = T_END  # simulation duration [s]
sim_dt = 0.001  # simulation time step [s]
sim_n = int(t_end / sim_dt)  # number of time steps
t_vec = sim_dt * np.arange(sim_n)  # time vector [s]

# Initialize the model
match USV_TYPE:
    case "seacat":
        params = parameters_seacat.SeaCatParameters()
        model = model_seacat.SeaCatModel(params)
    case "seadragon":
        params = parameters_seadragon.SeaDragonParameters()
        model = model_seadragon.SeaDragonModel(params)
model.set_time_step(sim_dt)  # set the time step for the model
model.set_integration_method("euler")  # set the integration method for the model
model.set_initial_conditions(Q0)  # set the initial conditions for USV model

# Initialize disturbances
dist = disturbances.Disturbances()
dist.set_current_direction(CURRENT_ANGLE)  # set the current direction [rad]
dist.set_current_speed(CURRENT_SPEED)  # set the current speed [m/s]
dist.set_wind_direction(0)  # set the wind direction [rad]
dist.set_wind_speed(0)  # set the wind speed [m/s]
v_current = dist.current()  # current exogenous input (stationary, measured)
v_wind = dist.wind()  # wind exogenous input (stationary, measured)

# Initialize variables
q = np.zeros(6)  # state
w = np.zeros(6)  # disturbance
q_meas = np.zeros(6)  # measured state
u = U  # control input

# Initialize time series
q_mat = np.zeros((6, sim_n + 1))  # state time series
w_mat = np.zeros((6, sim_n))  # disturbance time series
q_meas_mat = np.zeros((6, sim_n))  # state time series
u_mat = np.zeros((4, sim_n))  # control input time series

# Run the simulation
for i in progress_sim(range(sim_n), dt=sim_dt):

    # plant
    q = model(u, v_current, v_wind)  # update the model state

    # store step data
    q_mat[:, i + 1] = q
    w_mat[:, i] = w  # noise
    q_meas_mat[:, i] = q_meas  # measured state
    u_mat[:, i] = u  # control input

    # update time
    t += sim_dt

# Plot the simulation data
plot.plot_variables(t_vec, u_mat, q_mat[:, :-1])
plot.phase_plot(q_mat[:, :-1], v_current, v_wind)
plt.show()
