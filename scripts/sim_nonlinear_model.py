# import matplotlib.pyplot as plt
import numpy as np

from seacat_dp.models import nonlinear_model, parameters

# Simulation parameters
sim_t_end = 10.0  # simulation time [s]
sim_dt = 0.01  # time step [s]
sim_n = int(sim_t_end / sim_dt)  # number of time steps
t = sim_dt * np.arange(sim_n)  # time vector [s]

# Initialize the model
params = parameters.Parameters()
model_nl = nonlinear_model.NonlinearModel(params, sim_dt)

# Initialize time series
q = np.zeros((6, sim_n + 1))  # state time series

# Run the simulation
for i in range(sim_n):

    # model inputs
    u = np.zeros((4, 1))  # control input [N]
    b = np.zeros((3, 1))  # measured disturbance [N] and [Nm]
    w = np.zeros((6, 1))  # unmeasured disturbance

    # model state
    q[i + 1, :] = model_nl(u, b, w)  # update the model state
