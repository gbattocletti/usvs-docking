import matplotlib.pyplot as plt
import numpy as np

from seacat_dp.model import nonlinear_model, parameters

# Initialize model
param = parameters.Parameters()
model = nonlinear_model.NonlinearModel(param)
model.thrust_delay = np.array([3, 3, 2, 2])

# Simulation parameters
t = 0.0  # initial time [s]
t_end = 60.0  # simulation duration [s]
dt = 0.001  # time step [s]
n = int(t_end / dt)  # number of time steps
t_vec = dt * np.arange(n)  # time vector [s]
u = np.zeros(4)  # control input [N]
f = np.zeros(4)  # thrusters force [N]
u_mat = np.zeros((4, n))  # control input matrix [N]
f_mat = np.zeros((4, n))  # thrusters force matrix [N]

# Run thrusters response simulation
for i in range(n):

    # custom input signal (square wave)
    if t > 10.0:
        u = np.ones(4) * 1000.0

    if t > 20.0:
        u = np.zeros(4)

    if t > 30.0:
        u = np.ones(4) * 1000.0

    # rk4 step
    k1 = model.thruster_dynamics(f, u)
    k2 = model.thruster_dynamics(f + k1 * dt / 2, u)
    k3 = model.thruster_dynamics(f + k2 * dt / 2, u)
    k4 = model.thruster_dynamics(f + k3 * dt, u)
    f = f + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    # store data
    u_mat[:, i] = u
    f_mat[:, i] = f

    # update time
    t += dt

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_vec, u_mat.T)
plt.title("Control Input")
plt.xlabel("Time [s]")
plt.ylabel("Control Input [N]")
plt.grid()
plt.legend(["u1", "u2", "u3", "u4"])
plt.subplot(2, 1, 2)
plt.plot(t_vec, f_mat.T)
plt.title("Thrusters Force")
plt.xlabel("Time [s]")
plt.ylabel("Thrusters Force [N]")
plt.grid()
plt.legend(["f1", "f2", "f3", "f4"])
plt.tight_layout()

plt.show()
