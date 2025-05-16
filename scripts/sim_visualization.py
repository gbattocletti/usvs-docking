"""
Generates the plots from a previously generated .pkl file.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

from seacat_dp.utils import io
from seacat_dp.visualization import plot_functions

# select data to visualize
filename = "2025-05-16-0000.pkl"
SHOW_PLOTS = True
SAVE_PLOTS = False

# load the simulation data
script_dir = Path(__file__).parent
os.chdir(script_dir)
data = io.load_sim_data(filename)

# unpack the data
params = data["params"]
dist = data["dist"]
t_vec = data["t_vec"]
q_mat = data["q_mat"]
q_mat = q_mat[:, :-1]  # remove the last column (future state)
q_mat_measured = data["q_mat_measured"]
w_mat = data["w_mat"]
u_mat = data["u_control_mat"]
b_current_mat = data["b_current_mat"]
b_wind_mat = data["b_wind_mat"]

# plot the simulation data
plot_functions.plot_variables(t_vec, u_mat, q_mat)
plot_functions.phase_plot(q_mat)
plt.show()

# show plots
if SHOW_PLOTS:
    pass

# save plots
if SAVE_PLOTS:
    pass  # TODO
