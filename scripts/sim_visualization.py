"""
Generates the plots from a previously generated .pkl file.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from seacat_dp.utils import io
from seacat_dp.visualization import plot_functions

# select data to visualize
# filename = None  # to manually select a file, set filename to None
filename = "2025-06-05-0007.pkl"

# visualization options
SHOW_PLOTS = False
SAVE_PLOTS = True

# move to the directory of the script
script_dir = Path(__file__).parent
os.chdir(script_dir)

# manually select a file
if filename is None:
    filename = io.select_file_interactively()
    if not filename:  # User canceled the dialog
        print("No file selected. Exiting.")
        sys.exit(0)
elif not filename.endswith(".pkl"):
    filename += ".pkl"

# load the simulation data
data = io.load_sim_data(filename)

# unpack the data
params = data["params"]
dist = data["dist"]
t_vec = data["t_vec"]
q_ref_mat = data["q_ref_mat"]
q_mat = data["q_mat"]
q_mat = q_mat[:, :-1]  # remove the last column (future state)
q_mat_measured = data["q_mat_measured"]
w_mat = data["w_mat"]
u_mat = data["u_control_mat"]
b_current = data["b_current"]
b_wind = data["b_wind"]

# plot parameters
anim_speed_up_factor = 200  # speed up factor for the animation

# plot the simulation data
# plot_functions.plot_variables(t_vec, u_mat, q_mat)
plot_functions.phase_plot(q_mat, b_current, b_wind)
plot_functions.animate(
    "results/animation.gif",
    t_vec,
    q_mat,
    q_ref_mat,
    u_mat,
    b_current,
    b_wind,
    anim_speed_up_factor,
    SAVE_PLOTS,  # save the animation
    SHOW_PLOTS,  # show the animation
)

# show plots
if SHOW_PLOTS:
    plt.show()
