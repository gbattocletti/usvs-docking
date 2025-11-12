"""
Generates the plots from a previously generated .pkl file.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from seacat_dp.utils import io
from seacat_dp.visualization import animate, plot

# select data to visualize
filename = None  # manually select a file
# filename = "2025-06-05-0007.pkl"

# visualization options
SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_ANIM = False

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
t_vec = data["t_vec"]
q_ref_mat = data["q_ref_mat"]
q_mat = data["q_mat"]
q_mat = q_mat[:, :-1]  # remove the last column (future state)
q_mat_measured = data["q_mat_measured"]
w_q_mat = data["w_q_mat"]
u_mat = data["u_control_mat"]
w_u_mat = data["w_q_mat"]
v_current = data["v_current"]
v_wind = data["v_wind"]
b_current = data["b_current"]
b_wind = data["b_wind"]
cost_mat = data["cost_mat"]

# plot parameters
anim_speed_up_factor = 200  # speed up factor for the animation

# generate the plots and animation
if not SHOW_PLOTS and not SAVE_PLOTS and not SAVE_ANIM:
    print("No plots or animations will be generated. Exiting.")
    sys.exit(0)

fig_variables, _ = plot.plot_variables(t_vec, u_mat, q_mat, q_ref_mat, cost_mat)
# plt.gcf().set_size_inches(6, 6)
fig_variables.set_figheight(4)
fig_variables.set_figwidth(4)
# plt.tight_layout()

fig_phase, _ = plot.phase_plot(q_mat, v_current, v_wind)
# plt.gcf().set_size_inches(6, 6)
fig_phase.set_figheight(4)
fig_phase.set_figwidth(4)
# plt.tight_layout()

anim = animate.generate_animation(
    t_vec,
    q_mat,
    q_ref_mat,
    u_mat,
    v_current,
    v_wind,
    anim_speed_up_factor,
)

if SAVE_PLOTS:
    plt.tight_layout()

    io.save_figure(fig_variables, filename, "variables")
    io.save_figure(fig_phase, filename, "phase-plot")

if SAVE_ANIM:
    io.save_animation(anim, filename)

if SHOW_PLOTS:
    plt.show()
