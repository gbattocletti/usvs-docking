"""
Generates the plots from a previously generated .pkl file.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

import usvs_control
from usvs_control.utils import io, transformations
from usvs_control.visualization import plot_ma

# move to the directory of the script
script_dir = Path(__file__).parent
os.chdir(script_dir)

# select data to visualize
filename = "results/experiment-1-cooperative-qref-1.pkl"

# load the simulation data
sys.modules["seacat_dp"] = usvs_control  # module was renamed after saving pkl
with open(filename, "rb") as f:
    data = pickle.load(f)

# unpack the data
q_mat = data["q_mat"]
q_mat = q_mat[:, :-1]
q_ref_mat = data["q_ref_mat"]
q_ref_mat = q_ref_mat[:, :-1]
u_mat = data["u_control_mat"]
v_current = data["v_current"]
v_wind = data["v_wind"]

# generate the plots and animation
fig_phase, _ = plot_ma.phase_plot(q_mat, v_current, v_wind, idx=[0, -1])
io.save_figure(fig_phase, "results/figure", "phase-plot")

# Print evaluation metrics
Q: np.ndarray = np.diag(
    [
        10e3,  # x position SeaCat
        10e3,  # y position SeaCat
        10e3,  # yaw (heading) SeaCat
        10e0,  # x velocity SeaCat
        10e0,  # y velocity SeaCat
        10e0,  # yaw rate SeaCat
        10e3,  # x position SeaDragon
        10e3,  # y position SeaDragon
        10e3,  # yaw (heading) SeaDragon
        10e0,  # x velocity SeaDragon
        10e0,  # y velocity SeaDragon
        10e0,  # yaw rate SeaDragon
    ]
)
R: np.ndarray = np.diag(
    [
        10e-2,  # stern left SeaCat
        10e-2,  # stern right SeaCat
        10e-3,  # bow left SeaCat
        10e-3,  # bow right SeaCat
        10e-2,  # stern left SeaDragon
        10e-2,  # stern right SeaDragon
        10e-4,  # angle left SeaDragon
        10e-4,  # angle right SeaDragon
    ]
)
cost_eval = 0.0
for i in range(q_mat.shape[1] - 1):
    joint_err = q_mat[:, i] - q_ref_mat[:, i]
    joint_err[2] = transformations.angle_wrap(joint_err[2])  # SC yaw error
    joint_err[8] = transformations.angle_wrap(joint_err[8])  # SD yaw error
    joint_u = u_mat[:, i]
    cost_eval += joint_err.T @ Q @ joint_err + joint_u.T @ R @ joint_u
len_sc = np.sum(np.sqrt(np.diff(q_mat[0, :]) ** 2 + np.diff(q_mat[1, :]) ** 2))
len_sd = np.sum(np.sqrt(np.diff(q_mat[6, :]) ** 2 + np.diff(q_mat[7, :]) ** 2))
print(f"Cumulative evaluation cost: {cost_eval:.2f}")
print(f"Distance traveled by SeaCat: {len_sc:.2f}")
print(f"Distance traveled by SeaDragon: {len_sd:.2f}")
