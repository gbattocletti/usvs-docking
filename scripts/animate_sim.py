"""
Generates the docking animation from a previously generated .pkl file.
"""

import os
import pickle
import sys
from pathlib import Path

import usvs_control
from usvs_control.utils import io
from usvs_control.visualization import animate_ma

# move to the directory of the script
script_dir = Path(__file__).parent
os.chdir(script_dir)

# select data to visualize
experiment_name = "experiment-2-cooperative-qref-2"
filename = f"results/{experiment_name}.pkl"

# load the simulation data
sys.modules["seacat_dp"] = usvs_control  # module was renamed after saving pkl
with open(filename, "rb") as f:
    data = pickle.load(f)

# unpack the data
q_mat = data["q_mat"]
q_mat = q_mat[:, :-1]
v_current = data["v_current"]

# generate the plots and animation
anim = animate_ma.generate_animation(
    q_mat=q_mat,
    v_current=v_current,
    speed_up_factor=1,
)
io.save_animation(anim, f"{experiment_name}")
