import os
import pickle
from datetime import datetime
from pickletools import optimize

import numpy as np

from seacat_dp.model import disturbances, parameters


def save_sim_data(
    params: parameters.Parameters,
    dist: disturbances.Disturbances,
    t_vec: np.ndarray,
    q_mat: np.ndarray,
    w_mat: np.ndarray,
    q_mat_measured: np.ndarray,
    u_mat: np.ndarray,
    b_current_mat: np.ndarray,
    b_wind_mat: np.ndarray,
) -> None:
    """
    Run all the save functions that must be executed for every simulation

    Args:
        params (parameters.Parameters): parameters object
        dist (disturbances.Disturbances): disturbances object
        t_mat (np.ndarray): time vector
        q_mat (np.ndarray): 'real' plant output (q)
        w_mat (np.ndarray): noise (w)
        q_mat_measured (np.ndarray): plant output with noise (q + w)
        u_mat (np.ndarray): control input (u)
        b_current_mat (np.ndarray): current exogenous input (b_current))
        b_wind_mat (np.ndarray): wind exogenous input (b_wind)
    Returns:
        None
    """
    # Create the results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Generate unique simulation name
    date = datetime.now().strftime("%Y-%m-%d")
    counter = 0
    while os.path.exists(f"results\\{date}-{counter:04}.pkl"):
        counter += 1
    sim_id = f"{counter:04}"
    sim_name = f"{date}-{sim_id}"

    # Save the simulation data
    data = {
        "params": params,
        "dist": dist,
        "t_vec": t_vec,
        "q_mat": q_mat,
        "q_mat_measured": q_mat_measured,
        "w_mat": w_mat,
        "u_control_mat": u_mat,
        "b_current_mat": b_current_mat,
        "b_wind_mat": b_wind_mat,
    }  # Store data in a dictionary

    # Write data to pickle file
    filename: str = f"results\\{sim_name}.pkl"
    pickled = pickle.dumps(data)  # Dump data dictionary in pickle file
    optimized = optimize(pickled)
    with open(filename, "wb") as f:
        f.write(optimized)


def load_sim_data(filename: str):

    # Create complete filename
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"  # Add .pkl extension if not present
    filename = f"results\\{filename}"  # Add results folder path to filename

    # Load data
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise FileNotFoundError(filename)
