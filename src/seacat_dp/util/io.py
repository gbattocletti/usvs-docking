import os
import pickle
from datetime import datetime
from pickletools import optimize

import numpy as np


def save_sim_data(
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
        "q_mat": q_mat,
        "q_mat_measured": q_mat_measured,
        "w_mat": w_mat,
        "u_control_mat": u_mat,
        "b_current_mat": b_current_mat,
        "b_wind_mat": b_wind_mat,
    }  # Store data in a dictionary
    filename: str = f"results\\{sim_name}.pkl"
    pickled = pickle.dumps(data)  # Dump data dictionary in pickle file
    optimized = optimize(pickled)
    with open(filename, "wb") as f:
        f.write(optimized)


def load_sim_data(filename: str):
    # Create complete filename
    filename = f"results\\{filename}.pkl"

    # Load data
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise FileNotFoundError(filename)
