import os
import pickle
import tkinter as tk
from datetime import datetime
from pickletools import optimize
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

# import tqdm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from seacat_dp.control import linear_mpc, mpc, nonlinear_mpc
from seacat_dp.model import disturbances, nonlinear_model


def generate_filename() -> str:
    """
    Generate a unique filename for the simulation results.

    Returns:
        str: A string representing the unique simulation name based on the current date
        and a counter.
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

    # Return simulation name
    return sim_name


def save_sim_data(
    filename: str,
    model: nonlinear_model.NonlinearModel,
    controller: linear_mpc.LinearMpc | mpc.Mpc | nonlinear_mpc.NonlinearMpc,
    dist: disturbances.Disturbances,
    t_vec: np.ndarray,
    q_ref_mat: np.ndarray,
    q_mat: np.ndarray,
    q_meas_mat: np.ndarray,
    u_mat: np.ndarray,
    w_q_mat: np.ndarray,
    w_u_mat: np.ndarray,
    b_current: np.ndarray,
    b_wind: np.ndarray,
) -> None:
    """
    Run all the save functions that must be executed for every simulation

    Args:
        filename (str): Name of the simulation (unique identifier) to use to generate
            the filename to save the simulation data.
        model (nonlinear_model.NonlinearModel): plant model object.
        controller (linear_mpc.LinearMPC or mpc.MPC or nonlinear_mpc.NonlinearMPC):
            controller object.
        dist (disturbances.Disturbances): disturbances object.
        t_vec (np.ndarray): time vector. (N, )
        q_ref_mat (np.ndarray): reference state (q_ref). (6, N)
        q_mat (np.ndarray): 'real' plant output (q). (6, N)
        q_meas_mat (np.ndarray): measured plant state (q + w). (6, N)
        u_mat (np.ndarray): control input (u). (4, N)
        w_q_mat (np.ndarray): measurement noise (w). (6, N)
        w_u_mat (np.ndarray): actuation noise (w_u). (4, N)
        b_current (np.ndarray): current exogenous input. Assumed stationary. (3, )
        b_wind (np.ndarray): wind exogenous input. Assumed stationary. (3, )

    Returns:
        None
    """
    # Parse filename
    if not filename.startswith("results\\"):
        filename = f"results\\{filename}"
    if filename.endswith(".pkl"):
        filename = filename.replace(".pkl", "")
    if filename.endswith(".md"):
        filename = filename.replace(".md", "")
    if filename.endswith(".txt"):
        filename = filename.replace(".txt", "")

    # Create filenames for different formats
    txt_filename = f"{filename}.txt"
    pkl_filename = f"{filename}.pkl"
    sim_name = filename.split("\\")[-1]

    # Write README file
    with open(txt_filename, "w") as f:
        f.write(f"# Simulation Settings {sim_name}\n")
        f.write(f"## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write("\n## Parameters:\n")
        f.write(f"\t Simulation time: {t_vec[-1]:.2f} s\n")
        f.write(f"\t Simulation time step: {t_vec[1] - t_vec[0]:.4f} s\n")  # model.dt
        f.write(f"\t Number of time steps: {len(t_vec)}\n")
        f.write(f"\t Integration method: {model.integration_method}\n")

        f.write("\n## Controller:\n")
        f.write(f"\t Controller type: {type(controller).__name__}\n")
        f.write(f"\t Control time step: {controller.dt:.4f}\n")
        f.write(f"\t Control horizon: {controller.N}\n")
        f.write(f"\t Discretization method: {controller.discretization_method} \n")
        # f.write(f"\t Linearization method: {controller.linearization_method} \n")
        f.write(f"\t Q matrix: np.diag([{np.diag(controller.Q)}])\n")
        f.write(f"\t R matrix: np.diag([{np.diag(controller.R)}])\n")
        f.write(f"\t P matrix: np.diag([{np.diag(controller.P)}])\n")
        f.write(f"\t u_min: {controller.u_min}\n")
        f.write(f"\t u_max: {controller.u_max}\n")
        if controller.delta_u_min is not None:
            f.write(f"\t delta_u_min: {controller.delta_u_min}\n")
        if controller.delta_u_max is not None:
            f.write(f"\t delta_u_max: {controller.delta_u_max}\n")
        if controller.q_min is not None:
            f.write(f"\t q_min: {controller.q_min}\n")
        if controller.q_max is not None:
            f.write(f"\t q_max: {controller.q_max}\n")

        f.write("\n## Disturbances:\n")
        f.write(f"\t Current: {dist.current()} [m/s]\n")
        f.write(f"\t Wind: {dist.wind()} [m/s]\n")

        f.write("\n## Initial conditions and goal:\n")
        f.write(f"\t Initial state: {q_mat[:, 0]}\n")
        f.write(f"\t Goal state: {q_ref_mat[:, 0]} (initial goal)\n")

    # Collect the timeseries data in a dictionary to be saved in a pickle file
    data = {
        "t_vec": t_vec,
        "q_ref_mat": q_ref_mat,
        "q_mat": q_mat,
        "q_mat_measured": q_meas_mat,
        "w_q_mat": w_q_mat,
        "u_control_mat": u_mat,
        "w_u_mat": w_u_mat,
        "b_current": b_current,
        "b_wind": b_wind,
    }

    # Write data to pickle file
    pickled = pickle.dumps(data)  # Dump data dictionary in pickle file
    optimized = optimize(pickled)
    with open(pkl_filename, "wb") as f:
        f.write(optimized)

    print("Simulation data saved.")


def load_sim_data(filename: str) -> dict:
    """
    Load simulation data from a pickle file.

    Args:
        filename (str): Name of the file to load.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    Returns:
        data (dict): A dictionary containing the loaded simulation data.
    """

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


def select_file_interactively() -> str:
    """
    Open a file dialog to select a .pkl file.

    Returns:
        str: The selected file name.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        initialdir="results",
        title="Select .pkl file to visualize",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
    )
    root.destroy()
    return os.path.basename(file_path)


def save_figure(fig: plt.Figure, filename: str, name: str) -> None:
    """
    Save a matplotlib figure to a file.

    Args:
        fig (plt.Figure): plt.Figure object to save.
        filename (str): Filename to save the figure to.
        name (str): Name of the figure, to add to the filename.
    """
    # Parse filename
    if not filename.startswith("results\\"):
        filename = f"results\\{filename}"
    if not filename.endswith(".png"):
        filename = f"{filename}.png"

    # Add name to filename
    filename = filename.replace(".png", f"-{name}.png")

    # Save figure
    print(f"Saving figure {name}...")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print("Figure saving completed.")


def save_animation(anim: FuncAnimation, filename: str) -> None:
    """
    Save the animation to a file.

    Args:
        anim (FuncAnimation): The animation object to save.
        filename (str): The name of the file to save the animation to.
    """
    # Parse filename
    if not filename.startswith("results\\"):
        filename = f"results\\{filename}"
    if not filename.endswith(".gif"):
        filename += ".gif"

    # Save animation
    total_frames = anim._save_count  # pylint: disable=protected-access
    with tqdm(total=total_frames, desc="Generating animation") as pbar:
        anim.save(
            filename,
            writer="pillow",  # or "ffmpeg" for MP4
            dpi=150,
            progress_callback=lambda i, bar: progress_callback(i, pbar),
        )
    print(f"Animation saved as {filename}")


def progress_callback(current_frame: int, pbar: tqdm) -> None:
    """
    Update the progress bar during animation saving.

    Args:
        current_frame (int): The current frame number.
        pbar (tqdm): The tqdm progress bar instance to update.
    """
    # Update the progress bar
    pbar.n = current_frame
    pbar.refresh()
