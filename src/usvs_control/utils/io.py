import datetime
import os
import pickle
import tkinter as tk
from pickletools import optimize
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from tqdm import tqdm

from usvs_control.control import linear_mpc, mpc, nonlinear_mpc
from usvs_control.model import disturbances, model_seacat, parameters_seacat
from usvs_control.utils import settings_sa
from usvs_control.visualization.colors import CmdColors


def generate_filename() -> tuple[str, str]:
    """
    Generate a unique filename for the simulation results. The filename points to a
    directory in the results folder, named with the current date and a counter to ensure
    uniqueness. Files will be saved in this directory and their name will start with
    the same date and counter, followed (possibly) by a suffix, and by the file type.

    Returns:
        str: A string representing the unique simulation name based on the current date
        and a counter.
        str: A string representing the relative path to the directory where the
        simulation data will be saved, including the simulation name.
    """
    # Create the results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Generate unique simulation name
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    counter = 0
    while os.path.isdir(f"results\\{date}-{counter:04}"):
        counter += 1
    sim_id = f"{counter:04}"
    sim_name = f"{date}-{sim_id}"
    os.makedirs(f"results\\{sim_name}")

    # Create root file name in the results folder
    path_name = f"results\\{sim_name}\\{sim_name}"

    # Return simulation name
    return sim_name, path_name


def save_sim_data(
    sim_name: str,
    params: parameters_seacat.SeaCatParameters,
    sim_settings: settings_sa.SimSettings,
    model: model_seacat.SeaCatModel,
    controller: linear_mpc.LinearMpc | mpc.Mpc | nonlinear_mpc.NonlinearMpc,
    dist: disturbances.Disturbances,
    t_vec: np.ndarray,
    q_ref_mat: np.ndarray,
    q_mat: np.ndarray,
    q_meas_mat: np.ndarray,
    u_mat: np.ndarray,
    w_q_mat: np.ndarray,
    w_u_mat: np.ndarray,
    v_current: np.ndarray,
    v_wind: np.ndarray,
    b_current: np.ndarray,
    b_wind: np.ndarray,
    cost_mat: np.ndarray,
    t_sol_mat: np.ndarray,
) -> None:
    """
    Run all the save functions that must be executed for every simulation

    Args:
        sim_name (str): Name of the simulation (unique identifier) to use to generate
            the filename to save the simulation data.
        params (parameters.Parameters): parameters object.
        sim_settings (sim_settings.SimSettings): simulation settings object.
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
        v_current (np.ndarray): current velocity. Assumed stationary. (3, )
        v_wind (np.ndarray): wind velocity. Assumed stationary. (3, )
        b_current (np.ndarray): current exogenous input. Assumed stationary. (3, )
        b_wind (np.ndarray): wind exogenous input. Assumed stationary. (3, )
        cost_mat (np.ndarray): cost matrix. (N, )
        t_sol_mat (np.ndarray): solver time vector. (N, )

    Returns:
        None
    """
    # Parse filename (kind of redundant, but ensures consistency)
    if sim_name.startswith("results\\"):
        raise ValueError(
            "Filename should not start with 'results\\'. Only the simulation id should "
            "be provided, i.e., a string formatted as 'YYYY-MM-DD-XXXX'."
        )
    if sim_name.endswith(".pkl"):
        sim_name = sim_name.replace(".pkl", "")
    if sim_name.endswith(".md"):
        sim_name = sim_name.replace(".md", "")
    if sim_name.endswith(".txt"):
        sim_name = sim_name.replace(".txt", "")

    # Create filenames for different formats
    txt_filename = f"results\\{sim_name}\\{sim_name}.txt"
    pkl_filename = f"results\\{sim_name}\\{sim_name}.pkl"

    # Write README file
    with open(txt_filename, "w") as f:
        f.write(f"# Simulation Settings {sim_name}\n")
        f.write(f"## Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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
        if controller.u_rate_max is not None:
            f.write(f"\t delta_u_min: {controller.u_rate_max}\n")
        if controller.u_rate_min is not None:
            f.write(f"\t delta_u_max: {controller.u_rate_min}\n")
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
        "sim_name": sim_name,
        "params": params,
        "sim_settings": sim_settings,
        "t_vec": t_vec,
        "q_ref_mat": q_ref_mat,
        "q_mat": q_mat,
        "q_mat_measured": q_meas_mat,
        "w_q_mat": w_q_mat,
        "u_control_mat": u_mat,
        "w_u_mat": w_u_mat,
        "v_current": v_current,
        "v_wind": v_wind,
        "b_current": b_current,
        "b_wind": b_wind,
        "cost_mat": cost_mat,
        "t_sol_mat": t_sol_mat,
    }

    # Write data to pickle file
    pickled = pickle.dumps(data)  # Dump data dictionary in pickle file
    optimized = optimize(pickled)
    with open(pkl_filename, "wb") as f:
        f.write(optimized)

    print("Simulation data saved.")


def load_sim_data(sim_name: str) -> dict:
    """
    Load simulation data from a pickle file.

    Args:
        sim_name (str): Name of the simulation to load. The filename should be the
            unique identifier of the simulation, which is the same as the one used to
            save the simulation data, having format 'YYYY-MM-DD-XXXX'.

    Raises:
        ValueError: If the filename starts with 'results\\'.
        FileNotFoundError: If the specified file does not exist.

    Returns:
        data (dict): A dictionary containing the loaded simulation data.
    """
    # Parse filename
    if sim_name.startswith("results\\"):
        raise ValueError(
            "Filename should not start with 'results\\'. Only the simulation id should "
            "be provided, i.e., a string formatted as 'YYYY-MM-DD-XXXX'."
        )
    if sim_name.endswith(".pkl"):
        sim_name = sim_name.replace(".pkl", "")

    # Create complete filename
    filename = f"results\\{sim_name}\\{sim_name}.pkl"  # Create filename

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
    filename = filedialog.askopenfilename(
        initialdir="results",
        title="Select .pkl file to visualize",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
    )
    root.destroy()

    # extract filename without path and extension
    filename = os.path.splitext(os.path.basename(filename))[0]
    return os.path.basename(filename)


def save_figure(fig: plt.Figure, sim_name: str, name: str) -> None:
    """
    Save a matplotlib figure to a file.

    Args:
        fig (plt.Figure): plt.Figure object to save.
        sim_name (str): Sim name of the simulation being plotted.
        name (str): Name of the figure, to add to the sim name.
    """
    # Parse sim name
    if sim_name.endswith(".png"):
        sim_name = sim_name.replace(".png", "")

    # Create filename
    filename = f"{sim_name}-{name}.png"

    # Save figure
    print(f"Saving figure {name}...")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print("Figure saving completed.")


def save_animation(anim: FuncAnimation, sim_name: str, **kwargs) -> None:
    """
    Save the animation to a file.

    Args:
        anim (FuncAnimation): The animation object to save.
        sim_name (str): The name of the file to save the animation to.

    Kwargs:
        legacy_progress_bar (bool): If True, use tqdm for progress bar. If False, use
            rich. Default is False.
    """
    # Parse kwargs
    legacy_progress_bar: bool = False
    for key, value in kwargs.items():
        if key == "legacy_progress_bar":
            if not isinstance(value, bool):
                raise TypeError("legacy_progress_bar must be a boolean value.")
            legacy_progress_bar = value
        else:
            print(
                f"{CmdColors.WARNING}[Animate]{CmdColors.ENDC} Unrecognized kwarg "
                f"'{key}' passed to generate_animation()."
            )

    # Parse sim name
    if sim_name.endswith(".gif"):
        sim_name.replace(".gif", "")

    # Create filename
    filename = f"results\\{sim_name}\\{sim_name}-animation.gif"

    # Save animation
    total_frames = anim._save_count  # pylint: disable=protected-access

    # Use rich progress bar
    if legacy_progress_bar:
        with tqdm(total=total_frames, desc="Generating animation") as pbar:
            initial_time = datetime.datetime.now()

            def tqdm_progress_callback(current_frame: int, _total_frames: int) -> None:
                """
                Update the tqdm progress bar during animation saving.

                Args:
                    current_frame (int): The current frame number.
                    _total_frames (int): The total number of frames (unused).
                """
                # Update the progress bar
                pbar.n = current_frame
                pbar.refresh()

            anim.save(
                filename,
                writer="pillow",  # or "ffmpeg" for MP4
                dpi=150,
                progress_callback=tqdm_progress_callback,
            )
    else:
        with Progress(
            TextColumn("[bold blue]{task.description}"),  # custom label
            BarColumn(),  # progress bar
            "[progress.percentage]{task.percentage:>3.0f}%",  # completion percentage
            "{task.fields[iteration]}",  # iteration %
            TimeElapsedColumn(),  # elapsed real time
        ) as progress:
            initial_time = datetime.datetime.now()
            task = progress.add_task(
                f"Saving simulation [{initial_time.strftime('%H:%M:%S')}]",
                total=total_frames,
                iteration=0,
            )

            # Define the rich progress callback
            def rich_progress_callback(current_frame: int, total_frames: int) -> None:
                """
                Callback to update the rich progress bar during animation saving.

                Args:
                    current_frame (int): The current frame number.
                    total_frames (int): The total number of frames.

                """
                progress.update(
                    task, advance=1, iteration=current_frame, total=total_frames
                )

            anim.save(
                filename,
                writer="pillow",  # or "ffmpeg" for MP4
                dpi=150,
                progress_callback=rich_progress_callback,
            )
    print(f"Animation saved as {filename}")
