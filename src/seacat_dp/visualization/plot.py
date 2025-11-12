import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from seacat_dp.utils.transformations import R_i2b

plt.rcParams["text.usetex"] = True

""" 
Useful links:
- Matplotlib RF plot https://thomascountz.com/2018/11/18/2d-coordinate-fromes-matplotlib
"""


def plot_variables(
    t_vec: np.ndarray,
    u_mat: np.ndarray,
    q_mat: np.ndarray,
    q_ref_mat: np.ndarray = None,
    cost_mat: np.ndarray = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot the input forces and the state variables of the 3DoF simulation.

    Args:
        t_vec (numpy.ndarray): Time vector (1, N).
        u_mat (numpy.ndarray): Input forces matrix (4, N).
        q_mat (numpy.ndarray): State variables matrix (6, N).
        q_ref_mat (numpy.ndarray, optional): Reference state variables matrix (6, N).
        cost_mat (numpy.ndarray, optional): Cost matrix (1, N).

    Raises:
        ValueError: If the dimensions of the input matrices do not match the length of
            the time vector.
        UserWarning: If q_mat or q_ref_mat have one extra column, it will be ignored.

    Returns:
        fig (plt.Figure): Figure object.
        ax (Axes): Axes object.
    """
    # Check dimensions
    n = len(t_vec)
    if u_mat.shape[1] != n:
        raise ValueError(f"u_mat should have shape (4, {n})")
    if q_mat.shape[1] != n:
        if q_mat.shape[1] == n + 1:
            warnings.warn(
                "Warning: q_mat has one extra column. Ignoring last column.",
                UserWarning,
            )
            q_mat = q_mat[:, :-1]
        raise ValueError(f"q_mat should have shape (6, {n})")
    if q_ref_mat is not None and q_ref_mat.shape[1] != n:
        if q_ref_mat.shape[1] == n + 1:
            warnings.warn(
                "Warning: q_ref_mat has one extra column. Ignoring last column.",
                UserWarning,
            )
            q_ref_mat = q_ref_mat[:, :-1]
        raise ValueError(f"q_ref_mat should have shape (6, {n})")
    if cost_mat is not None and cost_mat.shape[0] != n:
        raise ValueError(f"cost_mat should have shape (1, {n})")

    # Initialize plot
    if cost_mat is not None:
        fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True)
    else:
        fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)

    # Plot input forces
    ax[0].plot(t_vec, u_mat[0, :])
    ax[0].plot(t_vec, u_mat[1, :])
    ax[0].plot(t_vec, u_mat[2, :])
    ax[0].plot(t_vec, u_mat[3, :])
    ax[0].set(xlabel="", ylabel="$u$ [N]")
    ax[0].grid()
    ax[0].set_axisbelow(True)

    ax[0].legend(["$u_1$", "$u_2$", "$u_3$", "$u_4$"])

    # Plot x, y
    ax[1].plot(t_vec, q_mat[0, :])
    ax[1].plot(t_vec, q_mat[1, :])
    if q_ref_mat is not None:
        ax[1].plot(t_vec, q_ref_mat[0, :], "--", color="k")
        ax[1].plot(t_vec, q_ref_mat[1, :], "--", color="k")
    ax[1].set(xlabel="", ylabel="$x, y$ [m]")
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].legend(["$x$", "$y$"])

    # Plot theta
    ax[2].plot(t_vec, q_mat[2, :])
    if q_ref_mat is not None:
        ax[2].plot(t_vec, q_ref_mat[2, :], "--", color="k")
    ax[2].set(xlabel="", ylabel=r"$\psi$ [rad]")
    ax[2].grid()
    ax[2].set_axisbelow(True)
    ax[2].set_ylim(-np.pi, np.pi)

    # Plot u, v
    ax[3].plot(t_vec, q_mat[3, :])
    ax[3].plot(t_vec, q_mat[4, :])
    ax[3].set(xlabel="", ylabel=r"$u, v$ [m/s]")
    ax[3].grid()
    ax[3].set_axisbelow(True)
    ax[3].legend(["$u$", "$v$"])

    # Plot omega
    ax[4].plot(t_vec, q_mat[5, :])
    ax[4].set(xlabel="time [s]", ylabel=r"$\omega$ [rad/s]")
    ax[4].grid()
    ax[4].set_axisbelow(True)

    # Plot cost
    if cost_mat is not None:
        ax[5].plot(t_vec, cost_mat)
        ax[5].set(xlabel="time [s]", ylabel="cost")
        ax[5].grid()
        ax[5].set_axisbelow(True)

    # Return figure and axes
    return fig, ax


def initialize_phase_plot(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[plt.Figure, Axes]:
    """
    Initialize the phase plot for the 3DoF simulation.
    Args:
        x_min (float): Minimum bound of the x-axis.
        x_max (float): Maximum bound of the x-axis.
        y_min (float): Minimum bound of the y-axis
        y_max (float): Maximum bound of the y-axis.

    Returns:
        fig (plt.Figure): Figure object.
        ax (Axes): Axes object.
    """
    # Initialize figure
    fig, ax = plt.subplots()  # possible option: figsize=(6, 6)
    ax.set_aspect("equal", "box")
    plt.xlim(x_min, x_max)  # Set x-axis range
    plt.ylim(y_min, y_max)  # Set y-axis range
    ax.set(xlabel="y earth [m]", ylabel="x earth [m]")

    # Return figure and axes
    return fig, ax


def phase_plot(
    q_mat: np.ndarray, v_current: np.ndarray, v_wind: np.ndarray, idx: list[int] = None
) -> tuple[plt.Figure, Axes]:
    """
    Plot the trajectory of the robot in the phase space (x-y plane).

    Args:
        q_mat (np.ndarray): state variables matrix (6, N).
        v_current (np.ndarray): current disturbance vector (3, ). Assumed stationary.
        v_wind (np.ndarray): wind speed vector (3, ). Assumed stationary.
        idx (list[int], optional): list of indexes of the state at which to plot the
            system's body reference frame. Defaults to [-1].

    Returns:
        fig (plt.Figure): Figure object.
        ax (Axes): Axes object.
    """
    # Parse input
    if idx is None:
        idx = [-1]
    elif not isinstance(idx, list):
        if isinstance(idx, int):
            idx = [idx]
        raise ValueError("idx must be a list of integers.")
    elif len(idx) == 0:
        idx = [-1]

    # Compute bounds:
    x_min = np.min(q_mat[1, :]) - 1
    x_max = np.max(q_mat[1, :]) + 1
    y_min = np.min(q_mat[0, :]) - 1
    y_max = np.max(q_mat[0, :]) + 1
    # x_min = -1.6
    # x_max = 1.1
    # y_min = -1.1
    # y_max = 1.6

    # Initialize figure
    fig, ax = initialize_phase_plot(x_min, x_max, y_min, y_max)
    ax.grid()
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.set_axisbelow(True)

    # World reference frame
    origin = np.array([0, 0])
    arrow_inertial_x = np.array([0, 1])
    arrow_inertial_y = np.array([1, 0])
    plt.arrow(
        *origin, *arrow_inertial_x, head_width=0.05, color="k", linewidth=1.5, zorder=10
    )
    plt.arrow(
        *origin, *arrow_inertial_y, head_width=0.05, color="k", linewidth=1.5, zorder=10
    )

    # Plot body reference frames
    for i in idx:
        if i >= q_mat.shape[1]:
            if i == q_mat.shape[1]:
                i = -1  # Fix small indexing errors
            else:
                raise ValueError(
                    f"Index {i} is out of bounds for q_mat with shape {q_mat.shape}."
                )
        psi = q_mat[2, i]
        robot = q_mat[0:2, i]
        robot = np.array([robot[1], robot[0]])
        arrow_body_x = 0.7 * arrow_inertial_x
        arrow_body_y = 0.7 * arrow_inertial_y
        rotation = R_i2b(psi)[0:2, 0:2]
        arrow_body_x = rotation.dot(arrow_body_x)
        arrow_body_y = rotation.dot(arrow_body_y)
        plt.arrow(
            x=robot[0],
            y=robot[1],
            dx=arrow_body_x[0],
            dy=arrow_body_x[1],
            head_width=0.02,
            color="b",
            zorder=9,
        )
        plt.arrow(
            x=robot[0],
            y=robot[1],
            dx=arrow_body_y[0],
            dy=arrow_body_y[1],
            head_width=0.02,
            color="b",
            zorder=9,
        )

    # Create a grid of points for the vector fields
    delta_x = 1.0
    delta_y = 1.0
    x = np.arange(x_min, x_max + delta_x, delta_x)
    y = np.arange(y_min, y_max + delta_y, delta_y)
    X, Y = np.meshgrid(x, y)

    # Plot current disturbance
    if v_current is not None and np.linalg.norm(v_current) > 0:
        v_x = v_current[1]
        v_y = v_current[0]

        # Repeat the same arrow (u, v) at every point
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y

        # Plot the vector field
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="blue", width=0.003, zorder=4)

    # Plot wind disturbance
    if v_wind is not None and np.linalg.norm(v_wind) > 0:
        v_x = v_wind[1]
        v_y = v_wind[0]

        # Repeat the same arrow (u, v) at every point
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y

        # Plot the vector field
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="green", width=0.003, zorder=5)

    # Trajectory (rotated to match the orientation of the inertial frame)
    x = q_mat[1, :]
    y = q_mat[0, :]
    ax.plot(x, y, linewidth=2, zorder=8)

    # Return figure and axes
    return fig, ax
