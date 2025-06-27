import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle

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
    ax[0].legend(["$u_1$", "$u_2$", "$u_3$", "$u_4$"])

    # Plot x, y
    ax[1].plot(t_vec, q_mat[0, :])
    ax[1].plot(t_vec, q_mat[1, :])
    if q_ref_mat is not None:
        ax[1].plot(t_vec, q_ref_mat[0, :], "--", color="k")
        ax[1].plot(t_vec, q_ref_mat[1, :], "--", color="k")
    ax[1].set(xlabel="", ylabel="$x, y$ [m]")
    ax[1].grid()
    ax[1].legend(["$x$", "$y$"])

    # Plot theta
    ax[2].plot(t_vec, q_mat[2, :])
    if q_ref_mat is not None:
        ax[2].plot(t_vec, q_ref_mat[2, :], "--", color="k")
    ax[2].set(xlabel="", ylabel=r"$\psi$ [rad]")
    ax[2].grid()
    ax[2].set_ylim(-np.pi, np.pi)

    # Plot u, v
    ax[3].plot(t_vec, q_mat[3, :])
    ax[3].plot(t_vec, q_mat[4, :])
    ax[3].set(xlabel="", ylabel=r"$u, v$ [m/s]")
    ax[3].grid()
    ax[3].legend(["$u$", "$v$"])

    # Plot omega
    ax[4].plot(t_vec, q_mat[5, :])
    ax[4].set(xlabel="time [s]", ylabel=r"$\omega$ [rad/s]")
    ax[4].grid()

    # Plot cost
    if cost_mat is not None:
        ax[5].plot(t_vec, cost_mat)
        ax[5].set(xlabel="time [s]", ylabel="cost")
        ax[5].grid()

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

    # World reference frame
    origin = np.array([0, 0])
    arrow_inertial_x = np.array([0, 1])
    arrow_inertial_y = np.array([1, 0])
    plt.arrow(*origin, *arrow_inertial_x, head_width=0.05, color="k", linewidth=1.5)
    plt.arrow(*origin, *arrow_inertial_y, head_width=0.05, color="k", linewidth=1.5)

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
        )
        plt.arrow(
            x=robot[0],
            y=robot[1],
            dx=arrow_body_y[0],
            dy=arrow_body_y[1],
            head_width=0.02,
            color="b",
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
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="blue")

    # Plot wind disturbance
    if v_wind is not None and np.linalg.norm(v_wind) > 0:
        v_x = v_wind[1]
        v_y = v_wind[0]

        # Repeat the same arrow (u, v) at every point
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y

        # Plot the vector field
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="green")

    # Trajectory (rotated to match the orientation of the inertial frame)
    x = q_mat[1, :]
    y = q_mat[0, :]
    ax.plot(x, y)

    # Return figure and axes
    return fig, ax


def animation_step(
    frame_idx: int,
    t_vec: np.ndarray,
    q_mat: np.ndarray,
    q_ref_mat: np.ndarray,
    u_mat: np.ndarray,
    arr_body_x: FancyArrow,
    arr_body_y: FancyArrow,
    traj_body: list[Line2D],
    arr_ref_x: FancyArrow,
    arr_ref_y: FancyArrow,
    u_line_0: list[Line2D],
    u_line_1: list[Line2D],
    u_line_2: list[Line2D],
    u_line_3: list[Line2D],
    title: plt.Text,
    speed_up_factor: int,
) -> tuple[
    FancyArrow,
    FancyArrow,
    Line2D,
    FancyArrow,
    FancyArrow,
    Line2D,
    Line2D,
    Line2D,
    Line2D,
    plt.Text,
]:
    """
    Update the animation components for the animation of the phase plot.

    Args:
        frame_idx (int): Frame index.
        t_vec (np.ndarray): Time vector (N, ).
        q_mat (np.ndarray): State variables matrix (6, N).
        u_mat (np.ndarray): Input forces matrix (4, N).
        q_ref_mat (np.ndarray): Reference state variables matrix (6, N).
        arr_body_x (FancyArrow): Arrow object for x-axis.
        arr_body_y (FancyArrow): Arrow object for y-axis.
        traj_body (Line2D): Trajectory plot object.
        arr_ref_x (FancyArrow): Arrow object for x-axis of reference RF.
        arr_ref_y (FancyArrow): Arrow object for y-axis of reference RF.
        u_line_0 (Line2D): Line object for the first input force.
        u_line_1 (Line2D): Line object for the second input force.
        u_line_2 (Line2D): Line object for the third input force.
        u_line_3 (Line2D): Line object for the fourth input force.
        title (plt.Text): Title text object for the plot.
        speed_up_factor (int): Speed-up factor for the animation.

    Returns:
        updated_objects (tuple): Updated arrow objects and trajectory object.
    """
    # Animation plot parameters
    arr_len = 0.7
    arr_inertial_x = arr_len * np.array([0, 1])
    arr_inertial_y = arr_len * np.array([1, 0])

    # Find data index adjusted for speed-up factor
    idx = frame_idx * speed_up_factor

    # Update title
    title.set_text(f"2D Trajectory (t = {t_vec[idx]:.2f} s)")

    # Update body reference frame
    psi_body = q_mat[2, idx]
    R_body = R_i2b(psi_body)[0:2, 0:2]
    x_arrow_body = R_body.dot(arr_inertial_x)
    y_arrow_body = R_body.dot(arr_inertial_y)
    arr_body_x.set_data(
        x=q_mat[1, idx], y=q_mat[0, idx], dx=x_arrow_body[0], dy=x_arrow_body[1]
    )
    arr_body_y.set_data(
        x=q_mat[1, idx], y=q_mat[0, idx], dx=y_arrow_body[0], dy=y_arrow_body[1]
    )

    # Update reference reference frame
    psi_ref = q_ref_mat[2, idx]
    R_ref = R_i2b(psi_ref)[0:2, 0:2]
    x_arrow_ref = R_ref.dot(arr_inertial_x)
    y_arrow_ref = R_ref.dot(arr_inertial_y)
    arr_ref_x.set_data(
        x=q_ref_mat[1, idx],
        y=q_ref_mat[0, idx],
        dx=x_arrow_ref[0],
        dy=x_arrow_ref[1],
    )
    arr_ref_y.set_data(
        x=q_ref_mat[1, idx],
        y=q_ref_mat[0, idx],
        dx=y_arrow_ref[0],
        dy=y_arrow_ref[1],
    )

    # Update robot trajectory
    traj_body.set_data(q_mat[1, 0:idx], q_mat[0, 0:idx])

    # Update input forces
    u_line_0.set_data(t_vec[0:idx], u_mat[0, 0:idx])
    u_line_1.set_data(t_vec[0:idx], u_mat[1, 0:idx])
    u_line_2.set_data(t_vec[0:idx], u_mat[2, 0:idx])
    u_line_3.set_data(t_vec[0:idx], u_mat[3, 0:idx])

    return (
        arr_body_x,
        arr_body_y,
        traj_body,
        arr_ref_x,
        arr_ref_y,
        u_line_0,
        u_line_1,
        u_line_2,
        u_line_3,
    )


def generate_animation(
    t_vec: np.ndarray,
    q_mat: np.ndarray,
    q_ref_mat: np.ndarray,
    u_mat: np.ndarray,
    v_current: np.ndarray,
    v_wind: np.ndarray,
    speed_up_factor: int = 1,
) -> FuncAnimation:
    """
    Animate the trajectory of the robot in the phase space (x-y plane).

    Args:
        t_vec (np.ndarray): Time vector (N, ).
        q_mat (np.ndarray): State variables matrix (6, N).
        q_ref_mat (np.ndarray): Reference state variables matrix (6, N).
        u_mat (np.ndarray): Input forces matrix (4, N).
        v_current (np.ndarray): Current velocity vector (3, ). Assumed stationary.
        v_wind (np.ndarray): Wind velocity vector (3, ). Assumed stationary.
        speed_up_factor (int, optional): Speed-up factor for the animation.

    Returns:
        anim (FuncAnimation): Animation object.
    """

    # Initialize figure
    # Compute bounds:
    x_min = np.min(q_mat[1, :]) - 1
    x_max = np.max(q_mat[1, :]) + 1
    y_min = np.min(q_mat[0, :]) - 1
    y_max = np.max(q_mat[0, :]) + 1

    # Initialize figure
    fig, ax = initialize_phase_plot(x_min, x_max, y_min, y_max)
    ax.grid()
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    title = ax.set_title(f"2D Trajectory (t = {t_vec[0]:.2f} s)", fontsize=10)

    # Determine number of frames to generate
    N = int(np.size(q_mat, 1) / speed_up_factor)

    # Initialize world reference frame (static)
    origin_body = np.array([0, 0])
    arrow_inertial_x = np.array([0, 1])
    arrow_inertial_y = np.array([1, 0])
    plt.arrow(*origin_body, *arrow_inertial_x, head_width=0.03, color="k")
    plt.arrow(*origin_body, *arrow_inertial_y, head_width=0.03, color="k")

    # Create a grid of points for the vector fields
    grid_delta_x = 1.0
    grid_delta_y = 1.0
    grid_x = np.arange(x_min, x_max + grid_delta_x, grid_delta_x)
    grid_y = np.arange(y_min, y_max + grid_delta_y, grid_delta_y)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    # Plot current vector field (static)
    if v_current is not None and np.linalg.norm(v_current) > 0:
        v_x = v_current[1]
        v_y = v_current[0]
        v_x_mat = np.ones_like(grid_X) * v_x
        v_y_mat = np.ones_like(grid_Y) * v_y
        plt.quiver(grid_X, grid_Y, v_x_mat, v_y_mat, color="blue")

    # Plot wind vector field (static)
    if v_wind is not None and np.linalg.norm(v_wind) > 0:
        v_x = v_wind[1]
        v_y = v_wind[0]
        v_x_mat = np.ones_like(grid_X) * v_x
        v_y_mat = np.ones_like(grid_Y) * v_y
        plt.quiver(grid_X, grid_Y, v_x_mat, v_y_mat, color="green")

    # Trajectory (animated)
    x = q_mat[1, 0]
    y = q_mat[0, 0]
    (traj_body,) = ax.plot(x, y)

    # Body reference frame (animated)
    psi_body = q_mat[2, 0]
    origin_body = q_mat[0:2, 0]
    arrow_body_x = 0.7 * arrow_inertial_x
    arrow_body_y = 0.7 * arrow_inertial_y
    R_body = R_i2b(psi_body)[0:2, 0:2]
    arrow_body_x = R_body.dot(arrow_body_x)  # R_body @ arrow_body_x
    arrow_body_y = R_body.dot(arrow_body_y)
    arr_body_x = plt.arrow(
        origin_body[0],
        origin_body[1],
        dx=arrow_body_x[0],
        dy=arrow_body_x[1],
        head_width=0.02,
        color="b",
    )
    arr_body_y = plt.arrow(
        origin_body[0],
        origin_body[1],
        dx=arrow_body_y[0],
        dy=arrow_body_y[1],
        head_width=0.02,
        color="b",
    )

    # Target reference frame (animated)
    psi_ref = q_ref_mat[2, 0]
    origin_ref = q_ref_mat[0:2, 0]
    arrow_ref_x = 0.7 * arrow_inertial_x
    arrow_ref_y = 0.7 * arrow_inertial_y
    R_ref = R_i2b(psi_ref)[0:2, 0:2]
    arrow_ref_x = R_ref.dot(arrow_ref_x)
    arrow_ref_y = R_ref.dot(arrow_ref_y)
    arr_ref_x = plt.arrow(
        origin_ref[0],
        origin_ref[1],
        dx=arrow_ref_x[0],
        dy=arrow_ref_x[1],
        head_width=0.012,
        color="k",
    )
    arr_ref_y = plt.arrow(
        origin_ref[0],
        origin_ref[1],
        dx=arrow_ref_y[0],
        dy=arrow_ref_y[1],
        head_width=0.012,
        color="k",
    )

    # Add actuators subplot
    # Position: [left, bottom, width, height] (w.r.t. main plot)
    ax_u = plt.axes([0.15, 0.65, 0.2, 0.2])  # Top-left corner
    rect = Rectangle((0, -2), 10, 4, alpha=0.2, color="gray")
    ax_u.add_patch(rect)
    ax_u.set_xlabel("t [s]")
    ax_u.set_ylabel("u [N]")
    ax_u.set_xlim(0, np.max(t_vec))
    ax_u.set_ylim(-1000, 1200)  # Thrusters force limits
    ax_u.grid()
    ax_u.set_title("Thrusters forces", fontsize=10)

    # Initialize empty lines for progressive filling
    (u_line_0,) = ax_u.plot([], [], color="darkred", lw=1)
    (u_line_1,) = ax_u.plot([], [], color="red", lw=1)
    (u_line_2,) = ax_u.plot([], [], color="orange", lw=1)
    (u_line_3,) = ax_u.plot([], [], color="gold", lw=1)

    # Add legend to the actuators subplot
    ax_u.legend(
        [
            "$u_1$",
            "$u_2$",
            "$u_3$",
            "$u_4$",
        ],
        loc="upper left",
        fontsize=8,
    )

    # Generate the animation
    anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(
            t_vec,
            q_mat,
            q_ref_mat,
            u_mat,
            arr_body_x,
            arr_body_y,
            traj_body,
            arr_ref_x,
            arr_ref_y,
            u_line_0,
            u_line_1,
            u_line_2,
            u_line_3,
            title,
            speed_up_factor,
        ),
        frames=N,
        interval=50,
        blit=True,
        repeat=False,
    )

    # Save the animation
    return anim
