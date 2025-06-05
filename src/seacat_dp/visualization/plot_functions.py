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
) -> None:
    """
    Plot the input forces and the state variables of the 3DoF simulation.

    Args:
        t_vec (numpy.ndarray): Time vector (1, N).
        u_mat (numpy.ndarray): Input forces matrix (4, N).
        q_mat (numpy.ndarray): State variables matrix (6, N).
        q_ref_mat (numpy.ndarray, optional): Reference state variables matrix (6, N).
        cost_mat (numpy.ndarray, optional): Cost matrix (1, N).
    """
    # Check dimensions
    n = len(t_vec)
    if u_mat.shape[1] != n:
        raise ValueError(f"u_mat should have shape (4, {n})")
    if q_mat.shape[1] != n:
        if q_mat.shape[1] == n + 1:
            print("Warning: q_mat has one extra column. Ignoring last column.")
            q_mat = q_mat[:, :-1]
        raise ValueError(f"q_mat should have shape (6, {n})")
    if q_ref_mat is not None and q_ref_mat.shape[1] != n:
        if q_ref_mat.shape[1] == n + 1:
            print("Warning: q_ref_mat has one extra column. Ignoring last column.")
            q_ref_mat = q_ref_mat[:, :-1]
        raise ValueError(f"q_ref_mat should have shape (6, {n})")
    if cost_mat is not None and cost_mat.shape[0] != n:
        raise ValueError(f"cost_mat should have shape (1, {n})")

    # Initialize plot
    if cost_mat is not None:
        fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True)
    else:
        fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
    fig.suptitle("3DoF simulation")

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
    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    plt.xlim(x_min, x_max)  # Set x-axis range
    plt.ylim(y_min, y_max)  # Set y-axis range
    ax.set(xlabel="y earth [m]", ylabel="x earth [m]")

    return fig, ax


def phase_plot(
    q_mat: np.ndarray, v_current: np.ndarray, v_wind: np.ndarray, idx: list[int] = None
) -> None:
    """
    Plot the trajectory of the robot in the phase space (x-y plane).

    Args:
        q_mat (np.ndarray): state variables matrix (6, N).
        v_current (np.ndarray): current disturbance vector (3, ). Assumed stationary.
        v_wind (np.ndarray): wind speed vector (3, ). Assumed stationary.
        idx (list[int], optional): list of indexes of the state at which to plot the
            system's body reference frame. Defaults to [-1].
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

    # Initialize figure
    _, ax = initialize_phase_plot(x_min, x_max, y_min, y_max)

    # World reference frame
    origin = np.array([0, 0])
    x_arrow_fixed = np.array([1, 0])
    y_arrow_fixed = np.array([0, 1])
    plt.arrow(*origin, *x_arrow_fixed, head_width=0.03, color="k")
    plt.arrow(*origin, *y_arrow_fixed, head_width=0.03, color="k")

    # Plot body reference frames
    for i in idx:
        if i >= q_mat.shape[1]:
            raise ValueError(
                f"Index {i} is out of bounds for q_mat with shape {q_mat.shape}."
            )
        psi = q_mat[2, i]
        robot = q_mat[0:2, i]
        robot = np.array([robot[1], robot[0]])  # Rotate to match the inertial frame
        x_arrow_body = 0.7 * x_arrow_fixed
        y_arrow_body = 0.7 * y_arrow_fixed
        rotation = R_i2b(psi)[0:2, 0:2]
        x_arrow_body = rotation.dot(x_arrow_body)
        y_arrow_body = rotation.dot(y_arrow_body)
        plt.arrow(
            x=robot[0],
            y=robot[1],
            dx=x_arrow_body[0],
            dy=x_arrow_body[1],
            head_width=0.02,
            color="b",
        )
        plt.arrow(
            x=robot[0],
            y=robot[1],
            dx=y_arrow_body[0],
            dy=y_arrow_body[1],
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
    if v_current is not None:
        v_x = v_current[1] / 10
        v_y = v_current[0] / 10

        # Repeat the same arrow (u, v) at every point
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y

        # Plot the vector field
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="blue")

    # Plot wind disturbance
    if v_wind is not None:
        v_x = v_wind[1] / 10
        v_y = v_wind[0] / 10

        # Repeat the same arrow (u, v) at every point
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y

        # Plot the vector field
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="green")

    # Trajectory (rotated to match the orientation of the inertial frame)
    x = q_mat[1, :]
    y = q_mat[0, :]
    ax.plot(x, y)


def animation_step(
    frame_idx: int,
    t_vec: np.ndarray,
    q_mat: np.ndarray,
    q_ref_mat: np.ndarray,
    u_mat: np.ndarray,
    arr_x: FancyArrow,
    arr_y: FancyArrow,
    traj: list[Line2D],
    ref_arr_x: FancyArrow,
    ref_arr_y: FancyArrow,
    u_line_0: list[Line2D],
    u_line_1: list[Line2D],
    u_line_2: list[Line2D],
    u_line_3: list[Line2D],
    speed_up_factor: int,
) -> tuple[
    FancyArrow,
    FancyArrow,
    list[Line2D],
    FancyArrow,
    FancyArrow,
    list[Line2D],
    list[Line2D],
    list[Line2D],
    list[Line2D],
]:
    """
    Update the animation components for the animation of the phase plot.

    Args:
        frame_idx (int): Frame index.
        t_vec (np.ndarray): Time vector (N, ).
        q_mat (np.ndarray): State variables matrix (6, N).
        u_mat (np.ndarray): Input forces matrix (4, N).
        q_ref_mat (np.ndarray): Reference state variables matrix (6, N).
        arr_x (FancyArrow): Arrow object for x-axis.
        arr_y (FancyArrow): Arrow object for y-axis.
        traj (list[Line2D]): Trajectory plot object.
        arr_ref_x (FancyArrow): Arrow object for x-axis of reference RF.
        arr_ref_y (FancyArrow): Arrow object for y-axis of reference RF.
        u_line_0 (list[Line2D]): Line object for the first input force.
        u_line_1 (list[Line2D]): Line object for the second input force.
        u_line_2 (list[Line2D]): Line object for the third input force.
        u_line_3 (list[Line2D]): Line object for the fourth input force.
        speed_up_factor (int): Speed-up factor for the animation.

    Returns:
        (tuple): Updated arrow objects and trajectory object.
    """
    # Animation plot parameters
    arr_len = 0.7

    # Find data index adjusted for speed-up factor
    idx = frame_idx * speed_up_factor

    # Update body reference frame
    psi_body = q_mat[5, idx]
    R_body = R_i2b(psi_body)[0:2, 0:2]
    x_arrow_body = R_body.dot(arr_len * np.array([1, 0]))
    y_arrow_body = R_body.dot(arr_len * np.array([0, 1]))
    arr_x.set_data(
        x=q_mat[0, idx], y=q_mat[1, idx], dx=x_arrow_body[0], dy=x_arrow_body[1]
    )
    arr_y.set_data(
        x=q_mat[0, idx], y=q_mat[1, idx], dx=y_arrow_body[0], dy=y_arrow_body[1]
    )

    # Update reference reference frame
    psi_ref = q_ref_mat[5, idx]
    R_ref = R_i2b(psi_ref)[0:2, 0:2]
    x_arrow_ref = R_ref.dot(arr_len * np.array([1, 0]))
    y_arrow_ref = R_ref.dot(arr_len * np.array([0, 1]))
    ref_arr_x.set_data(
        x=q_ref_mat[0, idx],
        y=q_ref_mat[1, idx],
        dx=x_arrow_ref[0],
        dy=x_arrow_ref[1],
    )
    ref_arr_y.set_data(
        x=q_ref_mat[0, idx],
        y=q_ref_mat[1, idx],
        dx=y_arrow_ref[0],
        dy=y_arrow_ref[1],
    )

    # Update robot trajectory
    traj[0].set_data(q_mat[0, 0:idx], q_mat[1, 0:idx])

    # Update input forces
    u_line_0[0].set_data(t_vec[0:frame_idx], u_mat[0, 0:frame_idx])
    u_line_1[0].set_data(t_vec[0:frame_idx], u_mat[1, 0:frame_idx])
    u_line_2[0].set_data(t_vec[0:frame_idx], u_mat[2, 0:frame_idx])
    u_line_3[0].set_data(t_vec[0:frame_idx], u_mat[3, 0:frame_idx])

    return (
        arr_x,
        arr_y,
        traj[0],
        ref_arr_x,
        ref_arr_y,
        u_line_0[0],
        u_line_1[0],
        u_line_2[0],
        u_line_3[0],
    )


def animate(
    filename: str,
    t_vec: np.ndarray,
    q_mat: np.ndarray,
    q_ref_mat: np.ndarray,
    u_mat: np.ndarray,
    v_current: np.ndarray,
    v_wind: np.ndarray,
    speed_up_factor: int = 1,
    save: bool = False,
    show: bool = False,
) -> None:
    """
    Animate the trajectory of the robot in the phase space (x-y plane).

    Args:
        filename (str): Filename to save the animation.
        t_vec (np.ndarray): Time vector (N, ).
        q_mat (np.ndarray): State variables matrix (6, N).
        q_ref_mat (np.ndarray): Reference state variables matrix (6, N).
        u_mat (np.ndarray): Input forces matrix (4, N).
        v_current (np.ndarray): Current velocity vector (3, ). Assumed stationary.
        v_wind (np.ndarray): Wind velocity vector (3, ). Assumed stationary.
        speed_up_factor (int, optional): Speed-up factor for the animation.
        save (bool, optional): Whether to save the animation. Defaults to False.
        show (bool, optional): Whether to show the animation. Defaults to False.
    """

    # Initialize figure
    # Compute bounds:
    x_min = np.min(q_mat[1, :]) - 1
    x_max = np.max(q_mat[1, :]) + 1
    y_min = np.min(q_mat[0, :]) - 1
    y_max = np.max(q_mat[0, :]) + 1

    # Initialize figure
    fig, ax = initialize_phase_plot(x_min, x_max, y_min, y_max)
    plt.xticks([])
    plt.yticks([])

    # Determine number of frames to generate
    N = int(np.size(q_mat, 1) / speed_up_factor)

    # Initialize world reference frame (static)
    origin = np.array([0, 0])
    x_arrow_fixed = np.array([1, 0])
    y_arrow_fixed = np.array([0, 1])
    plt.arrow(*origin, *x_arrow_fixed, head_width=0.03, color="k")
    plt.arrow(*origin, *y_arrow_fixed, head_width=0.03, color="k")

    # Create a grid of points for the vector fields
    delta_x = 1.0
    delta_y = 1.0
    x = np.arange(x_min, x_max + delta_x, delta_x)
    y = np.arange(y_min, y_max + delta_y, delta_y)
    X, Y = np.meshgrid(x, y)

    # Plot current vector field (static)
    if v_current is not None:
        v_x = v_current[1]
        v_y = v_current[0]
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="blue")

    # Plot wind vector field (static)
    if v_wind is not None:
        v_x = v_wind[1]
        v_y = v_wind[0]
        v_x_mat = np.ones_like(X) * v_x
        v_y_mat = np.ones_like(Y) * v_y
        plt.quiver(X, Y, v_x_mat, v_y_mat, color="green")

    # Body reference frame (animated)
    psi = q_mat[5, 0]
    robot = q_mat[0:2, 0]
    arrow_body_x = 0.7 * x_arrow_fixed
    arrow_body_y = 0.7 * y_arrow_fixed
    rotation = R_i2b(psi)[0:2, 0:2]
    arrow_body_x = rotation.dot(arrow_body_x)
    arrow_body_y = rotation.dot(arrow_body_y)
    arr_x = plt.arrow(
        robot[0], robot[1], arrow_body_x[0], arrow_body_x[1], head_width=0.02, color="b"
    )
    arr_y = plt.arrow(
        robot[0], robot[1], arrow_body_y[0], arrow_body_y[1], head_width=0.02, color="b"
    )

    # Trajectory (animated)
    x = q_mat[0, 0]
    y = q_mat[1, 0]
    traj = ax.plot(x, y)

    # Target reference frame (animated)
    ref_psi = q_ref_mat[5, 0]
    ref_O = q_ref_mat[0:2, 0]
    arrow_ref_x = 0.7 * x_arrow_fixed
    arrow_ref_y = 0.7 * y_arrow_fixed
    rotation = R_i2b(ref_psi)[0:2, 0:2]
    arrow_ref_x = rotation.dot(arrow_ref_x)
    arrow_ref_y = rotation.dot(arrow_ref_y)
    ref_arr_x = plt.arrow(
        ref_O[0], ref_O[1], arrow_ref_x[0], arrow_ref_x[1], head_width=0.012, color="k"
    )
    ref_arr_y = plt.arrow(
        ref_O[0], ref_O[1], arrow_ref_y[0], arrow_ref_y[1], head_width=0.012, color="k"
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

    # Initialize empty lines for progressive filling
    u_line_0 = ax_u.plot([], [], "r-", lw=2)
    u_line_1 = ax_u.plot([], [], "r-", lw=2)
    u_line_2 = ax_u.plot([], [], "r-", lw=2)
    u_line_3 = ax_u.plot([], [], "r-", lw=2)

    # Generate the animation
    anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(
            t_vec,
            q_mat,
            q_ref_mat,
            u_mat,
            arr_x,
            arr_y,
            traj,
            ref_arr_x,
            ref_arr_y,
            u_line_0,
            u_line_1,
            u_line_2,
            u_line_3,
            speed_up_factor,
        ),
        frames=N,
        interval=50,
        blit=True,
        repeat=False,
    )

    # Save the animation
    if save:
        print("Saving animation...")
        anim.save(filename, writer="pillow", dpi=300)
        print("Animation saved.")

    if show:
        plt.show()
