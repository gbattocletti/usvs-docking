import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

plt.rcParams["text.usetex"] = True

""" 
Useful links:
- Matplotlib RF plot https://thomascountz.com/2018/11/18/2d-coordinate-fromes-matplotlib
"""


def plot_variables(t_vec: np.ndarray, u_mat: np.ndarray, q_mat: np.ndarray):
    """
    Plot the input forces and the state variables of the 3DoF simulation.

    Args:
        t_vec (numpy.ndarray): Time vector (1, N).
        u_mat (numpy.ndarray): Input forces matrix (4, N).
        q_mat (numpy.ndarray): State variables matrix (6, N).
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

    # Initialize plot
    fig, ax = plt.subplots(nrows=5)
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
    ax[1].set(xlabel="", ylabel="$x, y$ [m]")
    ax[1].grid()
    ax[1].legend(["$x$", "$y$"])

    # Plot theta
    ax[2].plot(t_vec, q_mat[2, :])
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


def initialize_phase_plot() -> tuple[plt.Figure, Axes]:
    """
    Initialize the phase plot for the 3DoF simulation.

    Returns:
        fig (plt.Figure): Figure object.
        ax (Axes): Axes object.
    """
    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    plt.xlim(-5, 5)  # Set x-axis range
    plt.ylim(-1, 10)  # Set y-axis range
    ax.set(xlabel="y earth [m]", ylabel="x earth [m]")

    return fig, ax


def phase_plot(q_mat: np.ndarray, idx: int = -1) -> None:
    """
    Plot the trajectory of the robot in the phase space (x-y plane).

    Args:
        q_mat (np.ndarray): state variables matrix (6, N).
        idx (int, optional): index of the state at which to plot the system's body
            reference frame. Defaults to -1.
    """
    # Initialize figure
    _, ax = initialize_phase_plot()

    # World reference frame
    origin = np.array([0, 0])
    x_arrow_fixed = np.array([1, 0])
    y_arrow_fixed = np.array([0, 1])
    plt.arrow(*origin, *x_arrow_fixed, head_width=0.03, color="k")
    plt.arrow(*origin, *y_arrow_fixed, head_width=0.03, color="k")

    # Body reference frame
    psi = q_mat[2, idx]
    robot = q_mat[0:2, idx]
    x_arrow_body = 0.7 * x_arrow_fixed
    y_arrow_body = 0.7 * y_arrow_fixed
    rotation = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
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

    # Trajectory
    x = q_mat[0, :]
    y = q_mat[1, :]
    ax.plot(x, y)

    # TODO: set axis limits depending on the size of the trajectory


def animation_step(
    frame_idx: int,
    q_mat: np.ndarray,
    arr_x: FancyArrow,
    arr_y: FancyArrow,
    traj: list[Line2D],
    speed_up_factor: int,
) -> tuple[FancyArrow, FancyArrow, list[Line2D]]:
    """
    Update the animation components for the animation of the phase plot.

    Args:
        frame_idx (int): Frame index.
        q_mat (np.ndarray): State variables matrix (6, N).
        arr_x (FancyArrow): Arrow object for x-axis.
        arr_y (FancyArrow): Arrow object for y-axis.
        traj (list[Line2D]): Trajectory plot object.
        speed_up_factor (int): Speed-up factor for the animation.

    Returns:
        tuple (tuple[FancyArrow, FancyArrow, list[Line2D]]): Updated arrow objects and
            trajectory object.
    """

    # Find data index adjusted for speed-up factor
    idx = frame_idx * speed_up_factor

    # Get updated values of robot location and direction
    robot = q_mat[0:2, idx]
    psi = q_mat[5, idx]
    rotation = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    # Update body reference frame
    arr_len = 0.7
    x_arrow_body = rotation.dot(arr_len * np.array([1, 0]))
    y_arrow_body = rotation.dot(arr_len * np.array([0, 1]))

    # Update plot elements
    arr_x.set_data(x=robot[0], y=robot[1], dx=x_arrow_body[0], dy=x_arrow_body[1])
    arr_y.set_data(x=robot[0], y=robot[1], dx=y_arrow_body[0], dy=y_arrow_body[1])
    traj[0].set_data(q_mat[0, 0:idx], q_mat[1, 0:idx])

    return (arr_x, arr_y, traj[0])


def animate(
    filename: str,
    q_mat: np.ndarray,
    speed_up_factor: int,
    save: bool = False,
    show: bool = False,
) -> None:
    """
    Animate the trajectory of the robot in the phase space (x-y plane).

    Args:
        filename (str): Filename to save the animation.
        q_mat (np.ndarray): State variables matrix (6, N).
        speed_up_factor (int): Speed-up factor for the animation.
        save (bool, optional): Whether to save the animation. Defaults to False.
        show (bool, optional): Whether to show the animation. Defaults to False.
    """

    # Initialize figure
    fig, ax = initialize_phase_plot()
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

    # Body reference frame (animated)
    psi = q_mat[5, 0]
    robot = q_mat[0:2, 0]
    arrow_body_x = 0.7 * x_arrow_fixed
    arrow_body_y = 0.7 * y_arrow_fixed
    rotation = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
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

    # Generate the animation
    anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(q_mat, arr_x, arr_y, traj, speed_up_factor),
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
