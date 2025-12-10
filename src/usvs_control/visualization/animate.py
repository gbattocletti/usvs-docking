import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Polygon, Rectangle

from usvs_control.utils.transformations import R_i2b
from usvs_control.visualization.colors import CmdColors
from usvs_control.visualization.plot import initialize_phase_plot


def animation_step(
    frame_idx: int,
    t_vec: np.ndarray,
    q_mat: np.ndarray,
    q_ref_mat: np.ndarray,
    u_mat: np.ndarray,
    arr_body_x: FancyArrow,
    arr_body_y: FancyArrow,
    traj_body: list[Line2D],
    poly: Polygon,
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
    Polygon,
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
        poly (Polygon): Polygon object for the robot body.
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

    # Update box
    poly.set_xy(generate_box_points(q_mat[1, idx], q_mat[0, idx], psi_body))

    # Update target reference frame
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
        poly,
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
    **kwargs,
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

    Kwargs:
        subplot_legend (bool, optional): Whether to show legend in the actuators
            subplot. Default is False.
    Returns:
        anim (FuncAnimation): Animation object.
    """
    # Parse kwargs
    subplot_legend: bool = False
    for key, value in kwargs.items():
        if key == "subplot_legend":
            if not isinstance(value, bool):
                raise TypeError("subplot_legend must be a boolean value.")
            subplot_legend = value
        else:
            print(
                f"{CmdColors.WARNING}[Animate]{CmdColors.ENDC} Unrecognized kwarg "
                f"'{key}' passed to generate_animation()."
            )

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
    ax.set_axisbelow(True)
    title = ax.set_title(f"2D Trajectory (t = {t_vec[0]:.2f} s)", fontsize=10)

    # Determine number of frames to generate
    N = int(np.size(q_mat, 1) / speed_up_factor)

    # Initialize world reference frame (static)
    origin_body = np.array([0, 0])
    arrow_inertial_x = np.array([0, 1])
    arrow_inertial_y = np.array([1, 0])
    plt.arrow(*origin_body, *arrow_inertial_x, head_width=0.03, color="k", zorder=10)
    plt.arrow(*origin_body, *arrow_inertial_y, head_width=0.03, color="k", zorder=10)

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
        plt.quiver(
            grid_X,
            grid_Y,
            v_x_mat,
            v_y_mat,
            color="blue",
            width=0.003,
            zorder=4,
        )

    # Plot wind vector field (static)
    if v_wind is not None and np.linalg.norm(v_wind) > 0:
        v_x = v_wind[1]
        v_y = v_wind[0]
        v_x_mat = np.ones_like(grid_X) * v_x
        v_y_mat = np.ones_like(grid_Y) * v_y
        plt.quiver(
            grid_X,
            grid_Y,
            v_x_mat,
            v_y_mat,
            color="green",
            width=0.003,
            zorder=5,
        )

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
        zorder=9,
    )
    arr_body_y = plt.arrow(
        origin_body[0],
        origin_body[1],
        dx=arrow_body_y[0],
        dy=arrow_body_y[1],
        head_width=0.02,
        color="b",
        zorder=9,
    )

    # Body shape (animated)
    poly = Polygon(
        generate_box_points(
            origin_body[0],
            origin_body[1],
            psi_body,
        ),
        closed=True,
        fill=False,
        edgecolor="b",
        linewidth=1,
        zorder=8,
    )
    ax.add_patch(poly)

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
        zorder=7,
    )
    arr_ref_y = plt.arrow(
        origin_ref[0],
        origin_ref[1],
        dx=arrow_ref_y[0],
        dy=arrow_ref_y[1],
        head_width=0.012,
        color="k",
        zorder=7,
    )

    # Add actuators subplot
    # Position: [left, bottom, width, height] (w.r.t. main plot)
    ax_u = plt.axes([0.15, 0.65, 0.3, 0.2])  # Top-left corner
    rect = Rectangle((0, -2), 10, 4, alpha=0.2, color="gray")
    ax_u.add_patch(rect)
    ax_u.set_xlabel("t [s]")
    ax_u.set_ylabel("u [N]")
    ax_u.set_xlim(0, np.max(t_vec))
    ax_u.set_ylim(-1000, 1200)  # Thrusters force limits
    ax_u.grid()
    ax_u.set_axisbelow(True)
    ax_u.set_title("Thrusters forces", fontsize=10)

    # Initialize empty lines for progressive filling
    (u_line_0,) = ax_u.plot([], [], color="darkred", lw=1)
    (u_line_1,) = ax_u.plot([], [], color="red", lw=1)
    (u_line_2,) = ax_u.plot([], [], color="orange", lw=1)
    (u_line_3,) = ax_u.plot([], [], color="gold", lw=1)

    # Add legend to the actuators subplot
    if subplot_legend is True:
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
            poly,
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


def generate_box_points(
    x: float,
    y: float,
    heading: float,
) -> np.ndarray:
    """
    Generate the vertices of a rectangle given its center, heading, width, and height.

    Args:
        x (float): x-coordinate of the rectangle center.
        y (float): y-coordinate of the rectangle center.
        heading (float): Heading angle of the rectangle in radians.

    Returns:
        np.ndarray: Array of shape (4, 2) containing the vertices of the rectangle
    """
    # Rectangle dimensions
    width: float = 0.8
    height: float = 1.6
    bow_length: float = 0.4

    # centered rectangle vertices
    pts = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [0, height / 2 + bow_length],  # bow point to indicate forward direction
            [-width / 2, height / 2],
        ]
    )

    # rotation matrix
    R = np.array(
        [
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ]
    )

    # rotate and translate points
    return pts @ R + np.array([x, y])
