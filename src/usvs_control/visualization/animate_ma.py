import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Polygon

from usvs_control.utils.transformations import R_i2b
from usvs_control.visualization import plot_ma


def animation_step(
    frame_idx: int,
    q_mat: np.ndarray,
    speed_up_factor: int,
    arr_body_x_1: FancyArrow,
    arr_body_y_1: FancyArrow,
    arr_body_x_2: FancyArrow,
    arr_body_y_2: FancyArrow,
    traj_body_1: list[Line2D],
    traj_body_2: list[Line2D],
    poly_1: Polygon,
    poly_2: Polygon,
) -> tuple[
    FancyArrow,
    FancyArrow,
    FancyArrow,
    FancyArrow,
    Line2D,
    Line2D,
    Polygon,
    Polygon,
]:
    """
    Update the animation components for the animation of the phase plot.

    Args:
        frame_idx (int): Frame index.
        q_mat (np.ndarray): State variables matrix (12, N).
        speed_up_factor (int): Speed-up factor for the animation.
        arr_body_x_1 (FancyArrow): Arrow object for x-axis of USV 1.
        arr_body_y_1 (FancyArrow): Arrow object for y-axis of USV 1.
        arr_body_x_2 (FancyArrow): Arrow object for x-axis of USV 2.
        arr_body_y_2 (FancyArrow): Arrow object for y-axis of USV 2.
        traj_body_1: list[Line2D],
        traj_body_2: list[Line2D],
        poly_1 (Polygon): Polygon object for the robot body for USV 1.
        poly_2 (Polygon): Polygon object for the robot body for USV 2.

    Returns:
        updated_objects (tuple): Updated arrow objects and trajectory object.
    """
    # Animation plot parameters
    arr_len = 0.7
    arr_inertial_x = arr_len * np.array([0, 1])
    arr_inertial_y = arr_len * np.array([1, 0])

    # Find data index adjusted for speed-up factor
    idx = frame_idx * speed_up_factor

    # Update body reference frames
    psi_body_1 = q_mat[2, idx]
    psi_body_2 = q_mat[8, idx]
    R_body_1 = R_i2b(psi_body_1)[0:2, 0:2]
    R_body_2 = R_i2b(psi_body_2)[0:2, 0:2]
    x_arrow_body_1 = R_body_1.dot(arr_inertial_x)
    y_arrow_body_1 = R_body_1.dot(arr_inertial_y)
    x_arrow_body_2 = R_body_2.dot(arr_inertial_x)
    y_arrow_body_2 = R_body_2.dot(arr_inertial_y)
    arr_body_x_1.set_data(
        x=q_mat[1, idx], y=q_mat[0, idx], dx=x_arrow_body_1[0], dy=x_arrow_body_1[1]
    )
    arr_body_y_1.set_data(
        x=q_mat[1, idx], y=q_mat[0, idx], dx=y_arrow_body_1[0], dy=y_arrow_body_1[1]
    )
    arr_body_x_1.set_data(
        x=q_mat[1, idx], y=q_mat[0, idx], dx=x_arrow_body_1[0], dy=x_arrow_body_1[1]
    )
    arr_body_y_1.set_data(
        x=q_mat[1, idx], y=q_mat[0, idx], dx=y_arrow_body_1[0], dy=y_arrow_body_1[1]
    )
    arr_body_x_2.set_data(
        x=q_mat[7, idx], y=q_mat[6, idx], dx=x_arrow_body_2[0], dy=x_arrow_body_2[1]
    )
    arr_body_y_2.set_data(
        x=q_mat[7, idx], y=q_mat[6, idx], dx=y_arrow_body_2[0], dy=y_arrow_body_2[1]
    )
    arr_body_x_2.set_data(
        x=q_mat[7, idx], y=q_mat[6, idx], dx=x_arrow_body_2[0], dy=x_arrow_body_2[1]
    )
    arr_body_y_2.set_data(
        x=q_mat[7, idx], y=q_mat[6, idx], dx=y_arrow_body_2[0], dy=y_arrow_body_2[1]
    )

    # Update box
    poly_1.set_xy(generate_box_points(q_mat[1, idx], q_mat[0, idx], psi_body_1))
    poly_2.set_xy(generate_box_points(q_mat[7, idx], q_mat[6, idx], psi_body_2))

    # Update robot trajectory
    traj_body_1.set_data(q_mat[1, 0:idx], q_mat[0, 0:idx])
    traj_body_2.set_data(q_mat[7, 0:idx], q_mat[6, 0:idx])

    return (
        arr_body_x_1,
        arr_body_y_1,
        arr_body_x_2,
        arr_body_y_2,
        traj_body_1,
        traj_body_2,
        poly_1,
        poly_2,
    )


def generate_animation(
    q_mat: np.ndarray,
    v_current: np.ndarray,
    speed_up_factor: int = 1,
) -> FuncAnimation:
    """
    Animate the trajectory of the robot in the phase space (x-y plane).

    Args:
        t_vec (np.ndarray): Time vector (N, ).
        q_mat (np.ndarray): State variables matrix (12, N).
        v_current (np.ndarray): Current velocity vector (3, ). Assumed stationary.
        speed_up_factor (int, optional): Speed-up factor for the animation.

    Returns:
        anim (FuncAnimation): Animation object.
    """

    # Initialize figure
    # Compute bounds:
    x_min = -1.0
    x_max = 11.5
    y_min = -1.0
    y_max = 11.0

    # Initialize figure
    fig, ax = plot_ma.initialize_phase_plot(x_min, x_max, y_min, y_max)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    # Determine number of frames to generate
    N = int(np.size(q_mat, 1) / speed_up_factor)

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
            color="#20509c",
            width=0.003,
            zorder=1,
        )

    # Trajectories (animated)
    (traj_body_1,) = ax.plot(
        q_mat[1, 0],
        q_mat[0, 0],
        color="#0CB143",
        zorder=7,
    )
    (traj_body_2,) = ax.plot(
        q_mat[7, 0],
        q_mat[6, 0],
        color="#AA1313",
        zorder=7,
    )

    # Body reference frames (animated)
    arrow_inertial_x = 0.7 * np.array([0, 1])
    arrow_inertial_y = 0.7 * np.array([1, 0])
    psi_body_1 = q_mat[2, 0]
    psi_body_2 = q_mat[8, 0]
    origin_body_1 = q_mat[0:2, 0]
    origin_body_2 = q_mat[6:8, 0]
    R_body_1 = R_i2b(psi_body_1)[0:2, 0:2]
    R_body_2 = R_i2b(psi_body_2)[0:2, 0:2]
    arrow_body_x_1 = R_body_1.dot(arrow_inertial_x)
    arrow_body_y_1 = R_body_1.dot(arrow_inertial_y)
    arrow_body_x_2 = R_body_2.dot(arrow_inertial_x)
    arrow_body_y_2 = R_body_2.dot(arrow_inertial_y)
    arr_body_x_1 = plt.arrow(
        origin_body_1[0],
        origin_body_1[1],
        dx=arrow_body_x_1[0],
        dy=arrow_body_x_1[1],
        head_width=0.02,
        color="#000000",
        zorder=9,
    )
    arr_body_y_1 = plt.arrow(
        origin_body_1[0],
        origin_body_1[1],
        dx=arrow_body_y_1[0],
        dy=arrow_body_y_1[1],
        head_width=0.02,
        color="#000000",
        zorder=9,
    )
    arr_body_x_2 = plt.arrow(
        origin_body_2[0],
        origin_body_2[1],
        dx=arrow_body_x_2[0],
        dy=arrow_body_x_2[1],
        head_width=0.02,
        color="#000000",
        zorder=9,
    )
    arr_body_y_2 = plt.arrow(
        origin_body_2[0],
        origin_body_2[1],
        dx=arrow_body_y_2[0],
        dy=arrow_body_y_2[1],
        head_width=0.02,
        color="#000000",
        zorder=9,
    )

    # Body shapes (animated)
    poly_1 = Polygon(
        generate_box_points(
            origin_body_1[0],
            origin_body_1[1],
            psi_body_1,
        ),
        closed=True,
        fill=False,
        color="#006421",
        linewidth=1,
        zorder=8,
    )
    poly_2 = Polygon(
        generate_box_points(
            origin_body_2[0],
            origin_body_2[1],
            psi_body_2,
        ),
        closed=True,
        fill=False,
        color="#790000",
        linewidth=1,
        zorder=8,
    )
    ax.add_patch(poly_1)
    ax.add_patch(poly_2)

    # Generate the animation
    anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(
            q_mat,
            speed_up_factor,
            arr_body_x_1,
            arr_body_y_1,
            arr_body_x_2,
            arr_body_y_2,
            traj_body_1,
            traj_body_2,
            poly_1,
            poly_2,
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
