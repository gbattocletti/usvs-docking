import numpy as np


def linearize(phi: float, H: np.ndarray) -> np.ndarray:
    """
    Linearize the system around the current state. The linearization assumes small
    rotations around the current heading angle and removes the nonlinear damping from
    the equations of motion.

    Args:
        phi (float): heading angle of the USV [rad] (0 <= phi < 2 * pi)
        H (np.ndarray): dynamic matrix defined as H = -M^-1 @ D_L

    Returns:
        A (np.ndarray): A matrix of the linearized system (6, 6).
    """

    # Build the A matrix
    A = np.array(
        [
            [0, 0, 0, np.cos(phi), -np.cos(phi), 0],
            [0, 0, 0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    A[3:6, 3:6] = H

    return A
