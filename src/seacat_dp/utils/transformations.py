"""
This module contains functions to compute rotation matrices for transforming vectors
between the body reference frame and the inertial reference frame of the USV.
"""

import numpy as np


def R_b2i(theta: float) -> np.ndarray:
    """
    Computes the rotation matrix to transform coordinates from the body reference
    frame to the inertial reference frame.
    Args:
        theta (float): angle in radians.
    Returns:
        R (np.ndarray): rotation matrix (3, 3).
    """
    if not isinstance(theta, float):
        try:
            theta = float(theta)
        except ValueError as exc:
            raise TypeError("Theta must be a float or convertible to float.") from exc
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R


def R_i2b(theta: float) -> np.ndarray:
    """
    Computes the rotation matrix to transform coordinates from the inertial
    reference frame to the body reference frame.
    Args:
        theta (float): angle in radians.
    Returns:
        R (np.ndarray): rotation matrix (3, 3).
    """
    if not isinstance(theta, float):
        try:
            theta = float(theta)
        except ValueError as exc:
            raise TypeError("Theta must be a float or convertible to float.") from exc
    R = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R


def angle_wrap(a: float) -> float:
    """
    Helper function to avoid numerical issues in modulo arithmetic.

    Args:
        a (float): Angle in radians.

    Returns:
        float: Wrapped angle in radians within [-pi, pi].
    """
    return np.atan2(np.sin(a), np.cos(a))
