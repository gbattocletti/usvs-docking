import numpy as np

from seacat_dp.utils.transformations import R_i2b


def wind_load(q: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
    """
    Computes the forces acting on the boat due to wind using a simple model.

    Args:
        v_wind (np.ndarray): wind speed vector expressed in the inertial reference
        frame [m/s].

    Returns:
        b_wind (np.ndarray): a (3, ) ndarray representing the force vector due to
        wind acting on the center of mass of the boat. The force vector is expressed
        in the body reference frame.
    """
    # validate input
    if v_wind.shape != (3,):
        raise ValueError("v_wind must be a (3, ) vector.")

    # transform to body frame
    v_wind_b = R_i2b(q[2]) @ v_wind

    # compute the wind load
    b_wind = np.zeros(v_wind_b.shape)  # TODO: implement wind load model

    return b_wind
