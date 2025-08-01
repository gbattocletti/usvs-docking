from typing import TYPE_CHECKING

import numpy as np

from seacat_dp.utils.transformations import R_i2b

if TYPE_CHECKING:
    from seacat_dp.model.parameters import Parameters


def added_mass_surge(pars: "Parameters") -> float:
    """
    Computes an approximation of the added mass in surge (i.e., along the x axis)
    for a boat of mass m and length L. The function is adapted from the MSS. Note
    that in the pythonVehicleSimulator the value Xudot = 0.1*mass is used instead.

    Args:
        pars (Parameters): object containing the parameters for the USV model.

    Returns:
        float: added mass in surge [kg]
    """

    rho = 1025  # default density of water [kg/m^3]
    nabla = pars.m / rho  # volume displacement

    # compute the added mass in surge using the formula by Söding (1982)
    Xudot = (2.7 * rho * nabla ** (5 / 3)) / (pars.l_tot**2)

    return Xudot


def crossflow_drag(q: np.ndarray, pars: "Parameters", v_curr: np.ndarray) -> np.ndarray:
    """
    Computes the forces acting on the boat due to water currents using strip theory.
    The function is adapted from the PythonVehicleSimulator function crossFlowDrag
    (see /lib/gnc.py and /vehicles/otter.py).

    Note: the nominal draft is used instead of the approximated one estimated from
    the 'box' submerged volume. This should result in a more accurate drag value.

    Args:
        q (np.ndarray): (6, ) state vector of the USV
        pars (Parameters): object containing the parameters of the USV model
        v_curr (np.ndarray): (3, ) vector of the water current velocity expressed in
            the inertial reference frame.

    Returns:
        tau (np.ndarray): a (3, ) ndarray representing the force vector due to water
        drag acting on the center of mass of the boat. The force vector is expressed
        in the body reference frame.
    """
    # validate input
    if v_curr.shape != (3,):
        raise ValueError("v_curr must be a (3, ) vector.")

    # transform to body frame
    v_curr_b = R_i2b(q[2]) @ v_curr

    # initialize strip theory parameters
    # NOTE: we assume symmetry along the x axis, i.e., the current effect on the
    # front and back of the boat is the same, so there is no torque (tau[2] = 0).
    c_d = hoerner(pars)  # cross-flow drag coefficient
    n_strips = 20  # number of strips
    dx = pars.l_tot / n_strips
    x = pars.l_tot / 2  # strip position along the x axis

    # compute the cross-flow velocity
    v_r_y = q[4] - v_curr_b[1]  # relative velocity along y axis
    v_cf = np.abs(v_r_y) * v_r_y  # cross-flow velocity

    # initialize force vector
    tau = np.zeros(3)  # force vector due to water drag acting on the center of mass

    # compute the forces acting on each strip
    for _ in range(n_strips + 1):

        tau[0] += 0
        tau[1] += -0.5 * pars.rho * c_d * pars.draft * dx * v_cf
        tau[2] += -0.5 * pars.rho * c_d * pars.draft * dx * v_cf * x

        x += dx  # move to the next strip

    return tau


def hoerner(pars: "Parameters") -> float:
    """
    Helper function for crossflow_drag.

    Computes the 2D Hoerner cross-flow coefficient as a function of beam and draft
    values. The data is interpolated to find the cross-flow coefficient for any
    beam/draft pair. The function is adapted from the PythonVehicleSimulator (see
    see /lib/gnc.py/Hoerner).

    Args:
        pars (Parameters): object containing the parameters of the USV model.

    Returns:
        float: 2D Hoerner cross-flow coefficient [-]
    """

    # DATA = [B/2T  C_D]
    DATA_B2T = np.array(
        [
            0.0109,
            0.1766,
            0.3530,
            0.4519,
            0.4728,
            0.4929,
            0.4933,
            0.5585,
            0.6464,
            0.8336,
            0.9880,
            1.3081,
            1.6392,
            1.8600,
            2.3129,
            2.6000,
            3.0088,
            3.4508,
            3.7379,
            4.0031,
        ]
    )
    DATA_CD = np.array(
        [
            1.9661,
            1.9657,
            1.8976,
            1.7872,
            1.5837,
            1.2786,
            1.2108,
            1.0836,
            0.9986,
            0.8796,
            0.8284,
            0.7599,
            0.6914,
            0.6571,
            0.6307,
            0.5962,
            0.5868,
            0.5859,
            0.5599,
            0.5593,
        ]
    )

    # Interpolate the data to get the cross-flow coefficient
    x = pars.b_tot / (2 * pars.draft)  # B/2T
    h_coeff = np.interp(x, DATA_B2T, DATA_CD)

    return h_coeff
