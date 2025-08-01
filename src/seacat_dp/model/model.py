from typing import TYPE_CHECKING

import numpy as np

from seacat_dp.model import hydrodynamics

if TYPE_CHECKING:
    from seacat_dp.model.parameters import Parameters


class USVModel:
    """
    Nonlinear model class to compute the 2D dynamics of a USV.

    The state vector is defined as a (6, ) vector:
    q (np.ndarray): [x, y, theta, u_x, u_y, omega]

    where:
        x (float): position along x axis (inertial frame)
        y (float): position along y axis (inertial frame)
        theta (float): orientation (angle between body frame and inertial frame)
        u_x (float): velocity along x axis (body frame)
        u_y (float): velocity along y axis (body frame)
        omega (float): angular velocity (derivative of theta)

    The control input is vessel-specific and is defined in each class separately. Since
    the thrusters have a dynamics of their own, the force and angle at a given time are
    technically also part of the state vector.

    Note that the USV parameters are stored in each instance of the class, allowing to
    create models with different parameters for model mismatch studies.
    """

    def __init__(self, pars: "Parameters"):

        # Model parameters
        self.pars: Parameters = pars
        self.dt: float = 0.01  # model time step [s]
        self.integration_method: str = "euler"

        # State vector and control input
        self.q: np.ndarray = np.zeros(6)  # state vector [x, y, theta, u_x, u_y, omega]
        self.u: np.ndarray  # Define in subclasses

        ### CONSTANT MODEL PARAMETERS ###
        # Rigid body mass and inertia properties
        self.m = pars.m
        self.m_xg = pars.xg * pars.m
        M_rb = np.diag([pars.m, pars.m, pars.I_zz])
        M_rb[1, 2] = self.m_xg
        M_rb[2, 1] = self.m_xg

        # Added inertia properties. Multiplication factors are taken from the otter.py
        # model in the PythonVehicleSimulator.
        self.X_udot = hydrodynamics.added_mass_surge(pars)
        self.Y_vdot = 1.5 * pars.m
        self.N_rdot = 1.7 * pars.I_zz
        self.Y_rdot = 0
        M_a = np.diag([self.X_udot, self.Y_vdot, self.N_rdot])
        M_a[1, 2] = self.Y_rdot
        M_a[2, 1] = self.Y_rdot

        # Mass and inertia matrix (sum rigid and added mass and inertia properties)
        self.M = M_rb + M_a  # (3, 3) matrix
        self.M_inv = np.linalg.inv(self.M)  # (3, 3) matrix

        # Coriolis centripetal matrix
        # NOTE: The Coriolis matrix depends on the current USV state and therefore it is
        # updated at each time step in the dynamics function.
        self.C = np.zeros((3, 3))  # (3, 3) matrix
