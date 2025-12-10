from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from usvs_control.model import hydrodynamics

if TYPE_CHECKING:
    from usvs_control.model.parameters import Parameters


class USVModel(ABC):
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

    def __init__(self, pars: "Parameters") -> None:

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
        # Note: The Coriolis matrix depends on the current USV state and therefore it is
        # updated at each time step in the dynamics function.
        self.C = np.zeros((3, 3))  # (3, 3) matrix

        # Linear damping matrix
        # Note: the 10 factor is added to get a larger Xu value, since in some sources
        # the value of Xu is in the order of 1000.
        self.Xu = 24.4 * pars.g / pars.u_max * 10
        self.Yv = self.M[1, 1] / pars.t_sway
        self.Nr = self.M[2, 2] / pars.t_yaw
        self.D_L = np.diag([self.Xu, self.Yv, self.Nr])

        # Nonlinear damping matrix
        # Note: The nonlinear damping matrix depends on the current USV state and is
        # therefore updated directly in the dynamics function.
        self.D_NL = np.zeros((3, 3))  # (3, 3) matrix

        # Draft from Fossen eq
        self.draft_fossen = pars.m / (
            2 * pars.rho * pars.c_b_pontoon * pars.l_pontoon * pars.b_pontoon
        )  # draft estimated from approximated submerged volume (submerged 'box')

    def set_time_step(self, dt: float) -> None:
        """
        Sets the time step for the model.

        Args:
            dt (float): time step [s]
        """
        self.dt = dt

    def set_integration_method(self, method: str):
        """
        Sets the integration method for the model.

        Args:
            method (str): integration method {"euler", "rk4"}
        """
        if not method in ["euler", "rk4"]:
            raise ValueError(
                f"Integration method {method} not implemented. Use 'euler' or 'rk4'."
            )
        self.integration_method = method

    def set_initial_conditions(self, q0: np.ndarray) -> None:
        """
        Sets the initial conditions for the model.

        Args:
            q0 (np.ndarray): initial state vector [x, y, theta, u_x, u_y, omega]^T. The
                    position components are expressed in the inertial reference frame,
                    while the velocity ones are expressed in the body reference frame.
        """
        if q0.shape != (6,):
            raise ValueError("Initial conditions must be a (6, ) vector.")
        self.q = q0

    def __str__(self) -> None:
        """
        Returns a string representation of the model state.
        """
        q_str = np.array2string(
            self.q, formatter={"float": lambda x: f"{x:.2f}"}, separator=", "
        )
        u_str = np.array2string(
            self.u, formatter={"float": lambda x: f"{x:.2f}"}, separator=", "
        )
        return f"NL model state:\n\tq={q_str}\n\tu={u_str}"

    def __call__(
        self,
        u: np.ndarray,
        v_current: np.ndarray,
        v_wind: np.ndarray,
    ) -> np.ndarray:
        return self.step(u, v_current, v_wind)

    def step(
        self,
        u: np.ndarray,
        v_current: np.ndarray,
        v_wind: np.ndarray,
        use_thruster_dynamics: bool = False,
    ) -> np.ndarray:
        """
        Computes the next state of the model using the nonlinear dynamics.

        Args:
            u (np.ndarray): (4, ) control input vector.
                where:
                    u[0] (float): force of the left stern (rear) thruster
                    u[1] (float): force of the right stern (rear) thruster
                    u[2] (float): force of the left bow thruster
                    u[3] (float): force of the right bow thruster

            v_current (np.ndarray): (3, ) vector of the water current speed (measured or
                    estimated) acting on the boat [m/s]. The vector is expressed in the
                    inertial reference frame.
                where:
                    v_current[0] (float): current speed along x axis
                    v_current[1] (float): current speed along y axis
                    v_current[2] (float): rotational speed around z axis (CW = positive)

            v_wind (np.ndarray): (3, ) vector of the wind speed (measured or estimated
                    disturbance) acting on the boat [m/s]. The vector is expressed in
                    the intertial reference frame.
                where:
                    v_wind[0] (float): wind speed along x axis
                    v_wind[1] (float): wind speed along y axis
                    v_wind[2] (float): rotational speed around z axis (positive for CW)

            use_thruster_dynamics (bool): whether to use the thruster dynamics or not.
                Default is False, meaning that the control input u is directly applied
                to the USV dynamics.

        Returns:
            q_plus (np.ndarray): updated state vector q.
        """

        if self.integration_method == "euler":

            # thruster dynamics
            if use_thruster_dynamics:
                self.u = self.u + self.thrusters_dynamics(self.u, u) * self.dt
                self.u = self.thrusters_saturation(self.u)
            else:
                self.u = self.thrusters_saturation(u)

            # USV dynamics
            self.q = self.q + self.dynamics(self.q, self.u, v_current, v_wind) * self.dt

        elif self.integration_method == "rk4":
            # thrusters dynamics
            if use_thruster_dynamics:
                k1 = self.thrusters_dynamics(self.u, u)
                k2 = self.thrusters_dynamics(self.u + k1 * self.dt / 2, u)
                k3 = self.thrusters_dynamics(self.u + k2 * self.dt / 2, u)
                k4 = self.thrusters_dynamics(self.u + k3 * self.dt, u)
                self.u = self.u + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
                self.u = self.thrusters_saturation(self.u)
            else:
                self.u = self.thrusters_saturation(u)

            # USV dynamics
            k1 = self.dynamics(self.q, self.u, v_current, v_wind)
            k2 = self.dynamics(self.q + k1 * self.dt / 2, self.u, v_current, v_wind)
            k3 = self.dynamics(self.q + k2 * self.dt / 2, self.u, v_current, v_wind)
            k4 = self.dynamics(self.q + k3 * self.dt, self.u, v_current, v_wind)
            self.q = self.q + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

            # Correct angle to be in the range (-pi, pi]
            self.q[2] = (self.q[2] + np.pi) % (2 * np.pi) - np.pi  # normalize angle

        else:
            raise ValueError(
                f"Integration method {self.integration_method} not implemented."
            )

        # Return the updated state
        return self.q

    @abstractmethod
    def dynamics(
        self, q: np.ndarray, u: np.ndarray, v_current: np.ndarray, v_wind: np.ndarray
    ) -> np.ndarray:
        """
        Computes the dynamics of the USV model.

        Args:
            q (np.ndarray): state vector [x, y, theta, u_x, u_y, omega]
            u (np.ndarray): control input vector
            v_current (np.ndarray): current velocity vector [u_c, v_c, r_c]
            v_wind (np.ndarray): wind velocity vector [u_w, v_w, r_w]

        Returns:
            q_dot (np.ndarray): time derivative of the state vector
        """
        raise NotImplementedError(
            "The dynamics method must be implemented in the subclasses."
        )

    @abstractmethod
    def thrusters_dynamics(self, u: np.ndarray, u_input: np.ndarray) -> np.ndarray:
        """
        Computes the dynamics of the thrusters.

        Args:
            u (np.ndarray): vector of current control input
            u_input (np.ndarray): vector of desired control input

        Returns:
            u_dot (np.ndarray): time derivative of the control input
        """
        raise NotImplementedError(
            "The thruster dynamics method must be implemented in the subclasses."
        )

    @abstractmethod
    def thrusters_saturation(self, u: np.ndarray) -> np.ndarray:
        """
        Saturates the thrusters forces.

        Args:
            u (np.ndarray): vector of control inputs

        Returns:
            u_clipped (np.ndarray): clipped vector of control inputs
        """
        raise NotImplementedError(
            "The thrusters saturation method must be implemented in the subclasses."
        )
