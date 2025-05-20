import numpy as np

from seacat_dp.model.parameters import Parameters
from seacat_dp.utils.transformations import R_b2i, R_i2b


class NonlinearModel:
    """
    Nonlinear 2D model class that computes the nonlinear state and output equations.

    The state vector is defined as a (6, ) vector:
    q (np.ndarray): [x, y, theta, u_x, u_y, omega]

    where:
        x (float): position along x axis (inertial frame)
        y (float): position along y axis (inertial frame)
        theta (float): orientation (angle between body frame and inertial frame)
        u_x (float): velocity along x axis (body frame)
        u_y (float): velocity along y axis (body frame)
        omega (float): angular velocity (derivative of theta)

    Additionally, the thruster forces are stored in the model as a (4, ) vector:
    f (np.ndarray): [f_r, f_l, f_bow_r, f_bow_l]

    where:
        f_r (float): force to the right stern (rear) thruster
        f_l (float): force to the left stern (rear) thruster
        f_bow_r (float): force to the right bow thruster
        f_bow_l (float): force to the left bow thruster

    Finally, the instance attributes contain all the constant model parameters and
    matrices, which are computed at initialization time starting from the input
    parameter object.
    """

    def __init__(self, par: Parameters):
        """
        Initialize the nonlinear model of the SeaCat2.

        Args:
            parameters (Parameters): Parameters object containing the parameters for
            the model.
        """
        # NOTE: Some of the parameters are copied and saved in as attribute for later
        # use in the dynamics function.

        ### MODEL PARAMETERS ###
        # Model parameters
        self.par = par  # parameters object
        self.dt = 0.001  # time step [s]
        self.integration_method = "euler"  # {"euler", "rk4", ...}

        # Model state
        self.q = np.zeros(6)  # state vector [x, y, theta, u_x, u_y, omega]
        self.f = np.zeros(4)  # force vector [f_r, f_l, f_bow_r, f_bow_l]

        ### CONSTANT MODEL PARAMETERS ###
        # Rigid body mass and inertia properties
        self.m = par.m
        self.m_xg = par.xg * par.m
        M_rb = np.diag([par.m, par.m, par.I_zz])
        M_rb[1, 2] = self.m_xg
        M_rb[2, 1] = self.m_xg

        # Added inertia properties. Multiplication factors are taken from the otter.py
        # model in the PythonVehicleSimulator.
        self.X_udot = self.added_mass_surge(par)
        self.Y_vdot = 1.5 * par.m
        self.N_rdot = 1.7 * par.I_zz
        self.Y_rdot = 0
        M_a = np.diag([self.X_udot, self.Y_vdot, self.N_rdot])
        M_a[1, 2] = self.Y_rdot
        M_a[2, 1] = self.Y_rdot

        # Mass and inertia matrix (sum rigid and added mass and inertia properties)
        self.M = M_rb + M_a  # (3, 3) matrix
        self.M_inv = np.linalg.inv(self.M)  # (3, 3) matrix

        # Coriolis centripetal matrix
        # The Coriolis matrix depends on the current USV state and is therefore updated
        # directly in the dynamics function.
        self.C = np.zeros((3, 3))  # (3, 3) matrix

        # Linear damping matrix
        self.Xu = 24.4 * par.g / par.u_max
        self.Yv = self.M[1, 1] / par.t_sway
        self.Nr = self.M[2, 2] / par.t_yaw
        self.D_L = np.diag([self.Xu, self.Yv, self.Nr])

        # Nonlinear damping matrix
        # The nonlinear damping matrix depends on the current USV state and is therefore
        # updated directly in the dynamics function.
        self.D_NL = np.zeros((3, 3))  # (3, 3) matrix

        # Thrust allocation matrix
        self.T = np.array(
            [
                [1, 1, np.sin(par.alpha), np.sin(par.alpha)],
                [0, 0, np.cos(par.alpha), -np.cos(par.alpha)],
                [
                    par.d_stern,
                    -par.d_stern,
                    par.l_bow * np.cos(par.alpha) + par.d_bow * np.sin(par.alpha),
                    -par.l_bow * np.cos(par.alpha) - par.d_bow * np.sin(par.alpha),
                ],
            ]
        )  # (3, 4) matrix

        # Thrusters weight matrix
        W_mat = np.diag([1, 1, 10, 10])  # (4, 4) matrix

        # Pseudo-inverse of the thrust matrix (maps CoM forces to thruster forces)
        W_inv = np.linalg.inv(W_mat)  # Inverse of the weight matrix
        self.T_pinv = W_inv @ self.T.T @ np.linalg.inv(self.T @ W_inv @ self.T.T)

        # Thrusters time delay
        self.thrust_delay = np.array([par.t_stern, par.t_stern, par.t_bow, par.t_bow])

    def set_time_step(self, dt: float):
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
            method (str): integration method {"euler", "rk4", ...}
        """
        if not method in ["euler", "rk4"]:
            raise ValueError(
                f"Integration method {method} not implemented. Use 'euler' or 'rk4'."
            )
        self.integration_method = method

    def set_initial_conditions(self, q0: np.ndarray):
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

    def get_H(self) -> np.ndarray:
        """
        Returns the matrix H = -M^-1 @ D_L used to build the LTI state-space model of
        the system.

        Returns:
            H (np.ndarray): (3, 3) matrix.
        """
        H = -self.M_inv @ self.D_L
        return H

    def __str__(self):
        """
        Returns a string representation of the model state.
        """
        q_str = np.array2string(
            self.q, formatter={"float": lambda x: f"{x:.2f}"}, separator=", "
        )
        f_str = np.array2string(
            self.f, formatter={"float": lambda x: f"{x:.2f}"}, separator=", "
        )
        return f"NL model state:\n\tq={q_str}\n\tf={f_str}"

    def __call__(
        self,
        u: np.ndarray,
        b_current: np.ndarray,
        b_wind: np.ndarray,
    ) -> np.ndarray:
        return self.step(u, b_current, b_wind)

    def step(
        self, u: np.ndarray, b_current: np.ndarray, b_wind: np.ndarray
    ) -> np.ndarray:
        """
        Computes the next state of the model using the nonlinear dynamics.

        Args:
            u (np.ndarray): (4, ) control input vector.
            where:
                u[0] (float): force of the right stern (rear) thruster
                u[1] (float): force of the left stern (rear) thruster
                u[2] (float): force of the right bow thruster
                u[3] (float): force of the left bow thruster

            b_current (np.ndarray): (3, ) vector of the water current exogenous forces
                    (measured disturbance) acting on the center of mass of the system
                    and expressed in the inertial reference frame.
            where:
                b_current[0] (float): force along x axis
                b_current[1] (float): force along y axis
                b_current[2] (float): moment around z axis

            b_wind (np.ndarray): (3, ) vector of the wind exogenous forces (measured
                    disturbance) acting on the center of mass of the system and
                    expressed in the intertial reference frame.
            where:
                b_wind[0] (float): force along x axis
                b_wind[1] (float): force along y axis
                b_wind[2] (float): moment around z axis

        Returns:
            np.ndarray: updated state vector q.
        """

        if self.integration_method == "euler":
            self.f = self.f + self.thruster_dynamics(self.f, u) * self.dt
            self.q = self.q + self.dynamics(self.q, self.f, b_current, b_wind) * self.dt

        elif self.integration_method == "rk4":
            # thruster dynamics
            k1 = self.thruster_dynamics(self.f, u)
            k2 = self.thruster_dynamics(self.f + k1 * self.dt / 2, u)
            k3 = self.thruster_dynamics(self.f + k2 * self.dt / 2, u)
            k4 = self.thruster_dynamics(self.f + k3 * self.dt, u)
            self.f = self.f + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

            # USV dynamics
            k1 = self.dynamics(self.q, self.f, b_current, b_wind)
            k2 = self.dynamics(self.q + k1 * self.dt / 2, self.f, b_current, b_wind)
            k3 = self.dynamics(self.q + k2 * self.dt / 2, self.f, b_current, b_wind)
            k4 = self.dynamics(self.q + k3 * self.dt, self.f, b_current, b_wind)
            self.q = self.q + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

        else:
            raise ValueError(
                f"Integration method {self.integration_method} not implemented."
            )

        # Return the updated state
        return self.q

    def dynamics(
        self, q: np.ndarray, f: np.ndarray, b_current: np.ndarray, b_wind: np.ndarray
    ) -> np.ndarray:
        """
        SeaCat2 dynamic model ODE.

        Args:
            q (np.ndarray): (6, ) state vector (see class description).
            f (np.ndarray): (4, ) thrusters force vector (see class description).
            b_current (np.ndarray): (3, ) vector of exogenous forces (see step method).
            b_wind (np.ndarray): (3, ) vector of exogenous forces (see step method).

        Returns:
            q_dot (np.ndarray): (6, ) derivative of the state vector.
        """
        # Coriolis matrix update
        # self.C[2, 0] = (self.m + self.Y_vdot) * q[4] + (self.m_xg + self.Y_rdot)*q[5]
        # self.C[2, 1] = -(self.m + self.X_udot) * q[3]
        # self.C[0, 2] = -self.C[2, 0]
        # self.C[1, 2] = -self.C[2, 1]
        self.C = np.zeros((3, 3))  # set C matrix to zero

        # Nonlinear damping matrix update
        self.D_NL[2, 2] = 10 * self.Nr * np.abs(q[5])  # Fossen NL damping estimate
        D = self.D_L + self.D_NL  # pack damping matrices in a single (3, 3) matrix

        # Compute the exogenous inputs in the local (body) reference frame
        # TODO: add crossflow drag to represent the different effect that current and
        # wind have depending on the side of the boat they are acting on (fron or side)
        b_current = R_i2b(q[2]) @ b_current
        b_wind = R_i2b(q[2]) @ b_wind

        # Dynamic equations
        v = q[3:6]  # velocity vector [u_x, u_y, omega]
        x_dot = R_b2i(q[2]) @ v[0:3]
        v_dot = self.M_inv @ (-D @ v - self.C @ v + self.T @ f + b_current + b_wind)

        # Output state derivative
        q_dot = np.concat([x_dot, v_dot])
        return q_dot

    def thruster_dynamics(self, f: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Thrusters dynamic model ODE.

        Args:
            f (np.ndarray): (4, ) thrusters force vector (see class description).
            u (np.ndarray): (4, ) input force vector (see step function).

        Returns:
            f_dot (np.ndarray): (4, ) derivative of the thrusters force vector.
        """
        # Thruster dynamics (1st order system)
        f_dot = (u - f) / self.thrust_delay
        return f_dot

    def added_mass_surge(self, par: Parameters) -> float:
        """
        Computes an approximation of the added mass in surge (i.e., along the x axis)
        for a boat of mass m and length L. The function is adapted from the MSS. Note
        that in the pythonVehicleSimulator the value Xudot = 0.1*mass is used instead.

        Args:
            par (Parameters): parameters object containing the parameters for the model.

        Returns:
            float: added mass in surge [kg]
        """

        rho = 1025  # default density of water [kg/m^3]
        nabla = par.m / rho  # volume displacement

        # compute the added mass in surge using the formula by Söding (1982)
        Xudot = (2.7 * rho * nabla ** (5 / 3)) / (par.l_tot**2)

        return Xudot

    def crossflow_drag(self, par: Parameters) -> np.ndarray:
        """
        Computes the forces acting on the boat due to water currents using strip theory.
        The function is adapted from the PythonVehicleSimulator function crossFlowDrag
        (see python_vehicle_simulator/lib/gnc.py).

        Args:
            par (Parameters): parameters object containing the parameters for the model.

        Returns:
            np.ndarray: a (3, 1) ndarray representing the force vector due to water drag
            acting on the center of mass of the boat
        """
        raise NotImplementedError("The crossflow drag function is not implemented yet.")
        # TODO
