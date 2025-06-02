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
    f (np.ndarray): [f_l, f_r, f_bow_l, f_bow_r]

    where:
        f_l (float): force to the left stern (rear) thruster
        f_r (float): force to the right stern (rear) thruster
        f_bow_l (float): force to the left bow thruster
        f_bow_r (float): force to the right bow thruster


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
        # NOTE: The Coriolis matrix depends on the current USV state and therefore it is
        # updated at each time step in the dynamics function.
        self.C = np.zeros((3, 3))  # (3, 3) matrix

        # Linear damping matrix
        # self.Xu = 24.4 * par.g / par.u_max
        self.Xu = 1000  # set Xu to a constant value
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
                    par.b_stern,
                    -par.b_stern,
                    par.l_bow * np.cos(par.alpha) + par.b_bow * np.sin(par.alpha),
                    -par.l_bow * np.cos(par.alpha) - par.b_bow * np.sin(par.alpha),
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

        # Draft from Fossen eq
        self.draft_fossen = par.m / (
            2 * par.rho * par.c_b_pontoon * par.l_pontoon * par.b_pontoon
        )  # draft estimated from approximated submerged volume (submerged 'box')

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
        v_current: np.ndarray,
        v_wind: np.ndarray,
    ) -> np.ndarray:
        return self.step(u, v_current, v_wind)

    def step(
        self, u: np.ndarray, v_current: np.ndarray, v_wind: np.ndarray
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
                v_current[2] (float): rotational speed around z axis (positive for CW)

            v_wind (np.ndarray): (3, ) vector of the wind speed (measured or estimated
                    disturbance) acting on the boat [m/s]. The vector is expressed in
                    the intertial reference frame.
            where:
                v_wind[0] (float): wind speed along x axis
                v_wind[1] (float): wind speed along y axis
                v_wind[2] (float): rotational speed around z axis (positive for CW)

        Returns:
            np.ndarray: updated state vector q.
        """

        if self.integration_method == "euler":
            # thruster dynamics
            self.f = self.f + self.thruster_dynamics(self.f, u) * self.dt

            # USV dynamics
            self.q = self.q + self.dynamics(self.q, self.f, v_current, v_wind) * self.dt

        elif self.integration_method == "rk4":
            # thruster dynamics
            k1 = self.thruster_dynamics(self.f, u)
            k2 = self.thruster_dynamics(self.f + k1 * self.dt / 2, u)
            k3 = self.thruster_dynamics(self.f + k2 * self.dt / 2, u)
            k4 = self.thruster_dynamics(self.f + k3 * self.dt, u)
            self.f = self.f + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

            # USV dynamics
            k1 = self.dynamics(self.q, self.f, v_current, v_wind)
            k2 = self.dynamics(self.q + k1 * self.dt / 2, self.f, v_current, v_wind)
            k3 = self.dynamics(self.q + k2 * self.dt / 2, self.f, v_current, v_wind)
            k4 = self.dynamics(self.q + k3 * self.dt, self.f, v_current, v_wind)
            self.q = self.q + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

        else:
            raise ValueError(
                f"Integration method {self.integration_method} not implemented."
            )

        # Return the updated state
        return self.q

    def dynamics(
        self, q: np.ndarray, f: np.ndarray, v_current: np.ndarray, v_wind: np.ndarray
    ) -> np.ndarray:
        """
        SeaCat2 dynamic model ODE.

        Args:
            q (np.ndarray): (6, ) state vector (see class description).
            f (np.ndarray): (4, ) thrusters force vector (see class description).
            v_current (np.ndarray): (3, ) water current speed expressed in the inertial
            frame (see step method for more details).
            v_wind (np.ndarray): (3, ) wind speed expressed in the inertial frame (see
            step method for more details).

        Returns:
            q_dot (np.ndarray): (6, ) derivative of the state vector.
        """
        # Coriolis matrix update
        self.C[2, 0] = (self.m + self.Y_vdot) * q[4] + (self.m_xg + self.Y_rdot) * q[5]
        self.C[2, 1] = -(self.m + self.X_udot) * q[3]
        self.C[0, 2] = -self.C[2, 0]
        self.C[1, 2] = -self.C[2, 1]

        # Nonlinear damping matrix update
        self.D_NL[2, 2] = 10 * self.Nr * np.abs(q[5])  # Fossen NL damping estimate
        D = self.D_L + self.D_NL  # pack damping matrices in a single (3, 3) matrix

        # Compute the exogenous inputs in the local (body) reference frame
        b_current = self.crossflow_drag(v_current)  # water current drag
        b_wind = self.wind_load(v_wind)  # wind load

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
        # TODO: thrusters dynamics also depends on the current speed of the USV

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

    def crossflow_drag(self, v_curr: np.ndarray) -> np.ndarray:
        """
        Computes the forces acting on the boat due to water currents using strip theory.
        The function is adapted from the PythonVehicleSimulator function crossFlowDrag
        (see /lib/gnc.py and /vehicles/otter.py).

        Note: the nominal draft is used instead of the approximated one estimated from
        the 'box' submerged volume. This should result in a more accurate drag value.

        Args:
            v_curr (np.ndarray): (3, ) vector of the water current velocity expressed in
                the inertial reference frame.

        Returns:
            tau (np.ndarray): a (3, ) ndarray representing the force vector due to water
            drag acting on the center of mass of the boat. The force vector is expressed
            in the body reference frame.
        """
        # validate input and transform to body frame
        if v_curr.shape != (3,):
            raise ValueError("v_curr must be a (3, ) vector.")
        v_curr_b = R_i2b(self.q[2]) @ v_curr  # transform to body frame

        # initialize strip theory parameters
        c_d = self.hoerner()  # cross-flow drag coefficient
        n_strips = 20  # number of strips
        dx = self.par.l_tot / n_strips
        x = -self.par.l_tot / 2  # strip position along the x axis

        # compute the cross-flow velocity
        v_r_y = self.q[4] - v_curr_b[1]  # relative velocity along y axis
        v_cf = np.abs(v_r_y) * v_r_y  # cross-flow velocity

        # initialize force vector
        tau = np.zeros(3)  # force vector due to water drag acting on the center of mass

        # compute the forces acting on each strip
        for _ in range(n_strips + 1):

            # CHECKME: add some effect of the current on tau[0]?
            # CHECKME: is tau[2] always zero in this implementation?

            tau[0] += 0
            tau[1] += -0.5 * self.par.rho * c_d * self.par.draft * dx * v_cf
            tau[2] += -0.5 * self.par.rho * c_d * self.par.draft * dx * v_cf * x

            x += dx  # move to the next strip

        return tau

    def hoerner(self) -> float:
        """
        Helper function for crossflow_drag.

        Computes the 2D Hoerner cross-flow coefficient as a function of beam and draft
        values. The data is interpolated to find the cross-flow coefficient for any
        beam/draft pair. The function is adapted from the PythonVehicleSimulator (see
        see /lib/gnc.py/Hoerner).

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
        x = self.par.b_tot / (2 * self.par.draft)  # B/2T
        h_coeff = np.interp(x, DATA_B2T, DATA_CD)

        return h_coeff

    def wind_load(self, v_wind: np.ndarray) -> np.ndarray:
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
        # validate input and transform to body frame
        if v_wind.shape != (3,):
            raise ValueError("v_wind must be a (3, ) vector.")

        # TODO: implement a more realistic wind load model
        # NOTE: the wind load should be represented in the local (body) reference frame
        # by rotating it with something like v_wind_b = R_i2b(self.q[2]) @ v_wind

        b_wind = np.zeros(3)  # initialize wind load vector

        return b_wind
