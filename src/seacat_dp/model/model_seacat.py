import numpy as np

from seacat_dp.model import hydrodynamics, wind_dynamics
from seacat_dp.model.model import USVModel
from seacat_dp.model.parameters_seacat import SeaCatParameters
from seacat_dp.utils.transformations import R_b2i


class SeaCatModel(USVModel):
    """
    Nonlinear 2D dynamic model of the SeaCat2 vessel.

    The state vector is defined as a (6, ) vector:
    q (np.ndarray): [x, y, theta, u_x, u_y, omega]

    where:
        x (float): position along x axis (inertial frame)
        y (float): position along y axis (inertial frame)
        theta (float): orientation (angle between body frame and inertial frame)
        u_x (float): velocity along x axis (body frame)
        u_y (float): velocity along y axis (body frame)
        omega (float): angular velocity (derivative of theta)

    The thruster forces are also stored as the state of the thrusters dynamical model,
    forming a (4, ) thrusters state vector:
    u (np.ndarray): [f_l, f_r, f_bow_l, f_bow_r]

    where:
        f_l (float): force to the left stern (rear) thruster
        f_r (float): force to the right stern (rear) thruster
        f_bow_l (float): force to the left bow thruster
        f_bow_r (float): force to the right bow thruster

    Each instance of the class holds a copy of the USV parameters, which can be modified
    to create a model with model mismatch. The parameters are imported from the
    `SeaCatParameters` class in the `seacat_pars.py` module.
    """

    def __init__(self, pars: SeaCatParameters):
        """
        Initialize the nonlinear model of the SeaCat2.

        Args:
            pars (Parameters): Parameters object containing the parameters of the model
        """
        super().__init__(pars)

        # SeaCat control input
        self.u = np.zeros(4)  # control input vector [f_r, f_l, f_bow_r, f_bow_l]

        # Thrust allocation matrix
        self.T = np.array(
            [
                [1, 1, np.sin(pars.alpha), np.sin(pars.alpha)],
                [0, 0, np.cos(pars.alpha), -np.cos(pars.alpha)],
                [
                    pars.b_stern,
                    -pars.b_stern,
                    pars.l_bow * np.cos(pars.alpha) + pars.b_bow * np.sin(pars.alpha),
                    -pars.l_bow * np.cos(pars.alpha) - pars.b_bow * np.sin(pars.alpha),
                ],
            ]
        )  # (3, 4) matrix

        # Thrusters weight matrix
        W_mat = np.diag([1, 1, 10, 10])  # (4, 4) matrix

        # Pseudo-inverse of the thrust matrix (maps CoM forces to thruster forces)
        W_inv = np.linalg.inv(W_mat)  # Inverse of the weight matrix
        self.T_pinv = W_inv @ self.T.T @ np.linalg.inv(self.T @ W_inv @ self.T.T)

        # Thrusters time delay
        self.thrust_delay = np.array(
            [pars.delay_stern, pars.delay_stern, pars.delay_bow, pars.delay_bow]
        )

    def dynamics(
        self,
        q: np.ndarray,
        u: np.ndarray,
        v_current: np.ndarray,
        v_wind: np.ndarray,
        use_nonlinear_dynamics: bool = False,
    ) -> np.ndarray:
        """
        ODEs of the 2D nonlinear point-mass model of the SeaCat2.

        Args:
            q (np.ndarray): (6, ) state vector (see `step` method for more details).
            u (np.ndarray): (4, ) control input (real thrusters forces from the
            thrusters dynamics) (see `step` method for more details).
            v_current (np.ndarray): (3, ) water current speed expressed in the inertial
            frame (see `step` method for more details).
            v_wind (np.ndarray): (3, ) wind speed expressed in the inertial frame (see
            `step` method for more details).
            use_exact_model (bool): whether to use the exact model or the

        Returns:
            q_dot (np.ndarray): (6, ) derivative of the state vector.
        """
        # Coriolis matrix update
        if use_nonlinear_dynamics is True:
            self.C[2, 0] = (self.m + self.Y_vdot) * q[4] + (
                self.m_xg + self.Y_rdot
            ) * q[5]
            self.C[2, 1] = -(self.m + self.X_udot) * q[3]
            self.C[0, 2] = -self.C[2, 0]
            self.C[1, 2] = -self.C[2, 1]

            # Nonlinear damping matrix update
            self.D_NL[2, 2] = 10 * self.Nr * np.abs(q[5])  # Fossen NL damping estimate

        else:
            self.C[:, :] = 0.0
            self.D_NL[:, :] = 0.0
        D = self.D_L + self.D_NL  # pack damping matrices in a single (3, 3) matrix

        # Compute the exogenous inputs in the local (body) reference frame
        b_current = hydrodynamics.crossflow_drag(self.q, self.pars, v_current)  # (3, )
        b_wind = wind_dynamics.wind_load(self.q, v_wind)  # wind load (3, )

        # Dynamic equations
        v = q[3:6]  # velocity vector [u_x, u_y, omega]
        x_dot = R_b2i(q[2]) @ v[0:3]
        v_dot = self.M_inv @ (-D @ v - self.C @ v + self.T @ u + b_current + b_wind)

        # Output state derivative
        q_dot = np.concat([x_dot, v_dot])
        return q_dot

    def thrusters_dynamics(self, u: np.ndarray, u_input: np.ndarray) -> np.ndarray:
        """
        ODEs of the dynamics of the SeaCat2 thrusters.

        Args:
            u (np.ndarray): current control input vector
            u_input (np.ndarray): new control input vector

        Returns:
            u_dot (np.ndarray): time derivative of the control input vector
        """
        # Validate input
        if u.shape != (4,) or u_input.shape != (4,):
            raise ValueError("f and u must be (4, ) vectors.")

        # Thruster dynamics -- actuation delay (1st order system)
        u_dot = (u_input - u) / self.thrust_delay

        return u_dot

    def thrusters_saturation(self, u: np.ndarray) -> np.ndarray:
        """
        Saturates the thrusters forces. The maximum force of the stern thrusters is a
        function of the USV speed.

        Args:
            u (np.ndarray): control input vector

        Returns:
            u (np.ndarray): saturated control input vector
        """

        # Validate input
        if u.shape != (4,) or u.shape != (4,):
            raise ValueError("f and u must be (4, ) vectors.")

        # Stern thruster (function of speed)
        # NOTE: The real characteristics of the thrusters is likely nonlinear, but it
        # is currently unknown, so a linear approximation is used instead. The 2.7
        # coefficient has been computed from the speed/thrust characteristic.
        vel_abs = np.linalg.norm(self.q[3:5])  # absolute velocity
        f_max_forward = self.pars.max_stern_thrust_forward - (
            (self.pars.max_stern_thrust_forward_max_u / 2.7) * vel_abs
        )
        f_max_backward = self.pars.max_stern_thrust_backward - (
            (self.pars.max_stern_thrust_backward_max_u / 2.7) * vel_abs
        )
        u[0] = np.clip(u[0], f_max_backward, f_max_forward)
        u[1] = np.clip(u[1], f_max_backward, f_max_forward)

        # Bow thrusters (not speed dependent)
        u[2] = np.clip(
            u[2], self.pars.max_bow_thrust_backward, self.pars.max_bow_thrust_forward
        )
        u[3] = np.clip(
            u[3], self.pars.max_bow_thrust_backward, self.pars.max_bow_thrust_forward
        )

        return u
