import numpy as np

from seacat_dp.models.parameters import Parameters


class NonlinearModel:
    """
    Nonlinear 2D model class that computes the nonlinear state and output equations.

    The state vector is defined as:
    q = [x, y, theta, u_x, u_y, omega]^T

    where:
        x (float): position along x axis (inertial frame)
        y (float): position along y axis (inertial frame)
        theta (float): orientation (angle between body frame and inertial frame)
        u_x (float): velocity along x axis (body frame)
        u_y (float): velocity along y axis (body frame)
        omega (float): angular velocity
    """

    def __init__(self, par: Parameters, dt: float):
        """
        Initialize the nonlinear model of the SeaCat2.

        Args:
            parameters (Parameters): Parameters object containing the parameters for
            the model.
            dt (float): time step for the model [s].
        """
        # Model time step
        self.dt = dt  # time step [s]

        # Model state
        self.q = np.zeros((6, 1))  # state vector [x, y, theta, u_x, u_y, omega]
        self.f = np.zeros((4, 1))  # force vector [f_r, f_l, f_bow_r, f_bow_l]

        # Mass and inertia matrix
        # Rigid body mass and inertia
        M_RB = np.diag(
            [par.m, par.m, par.m, par.I_xx, par.I_yy, par.I_zz]
        )  # Rigid body inertia matrix
        M_RB[1, 5] = par.xg * par.m
        M_RB[5, 1] = par.xg * par.m

        # Added inertia properties. Multiplication factors are taken from otter.py model
        # in the PythonVehicleSimulator.
        X_udot = added_mass_surge(
            par.m, par.l_tot
        )  # Added mass along x and y, added inertia around z
        Y_vdot = 1.5 * par.m
        N_rdot = 1.7 * par.I_zz
        M_A = np.diag([X_udot, Y_vdot, 0, 0, 0, N_rdot])  # Added inertia matrix

        # Stack all mass and inertia properties in a 6x6 matrix
        self.M = M_RB + M_A  # mass matrix of the SeaCat2

        # Coriolis matrices C, C_A, and C_RB are ignored under assumption of low speeds.

        # Linear damping matrix
        self.DL = np.zeros(3)

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
        )

        # Pseudo-inverse of the thrust matrix (maps CoM forces to thruster forces)
        W_mat = np.diag([1, 1, 10, 10])  # Thrusters weight matrix
        W_inv = np.linalg.inv(W_mat)  # Inverse of the weight matrix
        self.T_pinv = W_inv @ self.T.T @ np.linalg.inv(self.T @ W_inv @ self.T.T)

        # Thrusters time delay
        self.thrust_delay = np.array([0.1, 0.1, 0.1, 0.1])  # thrust time delay [s]

    def __call__(
        self,
        u: np.ndarray,
        b: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        return self.step(u, b, w)

    def step(self, u: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Perform one step of the model nonlinear ODE.

        Args:
            u (np.ndarray): (4, 1) control input vector.

            where:
                u[0] (float): force of the right rear thruster
                u[1] (float): force of the left rear thruster
                u[2] (float): force of the right bow thruster
                u[3] (float): force of the left bow thruster

            b (np.ndarray): (3, 1) vector of external forces acting on the system
                            (measured disturbances).
            where:
                b[0] (float): force along x axis (inertial frame)
                b[1] (float): force along y axis (inertial frame)
                b[2] (float): moment around z axis (inertial frame)

            w (np.ndarray): (6, 1) vector of measurement noise (white Gaussian noise).
                            Indexes match the state vector q.

        Returns:
            q_plus (np.ndarray): (6, 1) updated state vector. Indexes match the state q.
        """

        # force vector time delay
        for i in range(4):
            self.f[i] = (
                self.f[i] + ((u[i] - self.f[i]) / self.thrust_delay[i]) * self.dt
            )

        # update state vector # TODO
        self.q = np.zeros((6, 1))

        return self.q


def added_mass_surge(mass: float, length: float) -> float:
    """
    Computes an approximation of the added mass in surge (i.e., along the x axis) for a
    boat of  mass m and length L. The function is adapted from the MSS. Note that in the
    pythonVehicleSimulator the value Xudot = 0.1*mass is used instead.

    Args:
        mass (float): boat mass in [kg]
        length (float): total boat length in [m]

    Returns:
        float: added mass in surge [kg]
    """

    rho = 1025  # default density of water [kg/m^3]
    nabla = mass / rho  # volume displacement

    # compute the added mass in surge using the formula by Söding (1982)
    Xudot = (2.7 * rho * nabla ** (5 / 3)) / (length**2)

    return Xudot


def crossflow_drag() -> np.ndarray:
    """
    Computes the forces acting on the boat due to water currents using strip theory.
    The function is adapted from the PythonVehicleSimulator function crossFlowDrag (see
    python_vehicle_simulator/lib/gnc.py).

    Args:
        None

    Returns:
        np.ndarray: a (3, 1) ndarray representing the force vector due to water drag
        acting on the center of mass of the boat
    """
    # TODO
