import warnings

import casadi as ca
import numpy as np

from seacat_dp.control.mpc import Mpc


class NonlinearMpc(Mpc):
    """
    Nonlinear Model Predictive Control (MPC) class for controlling a the SeaCat2.
    The MPC model is nonlinear in the A state matrix, which contains a rotation matrix.
    Additional nonlinearities can also be added
    """

    def __init__(self):
        """
        Initialize the MPC controller.
        """
        super().__init__()

        # Initialize the OCP problem
        self.ocp = ca.Opti()
        self.sol = None  # Solution to the OCP problem. To access stats sol.stats()

        # Initialize the solver options
        self.solver_options: dict = {
            "max_iter": 1000,  # Maximum iterations
            "ipopt.max_iter": 1000,
            "print_level": 0,  # 0-5 (0 = silent, 5 = verbose)
            "print_time": 0,
            "tol": 1e-6,  # Optimality tolerance
            "acceptable_tol": 1e-4,  # Acceptable tolerance for early termination
            "linear_solver": "mumps",  # Recommended for most problems
        }
        self.ocp_options: dict = {
            "expand": True,  # Faster runtime
            "error_on_fail": True,  # Raise exception if fails
            # "print_time": 0,  # Suppress compiler output
        }

        # Initialize OCP variables
        self.q: ca.Opti.variable = self.ocp.variable(self.n_q, self.N + 1)  # (n_q, N+1)
        self.u: ca.Opti.variable = self.ocp.variable(self.n_u, self.N)  # (n_u, N)
        self.cost_function: ca.Opti.variable = self.ocp.variable(1, 1)  # Cost function

        # Initialize OCP parameters
        self.q_0 = self.ocp.parameter(self.n_q, 1)  # Initial state (n_q, 1)
        self.q_ref = self.ocp.parameter(self.n_q, self.N + 1)  # Reference (n_q, N+1)
        self.b_curr = self.ocp.parameter(self.n_q / 2, 1)
        self.b_wind = self.ocp.parameter(self.n_q / 2, 1)

        # Check model discretization method
        self.discretization_options = ["euler", "rk4"]  # class-specific options
        if self.discretization_method == "zoh":
            raise NotImplementedError(
                "Zero-order hold (ZOH) discretization is not available in the "
                "nonlinear MPC class. Pleas use one of the following methods: "
                f"{self.discretization_options}."
            )
        elif self.discretization_method not in self.discretization_options:
            raise ValueError(
                f"Discretization method {self.discretization_method} is not supported "
                "by the nonlinear MPC class. Please use one of the following methods: "
                f"{self.discretization_options}."
            )

        # Model and model parameters
        self.M_inv: np.ndarray = None
        self.D_L: np.ndarray = None
        self.T: np.ndarray = None
        self.D_NL: np.ndarray = None  # Nonlinear damping (optional)
        self.C: np.ndarray = None  # Coriolis matrix (optional)

    def _set_model(
        self,
        M_inv: np.ndarray,
        D_L: np.ndarray,
        T: np.ndarray,
        **kwargs: dict,
    ):
        """
        Defines the model parameters for the MPC nonlinear prediction model.
        """
        # Parse kwargs
        for key, value in kwargs.items():
            if key == "D_NL":
                if not isinstance(value, np.ndarray):
                    raise TypeError(
                        f"Expected 'D_NL' to be a numpy array, got {type(value)}."
                    )
                if value.shape != (self.n_q, self.n_q):
                    raise ValueError(
                        f"Expected 'D_NL' to be of shape ({self.n_q}, {self.n_q}), "
                        f"got {value.shape}."
                    )
                self.D_NL = value
            elif key == "C":
                if not isinstance(value, np.ndarray):
                    raise TypeError(
                        f"Expected 'C' to be a numpy array, got {type(value)}."
                    )
                if value.shape != (self.n_q, self.n_q):
                    raise ValueError(
                        f"Expected 'C' to be of shape ({self.n_q}, {self.n_q}), "
                        f"got {value.shape}."
                    )
                self.C = value
                warnings.warn(
                    "The 'C' keyword is currently not used in the nonlinear MPC class."
                    "The value of the matrix will be stored but not used in the "
                    "dynamic model",
                    UserWarning,
                )
            elif key == "phi":
                warnings.warn(
                    "The 'phi' keyword is not used in the nonlinear MPC class and it "
                    "will be ignored during the model definition.",
                    UserWarning,
                )
            else:
                raise ValueError(
                    f"Unknown keyword argument '{key}' for nonlinear MPC model."
                )

        # Save the model parameters
        # NOTE: the type and shape of these parameters is checked in super().set_model()
        self.M_inv = M_inv
        self.D_L = D_L
        self.T = T

    def _dqdt(
        self,
        q: ca.SX | ca.MX,
        u: ca.SX | ca.MX,
        b_curr: ca.SX | ca.MX,
        b_wind: ca.SX | ca.MX,
    ) -> ca.SX | ca.MX:
        """
        Nonlinear continuous-time dynamics function for the MPC controller.
            dqdt = f(q, u, b_curr, b_wind)

        Args:
            q (ca.SX or ca.MX): State vector (n_q, ).
            u (ca.SX or ca.MX): Control input vector (n_u, ).
            b (ca.SX or ca.MX): Lumped exogenous input vector (n_q, ) in body frame.

        Returns:
            dq (ca.SX or ca.MX): Time derivative of the state vector (n_q, ).
        """

        # NOTE: some of the indexes are currently hardcoded, as the model is assumed to
        # have 6 states (3 position, 3 velocity) and 4 controls (2 stern, 2 bow).

        v = q[3:6]  # Extract velocity (vx, vy, vz)

        # Build rotation matrix from the yaw angle (q[2])
        R = ca.MX(
            [
                [ca.cos(q[2]), -ca.sin(q[2]), 0],
                [ca.sin(q[2]), ca.cos(q[2]), 0],
                [0, 0, 1],
            ]
        )

        if self.D_NL is not None:
            D = self.D_L + self.D_NL @ ca.fabs(v)
        else:
            D = self.D_L

        if self.C is not None:
            raise NotImplementedError(
                "Coriolis matrix (C) is not implemented in the nonlinear MPC class. "
                "Please remove it from the model parameters."
            )
        else:
            # NOTE: this matrix may do weird things if the number of states is odd. At
            # the moment it is not worth updating it as the number of states is even.
            C = np.zeros((self.n_q / 2, self.n_q / 2))

        # Nonlinear dynamics
        dx = R @ v
        dv = self.M_inv @ (-D @ v - C @ v + self.T @ u + b_curr + b_wind)

        return ca.vertcat(dx, dv)

    def _init_ocp(self):
        """
        Initialize the OCP problem for the nonlinear MPC controller using the casadi
        optimization framework.
        """

        ## Constraints
        # Dynamics
        for k in range(self.N):
            match self.discretization_method:
                case "euler":
                    q_next = self.q[:, k] + self.dt * self._dqdt(
                        self.q[:, k],
                        self.u[:, k],
                        self.b_curr,
                        self.b_wind,
                    )

                case "rk4":
                    k1 = self._dqdt(
                        self.q[:, k],
                        self.u[:, k],
                        self.b_curr,
                        self.b_wind,
                    )
                    k2 = self._dqdt(
                        self.q[:, k] + 0.5 * self.dt * k1,
                        self.u[:, k],
                        self.b_curr,
                        self.b_wind,
                    )
                    k3 = self._dqdt(
                        self.q[:, k] + 0.5 * self.dt * k2,
                        self.u[:, k],
                        self.b_curr,
                        self.b_wind,
                    )
                    k4 = self._dqdt(
                        self.q[:, k] + self.dt * k3,
                        self.u[:, k],
                        self.b_curr,
                        self.b_wind,
                    )
                    q_next = self.q[:, k] + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            self.ocp.subject_to(self.q[:, k + 1] == q_next)

        # Initial state constraint
        self.ocp.subject_to(self.q[:, 0] == self.q_0)

        # Input constraints
        if self.u_min is not None or self.u_max is not None:
            for k in range(self.N):
                self.ocp.subject_to(
                    self.ocp.bounded(
                        self.u[:, k],
                        self.u_min,
                        self.u_max,
                    )
                )

        # Input rate constraints
        if self.u_rate_min is not None and self.u_rate_max is not None:
            for k in range(1, self.N):
                self.ocp.subject_to(
                    self.ocp.bounded(
                        self.u[:, k:] - self.u[:, k - 1],
                        self.u_rate_min,
                        self.u_rate_max,
                    )
                )

        # Maximum input constraint
        if self.max_tot_u is not None:
            for k in range(self.N):
                self.ocp.subject_to(ca.norm_1(self.u[:, k]) <= self.max_tot_u)

            # Alternative with sum1 and fabs:
            # for k in range(self.N):
            #     self.ocp.subject_to(ca.sum1(ca.fabs(self.u[:, k])) <= self.max_tot_u)

            # Alternative implementation with slack variables:
            # for k in range(self.N):
            #     u_k = self.u[:, k]
            #     s = self.ocp.variable(4)  # Slack variables
            #     self.ocp.subject_to(s >= u_k)
            #     self.ocp.subject_to(s >= -u_k)
            #     self.ocp.subject_to(ca.sum1(s) <= self.u_max)

        # State constraints
        if self.q_min is not None or self.q_max is not None:
            for k in range(self.N + 1):
                self.ocp.subject_to(
                    self.ocp.bounded(
                        self.q[:, k],
                        self.q_min,
                        self.q_max,
                    )
                )

        # State rate constraints
        if self.q_rate_min is not None and self.q_rate_max is not None:
            for k in range(1, self.N + 1):
                self.ocp.subject_to(
                    self.ocp.bounded(
                        self.q[:, k] - self.q[:, k - 1],
                        self.q_rate_min,
                        self.q_rate_max,
                    )
                )

        # Terminal state constraint (bounds)
        if self.q_terminal_min is not None and self.q_terminal_max is not None:
            self.ocp.subject_to(
                self.ocp.bounded(
                    self.q[:, -1],
                    self.q_terminal_min,
                    self.q_terminal_max,
                )
            )

        ## Cost function
        # State cost
        for k in range(self.N):
            self.cost_function += ca.mtimes(
                (self.q[:, k] - self.q_ref[:, k]).T,
                self.Q,
                (self.q[:, k] - self.q_ref[:, k]),
            )

        # Input cost
        for k in range(self.N):
            self.cost_function += ca.mtimes(
                self.u[:, k].T,
                self.R,
                self.u[:, k],
            )

        # Terminal cost
        self.cost_function += ca.mtimes(
            (self.q[:, -1] - self.q_ref[:, -1]).T,
            self.P,
            (self.q[:, -1] - self.q_ref[:, -1]),
        )

        # Define the objective and set the solver options
        self.ocp.minimize(self.cost_function)
        self.ocp.solver("ipopt", self.ocp_options, self.solver_options)

    def _solve(self, q_0, q_ref, b_curr, b_wind):
        """
        Solve the OCP problem for the given initial state and reference state.
        See super().solve() for more details on the arguments and outputs. The input
        arguments of _solve() are assumed to have been already parsed and validated.

        Args:
            q0 (np.ndarray): Initial state (n_q, ).
            q_ref (np.ndarray): Reference state sequence (n_q, N+1).
            b_curr (np.ndarray): Current disturbance vector in body frame (3, ).
            b_wind (np.ndarray): Wind disturbance vector in body frame (3, ).
        """
        # Set the initial state
        self.ocp.set_value(self.q_0, q_0)
        self.ocp.set_value(self.q_ref, q_ref)

        # Set the disturbances
        self.ocp.set_value(self.b_curr, np.hstack([np.zeros(3), b_curr]))  # (6, )
        self.ocp.set_value(self.b_wind, np.hstack([np.zeros(3), b_wind]))  # (6, )

        # Solve the OCP problem
        self.sol = self.ocp.solve()
        if self.sol.stats()["success"] is not True:
            raise RuntimeError("MPC optimization failed: " + self.sol.stats()["status"])

        # Return the OCP solution
        u_opt = self.sol.value(self.u)
        q_pred = self.sol.value(self.q)
        cost = self.sol.value(self.cost_function)
        return u_opt, q_pred, cost

    def _cost(self) -> tuple[float, float, float, float]:
        """
        Compute the cost function for the MPC problem. See super().cost() for more
        details on this function and its outputs.

        Returns:
            - c_tot (float): The total cost of the MPC problem.
            - c_state (float): The cost associated with the state trajectory.
            - c_input (float): The cost associated with the control inputs.
            - c_terminal (float): The terminal cost at the end of the horizon.

        Raises:
            RuntimeError: If the OCP is not ready or has not been solved yet.
        """
        # Check if the OCP is ready and has been solved
        if self.ocp_ready is False or self.sol is None:
            raise RuntimeError("OCP is not ready or has not been solved yet.")

        # TODO: Implement the cost function computation
        c_tot = 0.0
        c_state = 0.0
        c_input = 0.0
        c_terminal = 0.0

        return c_tot, c_state, c_input, c_terminal
