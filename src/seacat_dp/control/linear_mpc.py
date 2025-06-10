import warnings

import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete

from seacat_dp.utils.transformations import R_b2i


class LinearMpc:
    """
    Model Predictive Control (MPC) class for controlling a system. The class assumes a
    LTI system model and implements a basic MPC algorithm.
    """

    def __init__(self):
        """
        Initialize the MPC controller. All the MPC variables are initialized to zero.
        To set the parameters of the MPC, use the dedicated setter methods.

        The dynamic model is of the form:
            q+ = q + dot{q}*dt
        with the state derivative given by:
            dot{q} = Aq + Bu + Eb
        where q is the state vector, u is the control input, and b is the known
        exogenous input (measured disturbance).
        """
        # MPC parameters
        self.N: int  # control horizon
        self.dt: float  # control time step [s]
        self.discretization_method: str = "euler"  # {"euler", "zoh"}

        # Continuous-time model matrices
        self.Ac: cp.Parameter = cp.Parameter((6, 6), value=np.zeros((6, 6)))
        self.Bc: np.ndarray
        self.Ec: np.ndarray

        # Discrete-time model matrices
        self.A: cp.Parameter = cp.Parameter((6, 6), value=np.zeros((6, 6)))
        self.B: np.ndarray
        self.E: np.ndarray

        # Cost matrices
        self.Q: np.ndarray
        self.R: np.ndarray
        self.P: np.ndarray

        # Input and state bounds (optional)
        self.u_min: np.ndarray = None
        self.u_max: np.ndarray = None
        self.delta_u_max: np.ndarray = None
        self.delta_u_min: np.ndarray = None
        self.q_min: np.ndarray = None
        self.q_max: np.ndarray = None

        # Optimization variables
        self.u: cp.Variable
        self.q: cp.Variable

        # MPC inputs
        self.q0: cp.Parameter = cp.Parameter((6,), value=np.zeros(6))
        self.q_ref: cp.Parameter = cp.Parameter((6,), value=np.zeros(6))

        # Disturbances (exogenous inputs)
        self.b_curr: cp.Parameter = cp.Parameter((6,), value=np.zeros(6))
        self.b_wind: cp.Parameter = cp.Parameter((6,), value=np.zeros(6))

        # Optimization problem
        self.prob: cp.Problem  # cvxpy optimization problem to be solved
        self.prob_ready: bool = False  # Flag to check if the OCP is initialized
        self.init_ocp_automatically: bool = False  # Toggle automatic initialization
        self.verbose: bool = False

    def __call__(
        self,
        q0: np.ndarray,
        q_ref: np.ndarray,
        b_curr: np.ndarray,
        b_wind: np.ndarray,
    ) -> np.ndarray:
        """
        Call the MPC controller to compute the control action. See the `solve` method
        for more details.

        Args:
            q0 (np.ndarray): The current state of the system (6, ).
            q_ref (np.ndarray): The desired reference state (6, ).
            b_curr (np.ndarray): The water current disturbance (3, ).
            b_wind (np.ndarray): The wind disturbance (3, ).

        Returns:
            u (np.ndarray): The computed control action.
        """
        # Call the solver function to compute the control action
        u = self.solve(q0, q_ref, b_curr, b_wind)
        return u

    def _warn_if_initialized(self):
        if self.prob_ready:
            if self.init_ocp_automatically:
                warnings.warn(
                    "Warning: MPC problem is already initialized. Reinitializing it to "
                    "apply the changes.",
                    UserWarning,
                )
                self.init_ocp()
            else:
                warnings.warn(
                    "MPC problem is already initialized and init_ocp_automatically is "
                    "set to False. For the changes to take effect, the problem must be "
                    "reinitialized manually by calling init_ocp() again.",
                    UserWarning,
                )

    def set_horizon(self, N: int):
        """
        Set the prediction horizon for the MPC.

        Args:
            N (int): The prediction horizon (int).
        """
        # Validate the input
        if not isinstance(N, int) or N <= 0:
            raise ValueError("Prediction horizon N must be a positive integer.")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the prediction horizon
        self.N = N

    def set_dt(self, dt: float):
        """
        Set the control time step for the MPC. This is the time step at which the MPC
        controller computes the control action, and is also used as prediction step in
        the MPC model. This time step is, in general, different from the simulation
        time step.

        Args:
            dt (float): The control time step [s].
        """
        # Validate the input
        if not isinstance(dt, (float, int)) or dt <= 0:
            raise ValueError("Control time step dt must be a positive float or int.")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the control time step
        self.dt = dt

    def set_discretization_method(self, method: str):
        """
        Set the discretization method for the MPC. The method is used to compute the
        discrete model matrices from the continuous model matrices.

        Args:
            method (str): The discretization method to use ("euler" or "zoh").
        """
        # Validate the input
        if method not in ["forward_euler", "euler", "fe", "zoh"]:
            raise ValueError("Discretization method must be 'euler' or 'zoh'.")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the discretization method
        self.discretization_method = method

    def set_weights(self, Q: np.ndarray, R: np.ndarray, P: np.ndarray):
        """
        Set the cost matrices for the MPC.

        Args:
            Q (np.ndarray): The state cost matrix (6, 6).
            R (np.ndarray): The input cost matrix (4, 4).
            P (np.ndarray): The terminal state cost matrix (6, 6).
        """
        # Validate the inputs
        if not Q.shape == (6, 6):
            raise ValueError("State cost matrix Q must be of shape (6, 6).")
        if not R.shape == (4, 4):
            raise ValueError("Input cost matrix R must be of shape (4, 4).")
        if not P.shape == (6, 6):
            raise ValueError("Terminal state cost matrix P must be of shape (6, 6).")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the cost matrices
        self.Q = Q
        self.R = R
        self.P = P

    def set_input_bounds(self, u_min: np.ndarray, u_max: np.ndarray):
        """
        Set the input bounds for the MPC.

        Args:
            u_min (np.ndarray): The minimum input bounds (4, ).
            u_max (np.ndarray): The maximum input bounds (4, ).
        """
        # Validate the inputs
        if not u_min.shape == (4,):
            raise ValueError("Minimum input bounds u_min must be of shape (4, ).")
        if not u_max.shape == (4,):
            raise ValueError("Maximum input bounds u_max must be of shape (4, ).")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the input bounds
        self.u_min = u_min
        self.u_max = u_max

    def set_input_rate_bounds(self, delta_u_min: np.ndarray, delta_u_max: np.ndarray):
        """
        Set the input rate bounds for the MPC. These bounds are used to limit the rate
        of change of the control inputs, and take into account the actuaor dynamics in
        the control problem.

        Args:
            delta_u_min (np.ndarray): The minimum input rate bounds (4, ).
            delta_u_max (np.ndarray): The maximum input rate bounds (4, ).

        Raises:
            ValueError: If the input shapes are not as expected.
        """
        # Validate the inputs
        if not delta_u_min.shape == (4,):
            raise ValueError(
                "Minimum input rate bounds delta_u_min must be of shape (4, )."
            )
        if not delta_u_max.shape == (4,):
            raise ValueError(
                "Maximum input rate bounds delta_u_max must be of shape (4, )."
            )

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the input rate bounds
        self.delta_u_min = delta_u_min
        self.delta_u_max = delta_u_max

    def set_state_bounds(self, q_min: np.ndarray, q_max: np.ndarray):
        """
        Set the state bounds for the MPC.

        Args:
            q_min (np.ndarray): The minimum state bounds (6, ).
            q_max (np.ndarray): The maximum state bounds (6, ).
        """
        raise NotImplementedError("State bounds are not implemented yet.")

    def set_model(
        self, M_inv: np.ndarray, D_L: np.ndarray, T: np.ndarray, phi: float = 0.0
    ):
        """
        Set the MPC prediction model.

        Args:
            M_inv: (np.ndarray): Inverse of the inertia matrix (3, 3).
            D_L (np.ndarray): Linear damping matrix (3, 3).
            T (np.ndarray): Thrust matrix (3, 4).
            phi (float): The angle around which to linearize the state matrix A [rad].
        """
        # Validate the input
        if not M_inv.shape == (3, 3):
            raise ValueError("Input matrix M_inv must be of shape (3, 3).")
        if not D_L.shape == (3, 3):
            raise ValueError("Damping matrix D_L must be of shape (3, 3).")
        if not T.shape == (3, 4):
            raise ValueError("Thrust matrix T must be of shape (3, 4).")
        if not isinstance(phi, float):
            raise ValueError("Heading angle phi must be a float.")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Define the continuous model matrices
        I = np.eye(6)
        self.Ac = np.zeros((6, 6))
        self.Ac[0:3, 3:6] = R_b2i(phi)
        self.Ac[3:6, 3:6] = -M_inv @ D_L
        self.Bc = np.zeros((6, 4))
        self.Bc[3:6, :] = M_inv @ T
        self.Ec = np.zeros((6, 6))
        self.Ec[3:6, 3:6] = M_inv

        # Set the discrete model matrices using the Forward Euler method
        match self.discretization_method:

            case "zoh":

                # Discretize the system
                BEc = np.hstack([self.Bc, self.Ec])  # [B | E]
                Ad, BEd, _, _, _ = cont2discrete(
                    (self.Ac, BEc, I, 0), self.dt, method="zoh"
                )

                # Assign matrices to the MPC
                self.A = Ad
                self.B = BEd[:, : self.Bc.shape[1]]
                self.E = BEd[:, self.Bc.shape[1] :]

            case "forward_euler" | "euler" | "fe":

                # Build the discrete model matrices using Forward Euler
                self.A = I + self.Ac * self.dt  # A_d = I + A_c * dt
                self.B = self.Bc * self.dt  # B_d = B_c * dt
                self.E = self.Ec * self.dt  # E_d = E_c * dt

    def update_model(self, phi: float):
        """
        Update the A state matrix for the MPC by linearizing it around the angle phi.

        Args:
            phi (float): The angle around which to linearize the state matrix A [rad].
        """
        # Validate the inputs
        if not isinstance(phi, float):
            raise ValueError("Heading angle phi must be a float.")
        if not -np.pi <= phi < np.pi:
            raise ValueError("Heading angle phi must be in the range [-pi, pi].")

        # Update the A matrix
        # NOTE: the update is performed on the continuous-time state matrix A_c, which
        # then propagates the change to the discrete-time state matrix A depending on
        # the discretization method used.
        self.Ac[0:3, 3:6] = R_b2i(phi)

    def init_ocp(self) -> None:
        """
        Initialize the MPC optimization problem. This method is called to set up the
        cvxpy problem before solving it.
        """
        # Check if all the required parameters are set
        if self.N is None:
            raise ValueError(
                "Prediction horizon N is not set. Call set_horizon() first."
            )
        if self.dt is None:
            raise ValueError("Control time step dt is not set. Call set_dt() first.")
        if self.A is None or self.B is None or self.E is None:
            raise ValueError(
                "Model matrices A, B, and E are not set. Call set_model() first."
            )
        if self.Q is None or self.R is None or self.P is None:
            raise ValueError(
                "Cost matrices Q, R, and P are not set. Call set_weights() first."
            )
        if self.u_min is None or self.u_max is None:
            raise ValueError(
                "Input bounds u_min and u_max are not set. Call set_input_bounds() "
                "first."
            )
        if self.delta_u_max is None or self.delta_u_min is None:
            raise ValueError(
                "Input rate bounds delta_u_max and delta_u_min are not set. "
                "Call set_input_bounds() first."
            )

        # Initialize the optimization variables
        # NOTE: we assume that control horizon and prediction horizon are the same (N)
        self.u = cp.Variable((4, self.N))
        self.q = cp.Variable((6, self.N + 1))

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.q[:, k] - self.q_ref, self.Q)  # state cost
            cost += cp.quad_form(self.u[:, k], self.R)  # input cost
        cost += cp.quad_form(self.q[:, self.N] - self.q_ref, self.P)  # Terminal cost

        # Initial condition
        constraints = [self.q[:, 0] == self.q0]

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            for k in range(self.N):
                constraints += [
                    self.u[:, k] >= self.u_min,
                    self.u[:, k] <= self.u_max,
                ]

        # Input rate constraints
        if self.delta_u_min is not None and self.delta_u_max is not None:
            for k in range(self.N - 1):
                constraints += [
                    self.u[:, k + 1] - self.u[:, k] <= self.delta_u_max,
                    self.u[:, k + 1] - self.u[:, k] >= self.delta_u_min,
                ]

        # State constraints
        if self.q_min is not None and self.q_max is not None:
            raise NotImplementedError("State constraints are not implemented yet. ")
            # TODO: implement state constraints

        # Dynamics
        for k in range(self.N):
            constraints += [
                self.q[:, k + 1]
                == (
                    self.A @ self.q[:, k]
                    + self.B @ self.u[:, k]
                    + self.E @ (self.b_curr + self.b_wind)  # stationary over horizon
                ),
            ]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        self.prob_ready = True

    def solve(
        self,
        q0: np.ndarray,
        q_ref: np.ndarray,
        b_curr: np.ndarray,
        b_wind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the control action based on the current state and reference.

        Args:
            q0 (np.ndarray): The current state of the system (6, ).
            q_ref (np.ndarray): The desired reference state (6, ).
            b_curr (np.ndarray): The current force (3, ). The force is a measured or
            estimated exogenous input expressed in the body reference frame, and is
            assumed to be stationary over the prediction horizon.
            b_wind (np.ndarray): The wind force (3, ). The force is a measured or
            estimated exogenous input expressed in the body reference frame, and is
            assumed to be stationary over the prediction horizon.

        Raises:
            ValueError: If the input shapes are not as expected.
            RuntimeError: If the MPC optimization problem is not initialized.

        Returns:
            u (np.ndarray): The computed control action (4, N).
            q (np.ndarray): The predicted state sequence (6, N + 1).
            cost (float): The cost of the MPC solution.
        """
        # TO DO LIST:
        # TODO: make q_ref a sequence of length N instead of a single value (to enable
        # trajectory tracking in addition to setpoint tracking) --> q_ref would then be
        # either a (6, N) matrix or a (6, ) vector (the former for trajectory tracking,
        # the latter for setpoint tracking/regulation)

        # Validate the inputs
        if not q0.shape == (6,):
            raise ValueError("Current state q must be of shape (6, ).")
        if not q_ref.shape == (6,) or q_ref.shape == (6, self.N):
            raise ValueError("Reference state q_ref must be of shape (6, ).")
        if not b_curr.shape == (3,):
            raise ValueError("Current force b_curr must be of shape (3, ).")
        if not b_curr.shape == (3,):
            raise ValueError("Current force b_curr must be of shape (3, ).")

        # Check if the optimization problem is initialized
        if not self.prob_ready:
            raise RuntimeError(
                "MPC optimization problem is not initialized. Call init_ocp() first."
            )

        # Build the exogenous input vector
        self.b_curr.value = np.hstack([np.zeros(3), b_curr])  # (6, )
        self.b_wind.value = np.hstack([np.zeros(3), b_wind])  # (6, )
        self.q_ref.value = q_ref
        self.q0.value = q0

        # Solve problem
        self.prob.solve(
            solver=cp.OSQP, verbose=self.verbose, warm_start=True, max_iter=500_000
        )

        # Check termination status
        if self.prob.status != cp.OPTIMAL:
            warnings.warn(
                "Warning: MPC problem not solved to optimality. "
                f"Solve status: {self.prob.status}",
                UserWarning,
            )

        # Return the control sequence and predicted state sequence
        return self.u.value, self.q.value, self.prob.value

    def cost(self) -> tuple[float, float, float, float]:
        """
        Get the cost of the MPC solution along with the separate state, input, and
        terminal costs.

        Raises:
            RuntimeError: If the MPC optimization problem is not initialized.

        Returns:
            cost (float): The total cost of the MPC solution.
            cost_state (float): The state cost.
            cost_terminal (float): The terminal state cost.
            cost_input (float): The input cost.
        """
        if not self.prob_ready:
            raise RuntimeError("MPC optimization problem is not initialized.")

        # Compute the total cost
        cost = self.prob.value

        # Compute the individual costs
        cost_state = sum(
            cp.quad_form(self.q.value[:, k] - self.q_ref, self.Q) for k in range(self.N)
        )
        cost_input = sum(
            cp.quad_form(self.u.value[:, k], self.R) for k in range(self.N)
        )
        cost_terminal = cp.quad_form(self.q.value[:, self.N] - self.q_ref, self.P)

        return cost, cost_state, cost_terminal, cost_input
