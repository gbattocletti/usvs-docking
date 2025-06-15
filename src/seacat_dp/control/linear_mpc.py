import warnings

import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete

from seacat_dp.control.mpc import Mpc
from seacat_dp.utils.transformations import R_b2i


class LinearMpc(Mpc):
    """
    Linear Model Predictive Control (MPC) class for controlling the SeaCat2. The class
    assumes a continuous-time LTI system model and implements an MPC control algorithm.
    The continuous-time dynamic model is of the form:
        dot{q} = Aq + Bu + Eb
    where q is the state vector, u is the control input, and b is the lumped
    exogenous input vector (estimated or measured disturbance).
    """

    def __init__(self):
        """
        Initialize the linear MPC controller.
        """
        super().__init__()

        # Initialize the OCP problem
        self.ocp: cp.Problem  # cvxpy optimization problem

        # Initialize the cvxpy solver options
        self.solver_options: dict = {
            "solver": cp.OSQP,
            "verbose": self.verbose,
            "warm_start": True,
            "max_iter": 500_000,
        }

        # Initialize OCP variables
        self.u: cp.Variable
        self.q: cp.Variable

        # Initialize OCP parameters
        self.q_0: cp.Parameter
        self.q_ref: cp.Parameter
        self.b_curr: cp.Parameter
        self.b_wind: cp.Parameter

        # Discretization method
        self.discretization_options = ["euler", "zoh"]  # class-specific options

        # Class-specific model parameters and constants
        # Continuous-time model matrices
        self.Ac: cp.Parameter
        self.Bc: np.ndarray
        self.Ec: np.ndarray

        # Discrete-time model matrices
        self.A: cp.Parameter
        self.B: np.ndarray
        self.E: np.ndarray

    def _set_model(self, M_inv: np.ndarray, D_L: np.ndarray, T: np.ndarray, **kwargs):
        """
        Set the continuous- and discrete-time LTI prediction model.

        Args:
            M_inv: (np.ndarray): Inverse of the inertia matrix (3, 3).
            D_L (np.ndarray): Linear damping matrix (3, 3).
            T (np.ndarray): Thrust matrix (3, 4).
            **kwargs (dict): Additional keyword arguments. Avaiable keys:
                phi (float): The angle around which to linearize the state
                matrix A [rad].
        """
        phi = None  # Default value for the heading angle

        # Parse kwargs
        for key, value in kwargs.items():
            if key == "phi":
                if not isinstance(value, (int, float)):
                    raise ValueError("Heading angle phi must be a float.")
                phi = float(value)
            else:
                raise ValueError(f"Unknown keyword argument: {key}")
        if phi is None:
            raise ValueError(
                "Heading angle phi must be provided as a keyword argument."
            )

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Initialize the model matrices that are OCP parameters
        self.Ac = cp.Parameter(
            (self.n_q, self.n_q), value=np.zeros((self.n_q, self.n_q))
        )
        self.A = cp.Parameter(
            (self.n_q, self.n_q), value=np.zeros((self.n_q, self.n_q))
        )

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

            case "euler":

                # Build the discrete model matrices using Forward Euler
                self.A = I + self.Ac * self.dt  # A_d = I + A_c * dt
                self.B = self.Bc * self.dt  # B_d = B_c * dt
                self.E = self.Ec * self.dt  # E_d = E_c * dt

    def update_model(self, phi: float):
        """
        Update the A state matrix for the MPC by linearizing it around the angle phi.

        NOTE: the update is performed on the continuous-time state matrix A_c. The
        change is then propagated to the discrete-time state matrix A according to the
        discretization method specified in the class attribute `discretization_method`.

        Args:
            phi (float): The angle around which to linearize the state matrix A [rad].
        """
        # Validate the inputs
        if not isinstance(phi, (int, float)):
            raise ValueError(f"phi must be a float or int, got {type(phi)}.")
        phi = float(phi)

        # Update the A matrix
        self.Ac[0:3, 3:6] = R_b2i(phi)

    def _init_ocp(self) -> None:
        """
        Initialize the linear MPC optimization problem with cvxpy.
        """

        # Initialize the OCP variables
        # NOTE: we assume that control horizon and prediction horizon are the same (N)
        self.u = cp.Variable((self.n_u, self.N))
        self.q = cp.Variable((self.n_q, self.N + 1))

        # Initialize the OCP parameters
        self.q_0 = cp.Parameter((self.n_q,), value=np.zeros(self.n_q))
        self.q_ref = cp.Parameter(
            (self.n_q, self.N + 1), value=np.zeros((self.n_q, self.N + 1))
        )
        self.b_curr = cp.Parameter((self.n_q,), value=np.zeros(self.n_q))
        self.b_wind = cp.Parameter((self.n_q,), value=np.zeros(self.n_q))

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.q[:, k] - self.q_ref[:, k], self.Q)  # state cost
            cost += cp.quad_form(self.u[:, k], self.R)  # input cost
        cost += cp.quad_form(
            self.q[:, self.N] - self.q_ref[:, self.N], self.P
        )  # terminal cost

        # Initial condition
        constraints = [self.q[:, 0] == self.q_0]

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            for k in range(self.N):
                constraints += [
                    self.u[:, k] >= self.u_min,
                    self.u[:, k] <= self.u_max,
                ]

        # Input rate constraints
        if self.u_rate_max is not None and self.u_rate_min is not None:
            for k in range(self.N - 1):
                constraints += [
                    self.u[:, k + 1] - self.u[:, k] <= self.u_rate_max,
                    self.u[:, k + 1] - self.u[:, k] >= self.u_rate_min,
                ]

        # State constraints
        if self.q_min is not None and self.q_max is not None:
            raise NotImplementedError("State constraints are not implemented yet. ")

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

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.ocp_ready = True

    def _solve(
        self,
        q_0: np.ndarray,
        q_ref: np.ndarray,
        b_curr: np.ndarray,
        b_wind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the linear MPC OCP and compute the control action based on the current
        state and reference.

        Args:
            q_0 (np.ndarray): The current state of the system (n_q, ).
            q_ref (np.ndarray): The desired reference state (n_q, N + 1).
            b_curr (np.ndarray): The current force (n_q/2, ). The force is a measured or
            estimated exogenous input expressed in the body reference frame, and is
            assumed to be stationary over the prediction horizon.
            b_wind (np.ndarray): The wind force (n_q/2, ). The force is a measured or
            estimated exogenous input expressed in the body reference frame, and is
            assumed to be stationary over the prediction horizon.

        Raises:
            RuntimeError: If the MPC optimization problem is not initialized.

        Returns:
            u (np.ndarray): The computed control action (n_u, N).
            q (np.ndarray): The predicted state sequence (n_q, N + 1).
            cost (float): The cost of the MPC solution.
        """
        # Check if the optimization problem is initialized
        if not self.ocp_ready:
            raise RuntimeError(
                "MPC optimization problem is not initialized. Call init_ocp() first."
            )

        # Build the exogenous input vector
        self.q_0.value = q_0
        self.q_ref.value = q_ref
        self.b_curr.value = np.hstack([np.zeros(self.n_q // 2), b_curr])  # (n_q, )
        self.b_wind.value = np.hstack([np.zeros(self.n_q // 2), b_wind])  # (n_q, )

        # Solve problem
        # self.ocp.solve(
        #     **self.solver_options
        # )  # TODO: check if options are passed correctly

        # Alternative implementation with explicit solver options
        self.ocp.solve(
            solver=cp.OSQP, verbose=self.verbose, warm_start=True, max_iter=500_000
        )

        # Check termination status
        if self.ocp.status != cp.OPTIMAL:
            warnings.warn(
                "Warning: MPC problem not solved to optimality. "
                f"Solve status: {self.ocp.status}",
                UserWarning,
            )

        # Return the control sequence and predicted state sequence
        return self.u.value, self.q.value, self.ocp.value

    def _cost(self) -> tuple[float, float, float, float]:
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
        if not self.ocp_ready:
            raise RuntimeError("MPC optimization problem is not initialized.")

        # Compute the total cost
        cost = self.ocp.value

        # Compute the individual costs
        cost_state = sum(
            cp.quad_form(self.q.value[:, k] - self.q_ref[:, k], self.Q)
            for k in range(self.N)
        )
        cost_input = sum(
            cp.quad_form(self.u.value[:, k], self.R) for k in range(self.N)
        )
        cost_terminal = cp.quad_form(
            self.q.value[:, self.N] - self.q_ref[:, self.N], self.P
        )

        if cost != cost_state + cost_input + cost_terminal:
            warnings.warn(
                "Warning: Total cost does not match the sum of individual costs. "
                f"Total cost: {cost}. State cost: {cost_state}, "
                f"Input cost: {cost_input}, Terminal cost: {cost_terminal}"
                f"(sum: {cost_state + cost_input + cost_terminal}).",
                UserWarning,
            )

        return cost, cost_state, cost_terminal, cost_input
