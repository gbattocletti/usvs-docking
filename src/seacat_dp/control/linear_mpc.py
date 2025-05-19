import cvxpy as cp
import numpy as np


class Mpc:
    """
    Model Predictive Control (MPC) class for controlling a system. The class assumes a
    LTI system model and implements a basic MPC algorithm.
    """

    def __init__(self):
        """
        Initialize the MPC controller. All the MPC variables are initialized to zero.
        To set the parameters of the MPC, use the dedicated setter methods.
        """
        self.A: np.ndarray = np.zeros((6, 6))
        self.B: np.ndarray = np.zeros((6, 4))
        self.Q: np.ndarray = np.eye(6)
        self.R: np.ndarray = np.eye(4)
        self.P: np.ndarray = np.eye(6)
        self.N: int = 0
        self.u_min: np.ndarray = np.zeros((4, 1))
        self.u_max: np.ndarray = np.zeros((4, 1))

    def __call__(
        self, q: np.ndarray, q_ref: np.ndarray, b_curr: np.ndarray, b_wind: np.ndarray
    ) -> np.ndarray:
        """
        Call the MPC controller to compute the control action. See the `compute_control`
        method for details.

        Args:
            q (np.ndarray): The current state of the system (6, ).
            q_ref (np.ndarray): The desired reference state (6, ).
            b_curr (np.ndarray): The current disturbance (3, ).
            b_wind (np.ndarray): The wind disturbance (3, ).

        Returns:
            u (np.ndarray): The computed control action.
        """
        # Call the solver function to compute the control action
        u = self.solve(q, q_ref, b_curr, b_wind)
        return u

    def set_A(self, phi: float, H: np.ndarray):
        """
        Set the A state matrix for the MPC.

        Args:
            phi (float): The heading angle of the USV [rad] (-pi <= phi <= pi).
            H (np.ndarray): The velocities dynamic matrix H = -M^-1 * D_L (3, 3).
        """
        # Note: The set_A method is separated from the set_B method to allow for a
        # separate call, since the A matrix is dependent on the heading angle phi and
        # is updated at each MPC solution call.

        # Validate the inputs
        if not isinstance(phi, float):
            raise ValueError("Heading angle phi must be a float.")
        if not -np.pi <= phi < np.pi:
            raise ValueError("Heading angle phi must be in the range [-pi, pi].")
        if not H.shape == (3, 3):
            raise ValueError("Input matrix H must be of shape (3, 3).")

        # Build the A matrix
        A = np.array(
            [
                [0, 0, 0, np.cos(phi), -np.cos(phi), 0],
                [0, 0, 0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        A[3:6, 3:6] = H

        # Set the A matrix
        self.A = A

    def set_B(self, B):
        """
        Set the B state matrix for the MPC.

        Args:
            B (np.ndarray): The input matrix (6, 4).
        """
        # Validate the input
        if not B.shape == (6, 4):
            raise ValueError("Input matrix B must be of shape (6, 4).")

        # Set the B matrix
        self.B = B

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
        if not np.all(np.linalg.eigvals(Q) > 0):
            raise ValueError("State cost matrix Q must be positive definite.")
        if not R.shape == (4, 4):
            raise ValueError("Input cost matrix R must be of shape (4, 4).")
        if not np.all(np.linalg.eigvals(R) > 0):
            raise ValueError("Input cost matrix R must be positive definite.")
        if not P.shape == (6, 6):
            raise ValueError("Terminal state cost matrix P must be of shape (6, 6).")
        if not np.all(np.linalg.eigvals(P) > 0):
            raise ValueError("Terminal state cost matrix P must be positive definite.")

        # Set the cost matrices
        self.Q = Q
        self.R = R
        self.P = P

    def set_horizon(self, N: int):
        """
        Set the prediction horizon for the MPC.

        Args:
            N (int): The prediction horizon (int).
        """
        # Validate the input
        if not isinstance(N, int) or N <= 0:
            raise ValueError("Prediction horizon N must be a positive integer.")

        # Set the prediction horizon
        self.N = N

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

        # Set the input bounds
        self.u_min = u_min
        self.u_max = u_max

    def solve(
        self, q0: np.ndarray, q_ref: np.ndarray, b_curr: np.ndarray, b_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the control action based on the current state and reference.

        Args:
            q0 (np.ndarray): The current state of the system (6, ).
            q_ref (np.ndarray): The desired reference state (6, ).
            b_curr (np.ndarray): The current force (measured exogenous input).
            b_wind (np.ndarray): The wind force (measured exogenous input).

        Returns:
            u (np.ndarray): The computed control action (4, N).
            x (np.ndarray): The predicted state sequence (6, N + 1).
        """
        # Validate the inputs
        if not q0.shape == (6,):
            raise ValueError("Current state q must be of shape (6, ).")
        if not q_ref.shape == (6,):
            raise ValueError("Reference state q_ref must be of shape (6, ).")
        if not b_curr.shape == (3,):
            raise ValueError("Current force b_curr must be of shape (3, ).")
        if not b_wind.shape == (3,):
            raise ValueError("Wind force b_wind must be of shape (3, ).")

        # Modifications:
        # TODO: make q_ref a sequence of length N instead of a single value
        # TODO: add state constraints

        # Build the exogenous input vector
        # CHECKME: check if the wind and current forces are in the same frame or need to
        # be rotated. Here the wind and current forces must be in the body frame
        # TODO: consider moving assembling the b vector to the file calling the MPC
        b = np.zeros(6)
        b[3:6] = b_curr + b_wind

        # Define the optimization variables
        # NOTE: we assume that input horizon and state horizon are the same (N)
        u = cp.Variable((4, self.N))
        x = cp.Variable((6, self.N + 1))

        # Cost function
        cost = 0
        for k in range(self.N):
            q_err = x[:, k] - q_ref  # error state
            cost += cp.quad_form(q_err, self.Q) + cp.quad_form(u[:, k], self.R)

        q_err = x[:, self.N] - q_ref  # error state
        cost += cp.quad_form(q_err, self.P)  # Terminal cost

        # Constraints
        constraints = [x[:, 0] == q0]  # initial condition
        for k in range(self.N):
            constraints += [
                x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k] + b,  # dynamics
                u[:, k] <= self.u_max,  # input upper bound
                u[:, k] >= self.u_min,  # input lower bound
                # state constraints (# TODO)
            ]

        # Solve problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        # Check termination status
        if prob.status != cp.OPTIMAL:
            print("Warning: MPC problem not solved to optimality")

        # Return the control sequence and predicted state sequence
        return u.value, x.value
