import numpy as np


class Mpc:
    """
    Model Predictive Control (MPC) class for controlling a system. The class assumes a
    LTI system model and implements a basic MPC algorithm.
    """

    def __init__(self):
        """
        Initialize the MPC controller.

        Args:
            model: The system model to be controlled.
            horizon: The prediction horizon for the MPC.
        """
        self.A = np.zeros((6, 6))
        self.B = np.zeros((6, 4))
        self.C = np.zeros((6, 6))
        self.D = np.zeros((6, 4))
        self.Q = np.eye(6)
        self.R = np.eye(4)
        self.Qf = np.eye(6)
        self.N = None

    def __call__(self, q, q_ref, b_curr, b_wind):
        """
        Call the MPC controller to compute the control action. See the `compute_control`
        method for details.

        Args:
            q: The current state of the system (6, ).
            q_ref: The desired reference state (6, ).
            b_curr: The current disturbance (3, ).
            b_wind: The wind disturbance (3, ).

        Returns:
            u: The computed control action.
        """

        u = self.compute_control(q, q_ref, b_curr, b_wind)
        return u

    def set_model(self, A, B, C, D):
        """
        Set the system model matrices.

        Args:
            A: The state matrix (6, 6).
            B: The input matrix (6, 4).
            C: The output matrix (6, 6).
            D: The feedforward matrix (6, 4).
        """
        if not A.shape == (6, 6):
            raise ValueError("State matrix A must be of shape (6, 6).")
        if not B.shape == (6, 4):
            raise ValueError("Input matrix B must be of shape (6, 4).")
        if not C.shape == (6, 6):
            raise ValueError("Output matrix C must be of shape (6, 6).")
        if not D.shape == (6, 4):
            raise ValueError("Feedforward matrix D must be of shape (6, 4).")
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def set_A(self, A):
        """
        Set the state matrix for the MPC.

        Args:
            A: The state matrix (6, 6).
        """
        if not A.shape == (6, 6):
            raise ValueError("State matrix A must be of shape (6, 6).")
        self.A = A

    def set_weights(self, Q, R, Qf):
        """
        Set the cost matrices for the MPC.

        Args:
            Q: The state cost matrix (6, 6).
            R: The input cost matrix (4, 4).
            Qf: The terminal state cost matrix (6, 6).
        """
        # Validate the shapes of the matrices
        # TODO: Consider adding a check for positive definiteness of the matrices
        if not Q.shape == (6, 6):
            raise ValueError("State cost matrix Q must be of shape (6, 6).")
        if not R.shape == (4, 4):
            raise ValueError("Input cost matrix R must be of shape (4, 4).")
        if not Qf.shape == (6, 6):
            raise ValueError("Terminal state cost matrix Qf must be of shape (6, 6).")

        self.Q = Q
        self.R = R
        self.Qf = Qf

    def set_horizon(self, N):
        """
        Set the prediction horizon for the MPC.

        Args:
            N: The prediction horizon (int).
        """
        if not isinstance(N, int) or N <= 0:
            raise ValueError("Prediction horizon N must be a positive integer.")
        self.N = N

    def compute_control(self, q, q_ref, b_curr, b_wind):
        """
        Compute the control action based on the current state and reference.

        Args:
            q: The current state of the system (6, ).
            q_ref: The desired reference state (6, ).
            b_curr: The current force (measured exogenous input).
            b_wind: The wind force (measured exogenous input).

        Returns:
            u: The computed control action (4, ).
        """
        u = np.zeros((4,))
        return u
