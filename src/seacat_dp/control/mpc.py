import warnings
from abc import ABC, abstractmethod

import numpy as np

# TODO: check if worth making the class an abstract class. This solution would be more
# elegant but a bit overkill. Would be nice to learn how to make abstract classes tho.


class Mpc:
    """
    Base MPC class for implementing Model Predictive Control (MPC) algorithms for the
    Dynamic Positioning (DP) control of the SeaCat2 USV.
    """

    def __init__(self):
        self.dt = None
        self.N = None
        self.discretization_method = None
        self.model = None  # TODO: needed? I don't think I will have such generic models
        self.A = None
        self.B = None
        self.E = None
        self.Q = None
        self.R = None
        self.P = None
        self.u_min = None
        self.u_max = None
        self.x_min = None
        self.x_max = None

    def set_dt(self, dt):
        pass

    def set_horizon(self, horizon):
        pass

    def set_discretization_method(self, method):
        pass

    def set_model(self, M_inv: np.ndarray, D_L: np.ndarray, T: np.ndarray, phi: float):
        # TODO: check if this can be a unique menthod or if it needs 2 variants (in
        # particular, phi may not be needed for the nonlinear MPC). In the nonliear
        # model additional parameters may be needed if I want to inlcude also the
        # nonlinear damping or the coriolis matrix.
        pass

    def set_weights(self, Q: np.ndarray, R: np.ndarray, P: np.ndarray):
        pass

    def set_input_bounds(self, u_min: np.ndarray, u_max: np.ndarray):
        pass

    def init_ocp(self):
        # TODO: this method needs to be implemented in the subclasses to match the
        # specific modeling language used there
        pass

    def solve(
        self, q0: np.ndarray, q_ref: np.ndarray, b_curr: np.ndarray, b_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the MPC optimization problem.

        Parameters:
        - q_meas: Initial state (6, ).
        - q_ref: State reference vector (6, N).
        - b_curr: Current disturbance vector (3, ).
        - b_wind: Wind disturbance vector (3, ).

        Returns:
        - u: Control input vector (4, N).
        - q_pred: Predicted state trajectory (6, N+1).
        - cost: Computed cost of the MPC solution.
        """

    def cost(self) -> tuple[float, float, float, float]:
        """
        Compute the cost of the MPC solution.

        Returns:
        - J: Total cost.
        - J_state: State cost.
        - J_input: Input cost.
        - J_terminal: Terminal cost.
        """
        warnings.warn(
            "The 'cost' method is not implemented in the base class. "
            "Please implement it in the subclass.",
            UserWarning,
        )
        return 0.0, 0.0, 0.0, 0.0
