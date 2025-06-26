import casadi as ca
import numpy as np

from seacat_dp.control.mpc import Mpc


class DockingMpc(Mpc):
    """
    Centralized Multi Agent Nonlinear Model Predictive Control (MPC) class for
    controlling the docking of the SeaCat2 and SeaDragon USVs.
    """

    def __init__(self):
        """
        Initialize the MPC controller.
        """
        super().__init__()

        # Initialize the OCP problem
        self.ocp = ca.Opti()
        self.sol = None  # Solution to the OCP problem

    def _set_model(
        self,
        M_inv: np.ndarray,
        D_L: np.ndarray,
        T: np.ndarray,
        **kwargs: dict,
    ):
        pass  # TODO

    def _init_ocp(self):
        pass  # TODO

    def _solve(
        self,
        q_0: np.ndarray,
        q_ref: np.ndarray | None,  # NOTE: added option to set it to None
        b_curr: np.ndarray,
        b_wind: np.ndarray,
    ):
        pass  # TODO

    def _cost(self):
        pass  # TODO
