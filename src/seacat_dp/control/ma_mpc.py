import casadi as ca
import numpy as np
import scipy.linalg

from seacat_dp.control.mpc import Mpc
from seacat_dp.model.model_seacat import SeaCatModel
from seacat_dp.model.model_seadragon import SeaDragonModel


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

        # Solution to the OCP problem. Used to retrieve solution and to provide initial
        # guess for next optimization (warm start).
        self.sol: ca.OptiSol | None = None

        # Solver options
        # Ipopt options: https://coin-or.github.io/Ipopt/OPTIONS.html
        self.solver = "ipopt"
        self.solver_options: dict = {
            "max_iter": 10_000,
            "max_wall_time": 60.0,  # Max solver time [s]
            "max_cpu_time": 60.0,  # Max CPU time [s]
            "print_level": 0,  # 0-5 (0 = silent, 5 = verbose)
            "tol": 1e-6,  # Optimality tolerance
            "acceptable_tol": 1e-4,  # Tolerance for early termination
            "linear_solver": "mumps",  # Recommended for most problems
        }

        # Ocp options
        self.ocp_options: dict = {
            "expand": True,
            "error_on_fail": False,
            "print_time": 0,  # Suppress compiler output
            "verbose": False,
            "record_time": True,  # Record time statistics
        }

        # Initialize OCP variables
        self.n_q: int = 12  # Number of states
        self.n_u: int = 8  # Number of inputs
        self.q: ca.Opti.variable  # (n_q, N+1)
        self.u: ca.Opti.variable  # (n_u, N)
        self.cost_function: ca.Opti.variable  # Cost function

        # Initialize OCP parameters
        self.q_0: ca.Opti.parameter  # Initial state (n_q, 1)
        self.b_curr_sc: ca.Opti.parameter  # SC current input (n_q/2, 1) in body frame
        self.b_wind_sc: ca.Opti.parameter  # SC wind input (n_q/2, 1) in body frame
        self.b_curr_sd: ca.Opti.parameter  # SD current input (n_q/2, 1) in body frame
        self.b_wind_sd: ca.Opti.parameter  # SD wind input (n_q/2, 1) in body frame

        # Check model discretization method
        self.discretization_options = ["euler", "rk4"]  # class-specific options

        # Model parameters (MPC prediction model)
        self.M_inv: np.ndarray | None = None
        self.D_L: np.ndarray | None = None

        # USV-specific model parameters
        self.M_inv_sc: np.ndarray | None = None
        self.D_L_sc: np.ndarray | None = None
        self.D_NL_sc: np.ndarray | None = None
        self.C_sc: np.ndarray | None = None
        self.T_sc: np.ndarray | None = None
        self.M_inv_sd: np.ndarray | None = None
        self.D_L_sd: np.ndarray | None = None
        self.D_NL_sd: np.ndarray | None = None
        self.C_sd: np.ndarray | None = None
        self.b_thrusters_sd: float | None = None
        self.l_thrusters_sd: float | None = None

    def _set_model(  # pylint: disable=arguments-differ
        self,
        plant_seacat: SeaCatModel,
        plant_seadragon: SeaDragonModel,
    ):
        """
        Define the prediction model for the docking MPC controller.

        Args:
            plant_seacat (SeaCatModel): SeaCat model object.
            plant_seadragon (SeaDragonModel): SeaDragon model object.

        Returns:
            None
        """
        # Extract the model parameters from the plant objects
        self.M_inv_sc = plant_seacat.M_inv
        self.D_L_sc = plant_seacat.D_L
        self.D_NL_sc = plant_seacat.D_NL
        self.C_sc = plant_seacat.C
        self.T_sc = plant_seacat.T
        self.M_inv_sd = plant_seadragon.M_inv
        self.D_L_sd = plant_seadragon.D_L
        self.D_NL_sd = plant_seadragon.D_NL
        self.C_sd = plant_seadragon.C
        self.b_thrusters_sd = plant_seadragon.pars.b_thrusters  # 0.85
        self.l_thrusters_sd = plant_seadragon.pars.l_thrusters  # 1.52

        # Validate model parameters
        # TODO: implement nonlinear damping and coriolis in the MPC
        if self.D_NL_sc is not None or self.D_NL_sd is not None:
            raise NotImplementedError(
                "Nonlinear damping matrix (D_NL) is not implemented in the "
                "DockingMPC class. Please remove it from the model parameters."
            )
        if self.C_sc is not None or self.C_sd is not None:
            raise NotImplementedError(
                "Coriolis matrix (C) is not implemented in the DockingMPC class. "
                "Please remove it from the model parameters."
            )

        # Assemble joint matrices for the prediction model
        self.M_inv = scipy.linalg.block_diag(self.M_inv_sc, self.M_inv_sd)
        self.D_L = scipy.linalg.block_diag(self.D_L_sc, self.D_L_sd)

    def _dqdt(
        self,
        q: ca.SX | ca.MX,
        u: ca.SX | ca.MX,
        b_curr_sc: ca.SX | ca.MX,
        b_wind_sc: ca.SX | ca.MX,
        b_curr_sd: ca.SX | ca.MX,
        b_wind_sd: ca.SX | ca.MX,
    ) -> ca.SX | ca.MX:
        """
        Nonlinear continuous-time dynamics function for the MPC controller. To be
        integrated numerically within the MPC prediction model.
            dqdt = f(q, u, b_curr_sc, b_wind_sc, b_curr_sd, b_wind_sd)

        The state is assumed to be structured as:
            q = [x_sc, y_sc, psi_sc, u_sc, v_sc, r_sc,
                 x_sd, y_sd, psi_sd, u_sd, v_sd, r_sd]^T

        The control input is assumed to be structured as:
            u = [T_stern_left_sc, T_stern_right_sc,
                 T_bow_left_sc, T_bow_right_sc,
                 F_left_sd, F_right_sd,
                 alpha_left_sd, alpha_right_sd]^T

        Args:
            q (ca.SX or ca.MX): State vector (n_q, ).
            u (ca.SX or ca.MX): Control input vector (n_u, ).
            b_curr_sc (ca.SX or ca.MX): Current exogenous input (n_q, ) in body frame
                acting on the SeaCat.
            b_wind_sc (ca.SX or ca.MX): Wind exogenous input (n_q, ) in body frame
                acting on the SeaCat.
            b_curr_sd (ca.SX or ca.MX): Current exogenous input (n_q, ) in body frame
                acting on the SeaDragon.
            b_wind_sd (ca.SX or ca.MX): Wind exogenous input (n_q, ) in body frame
                acting on the SeaDragon.

        Returns:
            dq (ca.SX or ca.MX): Time derivative of the state vector (n_q, ).
        """

        # Build joint rotation matrix (block diagonal)
        R = ca.vertcat(
            ca.horzcat(ca.cos(q[2]), -ca.sin(q[2]), 0.0, 0.0, 0.0, 0.0),
            ca.horzcat(ca.sin(q[2]), ca.cos(q[2]), 0.0, 0.0, 0.0, 0.0),
            ca.horzcat(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            ca.horzcat(0.0, 0.0, 0.0, ca.cos(q[8]), -ca.sin(q[8]), 0.0),
            ca.horzcat(0.0, 0.0, 0.0, ca.sin(q[8]), ca.cos(q[8]), 0.0),
            ca.horzcat(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )

        # Compute force generated by the thrusters in their current configuration
        f_sc = self.T_sc @ u[0:4]  # (3, ) force vector for SeaCat
        f_sd = ca.SX.zeros(3)  # Placeholder for SeaDragon force TODO: SeaDragon force
        f = ca.vertcat(f_sc, f_sd)  # (6, ) joint force vector

        # Nonlinear dynamics
        dx = R @ ca.vertcat(q[3:6], q[9:12])  # kinematics
        dv = self.M_inv @ (
            -self.D_L @ ca.vertcat(q[3:6], q[9:12])
            + f
            + ca.vertcat(b_curr_sc, b_curr_sd)
            + ca.vertcat(b_wind_sc, b_wind_sd)
        )

        # Output state derivative
        return ca.vertcat(dx[0:3], dv[0:3], dx[3:6], dv[3:6])

    def _init_ocp(self):
        # TODO
        pass

    def _solve(
        self,
        q_0: np.ndarray,
        b_curr_sc: np.ndarray,
        b_wind_sc: np.ndarray,
        b_curr_sd: np.ndarray,
        b_wind_sd: np.ndarray,
        use_warm_start: bool = True,
    ):
        # TODO implement solve method
        # TODO: consider adding an optional q_ref argument and a corresponding solution
        # strategy (e.g., distributed tracking MPC)
        pass

    def _cost(self):
        # TODO
        pass
