import casadi as ca
import numpy as np
import scipy.linalg

from usvs_control.control.mpc import Mpc
from usvs_control.model.model_seacat import SeaCatModel
from usvs_control.model.model_seadragon import SeaDragonModel
from usvs_control.visualization.colors import CmdColors

# TODO: consider adding an optional q_ref argument and a corresponding solution
# strategy (e.g., distributed tracking MPC, where both USVs have their own reference
# point and perform setpoint tracking towards it))


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
            "max_wall_time": 120.0,  # Max solver time [s]
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

        # Class-specific input constraints
        self.max_tot_u_sc: float | None = None  # Max total thrust for SeaCat
        self.max_tot_u_sd: float | None = None  # Max total thrust for seaDragon
        self.target_distance: float = 2.5  # Target distance along x between the USVs
        self.safe_distance: float = 2.0  # Safe distance between the USVs

        # Cost function weights
        self.delta_1 = 1000.0  # heading error weight
        self.delta_2 = 1000.0  # distance error weight
        self.delta_3 = 1.0  # input cost weight

        # Flag to indicate whether q_ref is used in the OCP (different cost function)
        self.mode: str | None = None

    @staticmethod
    def angle_wrap(a: float) -> float:
        """
        Helper function to avoid numerical issues in modulo arithmetic.

        Args:
            a (float): Angle in radians.

        Returns:
            float: Wrapped angle in radians within [-pi, pi].
        """
        return ca.atan2(ca.sin(a), ca.cos(a))

    def set_model(  # pylint: disable=arguments-differ
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
        if self.D_NL_sc is not None or self.D_NL_sd is not None:
            print(
                f"{CmdColors.WARNING}[MA-NMPC]{CmdColors.ENDC} "
                "The nonlinear damping matrix is currently not used in the docking "
                "MPC class."
            )
        if self.C_sc is not None or self.C_sd is not None:
            print(
                f"{CmdColors.WARNING}[MA-NMPC]{CmdColors.ENDC} "
                "The Coriolis matrix is currently not used in the docking MPC class."
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
        f_sd = ca.vertcat(
            ca.horzcat(u[4] * ca.cos(u[6]) + u[5] * ca.cos(u[7])),
            ca.horzcat(u[4] * ca.sin(u[6]) + u[5] * ca.sin(u[7])),
            ca.horzcat(
                u[4]
                * (
                    ca.cos(u[6]) * self.b_thrusters_sd
                    - u[4] * ca.sin(u[6]) * self.l_thrusters_sd
                )
                - u[5]
                * (
                    ca.cos(u[7]) * self.b_thrusters_sd
                    + u[5] * ca.sin(u[7]) * self.l_thrusters_sd
                )
            ),
        )
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

    def init_ocp(self, mode: str = "reference"):
        """
        Initialize centralized MPC optimization problem with the CasADi Opti framework.

        Args:
            mode (str): MPC operation mode. Options are:
                - 'reference': reference tracking mode using q_ref parameter. Used for
                    reference-based docking where the docking location is set
                    externally (by some other layer of the algorithm).
                - 'angle': docking mode with a prespecified docking heading.
                - 'distance': docking mode using distance cost term. No q_ref.
                - 'distance_heading': docking mode using distance and heading cost
                    terms. No q_ref.

        """
        # Check if the OCP has already been initialized
        if self.ocp_ready:
            raise RuntimeError("The OCP has already been initialized.")

        # Initialize the OCP variables
        self.q = self.ocp.variable(self.n_q, self.N + 1)
        self.u = self.ocp.variable(self.n_u, self.N)

        # Initialize the OCP parameters
        self.q_0 = self.ocp.parameter(self.n_q, 1)  # Initial state (n_q, 1)
        self.b_curr_sc = self.ocp.parameter(self.n_q // 4, 1)
        self.b_wind_sc = self.ocp.parameter(self.n_q // 4, 1)
        self.b_curr_sd = self.ocp.parameter(self.n_q // 4, 1)
        self.b_wind_sd = self.ocp.parameter(self.n_q // 4, 1)

        ## Constraints
        # Dynamics
        for k in range(self.N):
            match self.discretization_method:
                case "euler":
                    q_next = self.q[:, k] + self.dt * self._dqdt(
                        self.q[:, k],
                        self.u[:, k],
                        self.b_curr_sc,
                        self.b_wind_sc,
                        self.b_curr_sd,
                        self.b_wind_sd,
                    )

                case "rk4":
                    k1 = self._dqdt(
                        self.q[:, k],
                        self.u[:, k],
                        self.b_curr_sc,
                        self.b_wind_sc,
                        self.b_curr_sd,
                        self.b_wind_sd,
                    )
                    k2 = self._dqdt(
                        self.q[:, k] + 0.5 * self.dt * k1,
                        self.u[:, k],
                        self.b_curr_sc,
                        self.b_wind_sc,
                        self.b_curr_sd,
                        self.b_wind_sd,
                    )
                    k3 = self._dqdt(
                        self.q[:, k] + 0.5 * self.dt * k2,
                        self.u[:, k],
                        self.b_curr_sc,
                        self.b_wind_sc,
                        self.b_curr_sd,
                        self.b_wind_sd,
                    )
                    k4 = self._dqdt(
                        self.q[:, k] + self.dt * k3,
                        self.u[:, k],
                        self.b_curr_sc,
                        self.b_wind_sc,
                        self.b_curr_sd,
                        self.b_wind_sd,
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
                        self.u_min,
                        self.u[:, k],
                        self.u_max,
                    )
                )

        # Input rate constraints
        if self.u_rate_min is not None and self.u_rate_max is not None:
            for k in range(1, self.N):
                self.ocp.subject_to(
                    self.ocp.bounded(
                        self.u_rate_min,
                        self.u[:, k:] - self.u[:, k - 1],
                        self.u_rate_max,
                    )
                )

        # Maximum input constraint (1-norm constraint)
        # NOTE: this constraint is applied separately to the inputs of each USV, as each
        # of them has its own maximum total thrust capability (power budget). For the
        # SeaDragon, the power consumption is computed only for the thrust force and not
        for the azimuth angle.
        if self.max_tot_u is not None:
            for k in range(self.N):
                self.ocp.subject_to(ca.norm_1(self.u[0:4, k]) <= self.max_tot_u_sc)
                self.ocp.subject_to(ca.norm_1(self.u[4:6, k]) <= self.max_tot_u_sd)

        # State constraints
        if self.q_min is not None or self.q_max is not None:
            for k in range(self.N + 1):
                self.ocp.subject_to(
                    self.ocp.bounded(
                        self.q_min,
                        self.q[:, k],
                        self.q_max,
                    )
                )

        # Collision avoidance constraints between the two USVs
        for k in range(self.N):
            self.ocp.subject_to(
                (
                    (self.q[0, k] - self.q[6, k]) ** 2
                    + (self.q[1, k] - self.q[7, k]) ** 2
                )
                >= self.safe_distance**2
            )

        ## Cost function
        self.cost_function = 0

        # State cost
        self.mode = mode  # store mode in memory for solve function
        match mode:
            case "reference":
                # Initialize reference state sequence
                self.q_ref = self.ocp.parameter(self.n_q, self.N + 1)

                # State cost - reference tracking (including terminal cost)
                for k in range(self.N + 1):
                    err = self.q[:, k] - self.q_ref[:, k]
                    err[2] = self.angle_wrap(self.q[2, k] - self.q_ref[2, k])
                    err[8] = self.angle_wrap(self.q[8, k] - self.q_ref[8, k])
                    self.cost_function += err.T @ self.Q @ err

                for k in range(self.N):
                    self.cost_function += 1000 * (
                        1
                        / (
                            (self.q[0, k] - self.q[6, k]) ** 2
                            + (self.q[1, k] - self.q[7, k]) ** 2
                        )
                        - self.target_distance**2
                    )

            case "angle":
                # Initialize reference state sequence
                self.q_ref = self.ocp.parameter(1, self.N + 1)
                for k in range(self.N + 1):
                    self.cost_function += (
                        self.angle_wrap(self.q[2, k] - self.q_ref[2, k]) ** 2
                        + self.angle_wrap(self.q[8, k] - self.q_ref[8, k]) ** 2
                    )

                for k in range(self.N):
                    self.cost_function += self.delta_2 * (
                        (self.q[0, k] - self.q[6, k]) ** 2
                        + (self.q[1, k] - self.q[7, k]) ** 2
                    )

            case "distance":
                # distance cost (alternative)
                for k in range(self.N):
                    self.cost_function += self.delta_2 * (
                        (self.q[0, k] - self.q[6, k]) ** 2
                        + (self.q[1, k] - self.q[7, k]) ** 2
                    )

            case "distance_heading":
                # State cost - heading and distance matching
                for k in range(self.N):
                    self.cost_function += self.delta_1 * (
                        (self.angle_wrap(self.q[2, k] - self.q[8, k]) + np.pi) ** 2
                    )
                # rotation matrix from global to SeaCat body frame -- used also in cost
                R_i2b = ca.vertcat(
                    ca.horzcat(ca.cos(self.q[2]), ca.sin(self.q[2])),
                    ca.horzcat(-ca.sin(self.q[2]), ca.cos(self.q[2])),
                )
                for k in range(self.N):
                    xi = R_i2b @ ca.vertcat(
                        self.q[0, k] - self.q[6, k],
                        self.q[1, k] - self.q[7, k],
                    )  # relative position in SeaCat body frame
                    self.cost_function += self.delta_2 * (
                        (xi[0] - self.target_distance) ** 2 + ca.fabs(xi[1]) ** 2
                    )

            case _:
                raise ValueError(
                    f"Invalid mode '{mode}'. Supported modes are 'reference', "
                    "'free', 'angle', and 'distance'."
                )

        # Input cost
        for k in range(self.N):
            self.cost_function += self.delta_3 * (
                self.u[:, k].T @ self.R @ self.u[:, k]
            )

        # Define the objective and set the solver options
        self.ocp.minimize(self.cost_function)
        self.ocp.solver(self.solver, self.ocp_options, self.solver_options)

        # Flag ocp as initialized
        self.ocp_ready = True  # Set flag to true

    def solve(  # pylint: disable=arguments-renamed, arguments-differ
        self,
        q_0: np.ndarray,
        b_curr_sc: np.ndarray,
        b_wind_sc: np.ndarray,
        b_curr_sd: np.ndarray,
        b_wind_sd: np.ndarray,
        q_ref: np.ndarray | float | None = None,
        use_warm_start: bool = True,
    ):
        """
        Solve the OCP problem for the given initial state and reference state.
        See super().solve() for more details on the arguments and outputs. The input
        arguments of _solve() are assumed to have been already parsed and validated.

        Args:
            q0 (np.ndarray): Initial state (n_q, ).
            b_curr_sc (np.ndarray): SeaCat current disturbance in body frame (3, ).
            b_wind_sc (np.ndarray): SeaCat wind disturbance in body frame (3, ).
            b_curr_sd (np.ndarray): SeaDragon current disturbance in body frame (3, ).
            b_wind_sd (np.ndarray): SeaDragon wind disturbance in body frame (3, ).
            q_ref (np.ndarray): Reference state sequence (n_q, N+1). Default is None.
            use_warm_start (bool): Whether to use warm start from previous solution.
                Default is True.

        Returns:
            u (np.ndarray): The computed control action (n_u, N).
            q (np.ndarray): The predicted state sequence (n_q, N + 1).
            c (float): The cost of the MPC solution.
            t (float): CPU time to solve the MPC problem.

        Raises:
            ValueError: If the reference state trajectory q_ref is provided but the OCP
                is not initialized to use it.
            ValueError: If the OCP is initialized to use a reference state trajectory
                but q_ref is not provided.
            RuntimeError: If the OCP problem is not initialized.
            RuntimeError: If the MPC optimization fails.
        """
        # Parse inputs
        # NOTE: this section is largely copied from the Mpc base class, with minor
        # modifications to account for the additional disturbance inputs and the joint
        # state/input dimensions. This is done to avoid calling the base class solve
        # method, whose signature is incompatible with the multi-agent case.
        # TODO: improve parent class flexibility for multi-agent case.
        if q_0.shape != (self.n_q,):
            raise ValueError(f"q_0 must be of shape (n_q,), got {q_0.shape}")
        if q_ref is not None:
            if self.mode == "angle":
                if isinstance(q_ref, (float, int)):
                    q_ref = float(q_ref)
                    q_ref = np.array([[q_ref] * (self.N + 1)])  # (1, N+1)
            elif self.mode == "reference":
                if q_ref.shape not in [(self.n_q,), (self.n_q, self.N + 1)]:
                    raise ValueError(
                        "q_ref must be of shape (n_q,) or (n_q, N+1), "
                        f"got {q_ref.shape}"
                    )
                if q_ref.shape == (self.n_q,):
                    # cast from (n_q, ) to (n_q, N+1) to match OCP parameter shape
                    q_ref = np.tile(q_ref[:, np.newaxis], (1, self.N + 1))
                if any(q_ref[2, :] <= -np.pi) or any(q_ref[2, :] > np.pi):
                    # NOTE: q_ref has now shape (n_q, N+1) so the whole row [2] is
                    # checked to ensure that all angles are in the range (-pi, pi]
                    print(
                        f"{CmdColors.WARNING}[MA-NMPC]{CmdColors.ENDC}"
                        "q_ref contains angles outside the range (-pi, pi]. "
                        "The angles will be rebounded to this range."
                    )
                    q_ref[2, :] = (q_ref[2, :] + np.pi) % (2 * np.pi) - np.pi
            else:
                # q_ref provided but no use for it
                raise ValueError(
                    "The reference state trajectory q_ref was provided, but the OCP "
                    "was not initialized to use it. Please re-initialize the OCP with "
                    "use_q_ref=True."
                )
        elif q_ref is None and self.mode in ["reference", "angle"]:
            # q_ref required but not provided
            raise ValueError(
                "The OCP was initialized to use a reference state trajectory, but "
                "q_ref was not provided. Please provide q_ref when calling solve()."
            )

        # Set exogenous inputs
        if b_curr_sc is None:
            b_curr_sc = np.zeros(3)
        elif b_curr_sc.shape != (3,):
            raise ValueError(f"b_curr_sc must be of shape (3,), got {b_curr_sc.shape}")
        if b_wind_sc is None:
            b_wind_sc = np.zeros(3)
        elif b_wind_sc.shape != (3,):
            raise ValueError(f"b_wind_sc must be of shape (3,), got {b_wind_sc.shape}")
        if b_curr_sd is None:
            b_curr_sd = np.zeros(3)
        elif b_curr_sd.shape != (3,):
            raise ValueError(f"b_curr_sd must be of shape (3,), got {b_curr_sd.shape}")
        if b_wind_sd is None:
            b_wind_sd = np.zeros(3)
        elif b_wind_sd.shape != (3,):
            raise ValueError(f"b_wind_sd must be of shape (3,), got {b_wind_sd.shape}")

        # Check if the optimization problem is initialized
        if not self.ocp_ready:
            raise RuntimeError(
                "MPC optimization problem is not initialized. Call init_ocp() first."
            )

        # Set the initial state
        self.ocp.set_value(self.q_0, q_0)

        # Set reference
        self.ocp.set_value(self.q_ref, q_ref)

        # Set the disturbances
        self.ocp.set_value(self.b_curr_sc, b_curr_sc)  # (3, )
        self.ocp.set_value(self.b_wind_sc, b_wind_sc)  # (3, )
        self.ocp.set_value(self.b_curr_sd, b_curr_sd)  # (3, )
        self.ocp.set_value(self.b_wind_sd, b_wind_sd)  # (3, )

        # Set the initial guess (warm start) if available
        if use_warm_start is True and self.sol is not None:
            self.ocp.set_initial(self.sol.value_variables())
        elif self.sol is None:
            # warm start on 1st solution to avoid numerical errors
            # See: https://github.com/casadi/casadi/discussions/3539
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-"NaN-detected"in-my-optimization%3F  # pylint: disable=line-too-long
            for k in range(self.N + 1):
                self.ocp.set_initial(self.q[:, k], q_0)

        # Solve the OCP problem
        self.sol = self.ocp.solve()

        if self.sol.stats()["success"] is not True:
            raise RuntimeError("MPC optimization failed: " + self.sol.stats()["status"])

        # Extract the OCP solution
        u = self.sol.value(self.u)
        q = self.sol.value(self.q)
        c = self.sol.value(self.cost_function)
        t = self.sol.stats()["t_proc_total"]  # cpu time. Alternative: t_wall_total

        # Return the solution
        return u, q, c, t

    def cost(
        self, stop_on_sanity_check: bool = False
    ) -> tuple[float, float, float, float]:
        """
        Compute the cost function for the MPC problem. See super().cost() for more
        details on this function and its outputs.

        Args:
            stop_on_sanity_check (bool): Whether to raise an error if the sanity check
                fails. Default is False.

        Returns:
            - c_tot (float): The total cost of the MPC problem.
            - c_state (float): The cost associated with the state trajectory.
            - c_input (float): The cost associated with the control inputs.
            - c_terminal (float): The terminal cost at the end of the horizon.

        Raises:
            RuntimeError: If the OCP is not ready or has not been solved yet.
            ValueError: If the sum of the indivdual component of the cost does not match
                the total cost returned by the MPC.
        """
        # Check if the OCP is ready and has been solved
        if self.ocp_ready is False or self.sol is None:
            raise RuntimeError("OCP is not ready or has not been solved yet.")

        # Get the total cost
        c_tot: float = self.sol.value(self.cost_function)

        # Get the state and input variables
        q = self.sol.value(self.q)
        u = self.sol.value(self.u)

        # Initialize cost terms
        c_state_heading: float = 0.0
        c_state_distance: float = 0.0
        c_input: float = 0.0

        # Compute cost terms (see _init_ocp for reference)
        # TODO: update cost computation to match actual terms of the cost function
        # (needs to depend on MPC modes)
        c_state_heading = self.delta_1 * sum(
            ((np.mod((q[2, k] - q[8, k] + np.pi), 2 * np.pi) - np.pi) + np.pi) ** 2
            for k in range(self.N)
        )
        for k in range(self.N):
            dist_xy = np.linalg.norm(
                np.array(
                    [
                        q[0, k] - q[6, k],
                        q[1, k] - q[7, k],
                    ]
                )
            )
            c_state_distance += self.delta_2 * dist_xy**2
        c_input = self.delta_3 * sum(
            u[:, k].T @ self.R @ u[:, k] for k in range(self.N)
        )

        # Sanity check
        if stop_on_sanity_check is True and not np.isclose(
            c_tot, c_state_heading + c_state_distance + c_input, rtol=1e-4
        ):
            raise ValueError(
                "Warning: Total cost does not match the sum of individual costs. "
                f"\nTotal cost: {c_tot}"
                f"\nState cost (heading): {c_state_heading}"
                f"\nState cost (distance): {c_state_distance}"
                f"\nInput cost: {c_input}"
                f"\nSum: {c_state_heading + c_state_distance + c_input}.",
            )

        return c_tot, c_state_heading, c_state_distance, c_input

    # Dummy methods to avoid errors due to mismatch with abstract methods of parent Mpc
    def _solve(self):  # pylint: disable=arguments-differ
        raise NotImplementedError(
            "Method not implemented as the parent one (solve)"
            "has been replaced with respect to the parent class."
        )

    def _set_model(self):  # pylint: disable=arguments-differ
        raise NotImplementedError(
            "Method not implemented as the parent one (set_model)"
            "has been replaced with respect to the parent class."
        )

    def _cost(self):
        raise NotImplementedError(
            "Method not implemented as the parent one (cost)"
            "has been replaced with respect to the parent class."
        )
