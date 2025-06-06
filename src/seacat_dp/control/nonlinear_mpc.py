import warnings

# import casadi as ca  # CasADi used as modeling language along with Gurobi as solver
import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete

from seacat_dp.control.mpc import Mpc
from seacat_dp.utils.transformations import R_b2i

# TODO: unify the MPCs using a single base class which is then inherited by the
# linear_MPC and nonlinear_MPC classes. The base class should contain the common
# methods and attributes (setters, getters, variables...) and the subclasses only
# modifies the required parts, mainly the model matrices and the solve method.
# TODO: inherit from the base Mpc class and only update the required parts


class NonlinearMpc(Mpc):
    """
    Model Predictive Control (MPC) class for controlling a the SeaCat2. The MPC model is
    nonlinear in the A state matrix, which contains a rotation matrix.
    """

    def __init__(self):
        """
        Initialize the MPC controller. All the MPC variables are initialized to zero.
        To set the parameters of the MPC, use the dedicated setter methods.

        The dynamic model is of the form:
            q+ = Aq + Bu + Eb
        where q is the state vector, u is the control input, and b is the known
        exogenous input (measured disturbance). The A, B, and E matrices are discretized
        using either the Forward Euler method or Zero-Order Hold (ZOH) method.
        """
        super().__init__()

        # TODO: modify all to use CasADi for the model matrices instead of cvxpy

        # MPC parameters
        self.N: int  # control horizon
        self.dt: float  # control time step [s]
        self.discretization_method: str = "zoh"  # {"euler", "zoh"}

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

        # Input and state bounds
        self.u_min: np.ndarray
        self.u_max: np.ndarray
        self.q_min: np.ndarray
        self.q_max: np.ndarray

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
