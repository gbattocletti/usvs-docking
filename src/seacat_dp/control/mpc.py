import warnings
from abc import abstractmethod

import casadi as ca
import cvxpy as cp
import numpy as np


class Mpc:
    """
    Base MPC class for implementing Model Predictive Control (MPC) algorithms for the
    Dynamic Positioning (DP) control of the SeaCat2 USV.
    """

    def __init__(self):

        # MPC parameters
        self.dt: float = None
        self.N: int = None
        self.discretization_method: str = None
        self.discretization_options: list[str] = ["euler", "zoh"]

        # Optimization problem
        self.ocp: cp.Problem | ca.Opti
        self.ocp_ready: bool = False  # Flag to check if the OCP is initialized
        self.update_ocp_automatically: bool = False
        self.solver_options: dict = {}
        self.ocp_options: dict = {}

        # Optimization variables
        self.n_u: int = 4  # Number of control inputs
        self.n_q: int = 6  # Number of states
        self.u: cp.Variable | ca.Opti.variable  # Control input (n_u, N)
        self.q: cp.Variable | ca.Opti.variable  # State trajectory (n_q, N+1)

        # Optimization problem parameters
        self.q_0: cp.Parameter | ca.Opti.parameter  # Initial state (n_q, )
        self.q_ref: cp.Parameter | ca.Opti.parameter  # Reference sequence (n_q, N+1)
        self.b_curr: cp.Parameter | ca.Opti.parameter  # Current disturbance (3, )
        self.b_wind: cp.Parameter | ca.Opti.parameter  # Wind disturbance (3, )

        # Constraints
        self.u_min: np.ndarray = None
        self.u_max: np.ndarray = None
        self.u_rate_min: np.ndarray = None
        self.u_rate_max: np.ndarray = None
        self.max_tot_u: float = None
        self.q_min: np.ndarray = None
        self.q_max: np.ndarray = None
        self.q_rate_min: np.ndarray = None
        self.q_rate_max: np.ndarray = None
        self.q_terminal_min: np.ndarray = None
        self.q_terminal_max: np.ndarray = None

        # Cost function matrices
        self.Q: np.ndarray = None
        self.R: np.ndarray = None
        self.P: np.ndarray = None

        # Misc parameters
        self.verbose: bool = False

    def __call__(
        self,
        q0: np.ndarray,
        q_ref: np.ndarray,
        b_curr: np.ndarray = np.zeros(3),
        b_wind: np.ndarray = np.zeros(3),
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Call the MPC controller to compute the control action.
        See the `solve` method for more details.

        Args:
            q0 (np.ndarray): The current state of the system (n_q, ).
            q_ref (np.ndarray): The desired reference state (n_q, ) or (n_q, N+1).
            b_curr (np.ndarray, optional): The water current disturbance (3, ).
            b_wind (np.ndarray, optional): The wind disturbance (3, ).

        Returns:
            u (np.ndarray): The computed control action.
            x_pred (np.ndarray): The predicted state trajectory (n_q, N+1).
            cost (float): The computed cost of the MPC solution.
        """
        # Call the solver function to compute the control action
        u, x_pred, cost = self.solve(q0, q_ref, b_curr, b_wind)

        return u, x_pred, cost

    def _warn_if_initialized(self):
        if self.ocp_ready:
            if self.update_ocp_automatically:
                warnings.warn(
                    "Warning: MPC problem is already initialized. "
                    "Updating it to apply the changes.",
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

    def set_n_u(self, n_u: int):
        """
        Set the number of control inputs for the MPC controller.

        Parameters:
        - n_u (int): Number of control inputs.

        Raises:
        - ValueError: If n_u is not a positive integer.
        - TypeError: If n_u is not an int.
        """
        # Parse input
        if not isinstance(n_u, int):
            raise TypeError(f"n_u must be an int, got {type(n_u)}")
        if n_u <= 0:
            raise ValueError(f"n_u must be a positive integer, got {n_u}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the number of control inputs
        warnings.warn(
            "The value of n_u is being changed. This may not be supported by"
            "the MPC implementation.",
            UserWarning,
        )
        self.n_u = n_u

    def set_n_q(self, n_q: int):
        """
        Set the number of control inputs for the MPC controller.

        Parameters:
        - n_q (int): Number of control inputs.

        Raises:
        - ValueError: If n_u is not a positive integer.
        - TypeError: If n_u is not an int.
        """
        # Parse input
        if not isinstance(n_q, int):
            raise TypeError(f"n_q must be an int, got {type(n_q)}")
        if n_q <= 0:
            raise ValueError(f"n_u must be a positive integer, got {n_q}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the number of control inputs
        warnings.warn(
            "The value of n_q is being changed. This may not be supported by"
            "the MPC implementation.",
            UserWarning,
        )
        self.n_q = n_q

    def set_dt(self, dt: float | int):
        """
        Set the time step for the MPC controller.

        Parameters:
        - dt (float | int): Time step for the MPC controller in seconds.

        Raises:
        - ValueError: If dt is not a positive number.
        - TypeError: If dt is not a float or int.
        """
        # Parse input
        if not isinstance(dt, (float, int)):
            raise TypeError(f"dt must be a float or int, got {type(dt)}")
        if isinstance(dt, int):
            dt = float(dt)
        if dt <= 0:
            raise ValueError(f"dt must be a positive number, got {dt}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the time step
        self.dt = dt

    def set_horizon(self, horizon: int):
        """
        Set the prediction horizon for the MPC controller.

        Parameters:
        - horizon (int): Prediction horizon for the MPC controller.

        Raises:
        - ValueError: If horizon is not a positive integer.
        - TypeError: If horizon is not an int.
        """
        # Parse input
        if not isinstance(horizon, int):
            raise TypeError(f"horizon must be an int, got {type(horizon)}")
        if horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the prediction horizon
        self.N = horizon

    def set_discretization_method(self, method: str):
        """
        Set the discretization method for the MPC controller.

        Parameters:
        - method (str): Discretization method for the MPC controller. Must be part of
        the `self.discretization_options` list. The list can be customized in the
        subclasses to allow for specific discretization methods.

        Raises:
        - ValueError: If method is not in the `self.discretization_options` list.
        - TypeError: If method is not a string.
        """
        # Parse input
        if not isinstance(method, str):
            raise TypeError(f"method must be a string, got {type(method)}")
        if method not in self.discretization_options:
            raise ValueError(
                f"method must be one of {self.discretization_options}, got {method}"
            )

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the discretization method
        self.discretization_method = method

    def set_weights(self, Q: np.ndarray, R: np.ndarray, P: np.ndarray):
        """
        Set the cost function weights for the MPC controller.

        Parameters:
        - Q (np.ndarray): State cost matrix (n_q, n_q).
        - R (np.ndarray): Input cost matrix (n_u, n_u).
        - P (np.ndarray): Terminal cost matrix (n_q, n_q).

        Raises:
        - ValueError: If Q, R, or P are not of the expected shapes.
        - TypeError: If Q, R, or P are not numpy arrays.
        """
        # Parse inputs
        if not isinstance(Q, np.ndarray):
            raise TypeError(f"Q must be a numpy array, got {type(Q)}")
        if Q.shape != (self.n_q, self.n_q):
            raise ValueError(
                f"Q must be of shape ({self.n_q}, {self.n_q}), got {Q.shape}"
            )
        if not isinstance(R, np.ndarray):
            raise TypeError(f"R must be a numpy array, got {type(R)}")
        if R.shape != (self.n_u, self.n_u):
            raise ValueError(
                f"R must be of shape ({self.n_u}, {self.n_u}), got {R.shape}"
            )
        if not isinstance(P, np.ndarray):
            raise TypeError(f"P must be a numpy array, got {type(P)}")
        if P.shape != (self.n_q, self.n_q):
            raise ValueError(
                f"P must be of shape ({self.n_q}, {self.n_q}), got {P.shape}"
            )

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the cost function weights
        self.Q = Q
        self.R = R
        self.P = P

    def set_input_bounds(self, u_min: np.ndarray, u_max: np.ndarray):
        """
        Set the input bounds for the MPC controller.

        Note that this method only allows to set _bounds_ for the inputs. In input
        constraint is needed, it should be implemented in the subclass and added to
        the OCP constraints in the `_init_ocp` method.

        Parameters:
        - u_min (np.ndarray): Minimum input bounds (n_u, ).
        - u_max (np.ndarray): Maximum input bounds (n_u, ).

        Raises:
        - ValueError: If u_min or u_max are not of shape (n_u,).
        - TypeError: If u_min or u_max are not numpy arrays.
        """
        # Parse inputs
        if u_min.shape != (self.n_u,):
            raise ValueError(f"u_min must be of shape ({self.n_u},), got {u_min.shape}")
        if not isinstance(u_min, np.ndarray):
            raise TypeError(f"u_min must be a numpy array, got {type(u_min)}")
        if u_max.shape != (self.n_u,):
            raise ValueError(f"u_max must be of shape ({self.n_u},), got {u_max.shape}")
        if not isinstance(u_max, np.ndarray):
            raise TypeError(f"u_max must be a numpy array, got {type(u_max)}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the input bounds
        self.u_min = u_min
        self.u_max = u_max

    def set_input_rate_bounds(self, u_rate_min: np.ndarray, u_rate_max: np.ndarray):
        """
        Set the input rate bounds for the MPC controller.

        Parameters:
        - u_rate_min (np.ndarray): Minimum input rate bounds (n_u, ).
        - u_rate_max (np.ndarray): Maximum input rate bounds (n_u, ).

        Raises:
        - ValueError: If u_rate_min or u_rate_max are not of shape (n_u,).
        - TypeError: If u_rate_min or u_rate_max are not numpy arrays.
        """
        # Parse inputs
        if u_rate_min.shape != (self.n_u,):
            raise ValueError(
                f"u_rate_min must be of shape ({self.n_u},), got {u_rate_min.shape}"
            )
        if not isinstance(u_rate_min, np.ndarray):
            raise TypeError(f"u_rate_min must be a numpy array, got {type(u_rate_min)}")
        if u_rate_max.shape != (self.n_u,):
            raise ValueError(
                f"u_rate_max must be of shape ({self.n_u},), got {u_rate_max.shape}"
            )
        if not isinstance(u_rate_max, np.ndarray):
            raise TypeError(f"u_rate_max must be a numpy array, got {type(u_rate_max)}")

        # Set the input rate bounds
        self.u_rate_min = u_rate_min
        self.u_rate_max = u_rate_max

    def set_max_tot_u(self, max_tot_u: float):
        """
        Set the maximum total input force that can be applied by the MPC controller.

        Params:
        - max_tot_u (float): Maximum total input force that can be applied by the MPC
            controller. This is the sum of the absolute values of the control inputs.

        Raises:
        - ValueError: If max_tot_u is not a positive number.
        - TypeError: If max_tot_u is not a float or int.
        """
        # Parse input
        if not isinstance(max_tot_u, (float, int)):
            raise TypeError(f"max_tot_u must be a float or int, got {type(max_tot_u)}")
        if isinstance(max_tot_u, int):
            max_tot_u = float(max_tot_u)
        if max_tot_u <= 0:
            raise ValueError(f"max_tot_u must be a positive number, got {max_tot_u}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the maximum total input force
        self.max_tot_u = max_tot_u

    def set_state_bounds(self, q_min: np.ndarray, q_max: np.ndarray):
        """
        Set the state bounds for the MPC controller.

        Note that this method only allows to set _bounds_ for the state. In case a more
        complex state constraint set is needed, it should be implemented in the
        subclass and added to the OCP constraints in the `_init_ocp` method.

        Parameters:
        - q_min (np.ndarray): Minimum state bounds (n_q, ).
        - q_max (np.ndarray): Maximum state bounds (n_q, ).

        Raises:
        - ValueError: If q_min or q_max are not of shape (n_q,).
        - TypeError: If q_min or q_max are not numpy arrays.
        """
        # Parse inputs
        if q_min.shape != (self.n_q,):
            raise ValueError(f"q_min must be of shape ({self.n_q},), got {q_min.shape}")
        if not isinstance(q_min, np.ndarray):
            raise TypeError(f"q_min must be a numpy array, got {type(q_min)}")
        if q_max.shape != (self.n_q,):
            raise ValueError(f"q_max must be of shape ({self.n_q},), got {q_max.shape}")
        if not isinstance(q_max, np.ndarray):
            raise TypeError(f"q_max must be a numpy array, got {type(q_max)}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the state bounds
        self.q_min = q_min
        self.q_max = q_max

    def set_state_rate_bounds(self, q_rate_min: np.ndarray, q_rate_max: np.ndarray):
        """
        Set the state rate bounds for the MPC controller.

        Parameters:
        - q_rate_min (np.ndarray): Minimum state rate bounds (n_q, ).
        - q_rate_max (np.ndarray): Maximum state rate bounds (n_q, ).

        Raises:
        - ValueError: If q_rate_min or q_rate_max are not of shape (n_q,).
        - TypeError: If q_rate_min or q_rate_max are not numpy arrays.
        """
        # Parse inputs
        if q_rate_min.shape != (self.n_q,):
            raise ValueError(
                f"q_rate_min must be of shape ({self.n_q},), got {q_rate_min.shape}"
            )
        if not isinstance(q_rate_min, np.ndarray):
            raise TypeError(f"q_rate_min must be a numpy array, got {type(q_rate_min)}")
        if q_rate_max.shape != (self.n_q,):
            raise ValueError(
                f"q_rate_max must be of shape ({self.n_q},), got {q_rate_max.shape}"
            )
        if not isinstance(q_rate_max, np.ndarray):
            raise TypeError(f"q_rate_max must be a numpy array, got {type(q_rate_max)}")

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the state rate bounds
        self.q_rate_min = q_rate_min
        self.q_rate_max = q_rate_max

    def set_terminal_state_bounds(
        self, q_terminal_min: np.ndarray, q_terminal_max: np.ndarray
    ):
        """
        Set the terminal state bounds for the MPC controller.

        Note that this method only allows to set _bounds_ for the terminal state. In
        case a more complex terminal constraint is needed, it should be implemented in
        the subclass and added to the OCP constraints in the `_init_ocp` method.

        Parameters:
        - q_terminal_min (np.ndarray): Minimum terminal state bounds (n_q, ).
        - q_terminal_max (np.ndarray): Maximum terminal state bounds (n_q, ).

        Raises:
        - ValueError: If q_terminal_min or q_terminal_max are not of shape (n_q,).
        - TypeError: If q_terminal_min or q_terminal_max are not numpy arrays.
        """
        # Parse inputs
        if q_terminal_min.shape != (self.n_q,):
            raise ValueError(
                f"q_terminal_min must be of shape ({self.n_q},), got "
                f"{q_terminal_min.shape}"
            )
        if not isinstance(q_terminal_min, np.ndarray):
            raise TypeError(
                f"q_terminal_min must be a numpy array, got {type(q_terminal_min)}"
            )
        if q_terminal_max.shape != (self.n_q,):
            raise ValueError(
                f"q_terminal_max must be of shape ({self.n_q},), got "
                f"{q_terminal_max.shape}"
            )
        if not isinstance(q_terminal_max, np.ndarray):
            raise TypeError(
                f"q_terminal_max must be a numpy array, got {type(q_terminal_max)}"
            )

        # Check if the problem is already initialized
        self._warn_if_initialized()

        # Set the terminal state bounds
        self.q_terminal_min = q_terminal_min
        self.q_terminal_max = q_terminal_max

    def set_model(
        self,
        M_inv: np.ndarray,
        D_L: np.ndarray,
        T: np.ndarray,
        **kwargs: dict,
    ):
        """
        Set the model matrices for the MPC controller.

        Parameters:
        - M_inv (np.ndarray): Inverse of the mass matrix (n_q, n_q).
        - D_L (np.ndarray): Linear damping matrix (n_q, n_q).
        - T (np.ndarray): Transformation matrix (n_q, n_q).
        - **kwargs (dict): Additional keyword arguments for the specific MPC subclasses.
            The kwargs are passed directly to the `_set_model` method of the subclass.
        """
        # Parse inputs
        if not isinstance(M_inv, np.ndarray):
            raise TypeError(f"M_inv must be a numpy array, got {type(M_inv)}")
        if M_inv.shape != (self.n_q, self.n_q):
            raise ValueError(
                f"M_inv must be of shape ({self.n_q}, {self.n_q}), got {M_inv.shape}"
            )
        if not isinstance(D_L, np.ndarray):
            raise TypeError(f"D_L must be a numpy array, got {type(D_L)}")
        if D_L.shape != (self.n_q, self.n_q):
            raise ValueError(
                f"D_L must be of shape ({self.n_q}, {self.n_q}), got {D_L.shape}"
            )
        if not isinstance(T, np.ndarray):
            raise TypeError(f"T must be a numpy array, got {type(T)}")
        if T.shape != (self.n_q, self.n_q):
            raise ValueError(
                f"T must be of shape ({self.n_q}, {self.n_q}), got {T.shape}"
            )

        # Call the class-specific method to set the model
        self._set_model(M_inv, D_L, T, **kwargs)

    @abstractmethod
    def _set_model(
        self,
        M_inv: np.ndarray,
        D_L: np.ndarray,
        T: np.ndarray,
        **kwargs: dict,
    ):
        """
        Internal method to set the model matrices for the MPC controller. This method
        must be implemented in the subclasses. See the `set_model` method for more
        details on the inputs and outputs.

        Raises:
        - NotImplementedError: If the method is not implemented in the subclass.
        """

        raise NotImplementedError(
            "The _set_model method must be implemented in subclasses."
        )

    def init_ocp(self):
        """
        Initialize the OCP (Optimal Control Problem) for the MPC controller. This method
        must be called before solving the MPC problem. It sets up the optimization
        problem, including the cost function, constraints, and variables.

        Raises:
        - RuntimeError: If the OCP has already been initialized.
        - NotImplementedError: If the _init_ocp method is not implemented in the
        subclass.
        """
        # Check if the OCP has already been initialized
        if self.ocp_ready:
            raise RuntimeError("The OCP has already been initialized.")

        # Call the class-specific method to initialize the OCP
        self._init_ocp()
        self.ocp_ready = True  # Set flag to true

    @abstractmethod
    def _init_ocp(self):
        """
        Internal method to initialize the OCP. This method must be implemented in the
        subclasses. See the `init_ocp` method for more details.

        Raises:
        - NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "The _init_ocp method must be implemented in subclasses."
        )

    def solve(
        self,
        q_0: np.ndarray,
        q_ref: np.ndarray,
        b_curr: np.ndarray = np.zeros(3),
        b_wind: np.ndarray = np.zeros(3),
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the MPC optimization problem.

        Parameters:
        - q_0 (np.ndarray): Initial state (n_q, ).
        - q_ref: State reference vector (n_q, ) or (n_q, N+1). If (n_q, ) it is
            broadcasted to (n_q, N+1). The 1st element has no effect on the OCP solution
            as it is the reference for the initial state, on which the MPC has no
            agency. However, it influences the value of the cost by offsetting it by
            the initial state cost.
        - b_curr (np.ndarray, optional): Current force expressed in the body reference
            frame (3, ). The force is assumed to be stationary over the horizon.
            Deault is a zero vector.
        - b_wind (np.ndarray, optional): Wind force expressed in the body reference
            frame (3, ). The force is assumed to be stationary over the horizon.
            Default is a zero vector.

        Raises:
        - ValueError: If q_0, q_ref, b_curr, or b_wind are not of the expected shapes.
        - TypeError: If q_0, q_ref, b_curr, or b_wind are not numpy arrays.
        - RuntimeError: If the OCP has not been initialized.
        - NotImplementedError: If the _solve method is not implemented in the subclass.

        Returns:
        - u: Optimal control input sequence (n_u, N).
        - q_pred: Predicted state trajectory (n_q, N+1).
        - c: Computed cost of the MPC solution.
        """
        # Parse inputs
        if q_0.shape != (self.n_q,):
            raise ValueError(f"q_0 must be of shape (n_q,), got {q_0.shape}")
        if q_ref.shape not in [(self.n_q,), (self.n_q, self.N + 1)]:
            raise ValueError(
                f"q_ref must be of shape (n_q,) or (n_q, N+1), got {q_ref.shape}"
            )
        if q_ref.shape == (self.n_q,):
            q_ref = np.tile(q_ref[:, np.newaxis], (1, self.N + 1))  # cast to (n_q, N+1)
        if b_curr is None:
            b_curr = np.zeros(3)
        elif b_curr.shape != (3,):
            raise ValueError(f"b_curr must be of shape (3,), got {b_curr.shape}")
        if b_wind is None:
            b_wind = np.zeros(3)
        elif b_wind.shape != (3,):
            raise ValueError(f"b_wind must be of shape (3,), got {b_wind.shape}")

        # Check if the optimization problem is initialized
        if not self.ocp_ready:
            raise RuntimeError(
                "MPC optimization problem is not initialized. Call init_ocp() first."
            )

        # Solve the MPC optimization problem
        u, x_pred, cost = self._solve(q_0, q_ref, b_curr, b_wind)
        return u, x_pred, cost

    @abstractmethod
    def _solve(
        self,
        q_0: np.ndarray,
        q_ref: np.ndarray,
        b_curr: np.ndarray,
        b_wind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Internal method to solve the MPC optimization problem. This method must be
        implemented in the subclasses. See the `solve` method for more details on the
        inputs and outputs.

        Raises:
        - NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "The _solve method must be implemented in subclasses."
        )

    def cost(self) -> tuple[float, float, float, float]:
        """
        Compute the cost of the MPC solution.

        Returns:
            tuple: A tuple containing the total cost, state cost, input cost, and
            terminal cost.
            - c_tot (float): The total cost of the MPC problem.
            - c_state (float): The cost associated with the state trajectory.
            - c_input (float): The cost associated with the control inputs.
            - c_terminal (float): The terminal cost at the end of the horizon.

        Raises:
        - RuntimeError: If the OCP has not been initialized.
        - NotImplementedError: If the _cost method is not implemented in the subclass.
        """
        # Check if the OCP has been initialized
        if not self.ocp_ready:
            raise RuntimeError(
                "The OCP has not been initialized. Call init_ocp() first."
            )

        c_tot, c_state, c_terminal, c_input = self._cost()
        return c_tot, c_state, c_terminal, c_input

    @abstractmethod
    def _cost(self):
        """
        Internal method to compute the cost of the MPC solution. This method must be
        implemented in the subclasses. See the `cost` method for more details on the
        inputs and outputs.

        Raises:
        - NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("The _cost method must be implemented in subclasses.")
