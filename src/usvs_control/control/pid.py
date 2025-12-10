import numpy as np


class PID:
    """
    PID class for implementing a Proportional-Integral-Derivative controller for the
    dynamic positioning of the SeaCat2 USV.

    Currently only heading control is implemented.
    """

    def __init__(self):

        # Controller parameters
        self.dt = 0.01
        self.prev_t: float = 0.0

        # State variables
        self.q_des: np.ndarray = np.zeros(6)  # Desired state vector (6, )

        # PID gains and variables
        self.kp: np.ndarray = np.zeros(1)
        self.ki: np.ndarray = np.zeros(1)
        self.kd: np.ndarray = np.zeros(1)
        self.i_sat_in: np.ndarray = np.zeros(1)
        self.i_sat_out: np.ndarray = np.zeros(1)
        self.i_thresh: np.ndarray = np.zeros(1)
        self.q_err_int: np.ndarray = np.zeros(1)  # Integrated error
        self.q_err_prev: np.ndarray = np.zeros(1)

        # Thrust allocation matrix
        self.T: np.ndarray = np.zeros((4, 3))

        # Load default PID parameters
        self.load_sst_params()

        # Print warning message

    def load_sst_params(self) -> None:
        """
        Load the PID parameters for the SeaCat2 USV from the SST configuration.
        """
        self.kp = 0.005
        self.ki = 0.002
        self.kd = 0.0
        self.i_sat_in = 100.0
        self.i_sat_out = 1000.0
        self.i_omega_thresh = 4.0

    def set_gains(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> None:
        """
        Manually set the PID gains.

        Args:
            - kp (np.ndarray): Proportional gain vector
            - ki (np.ndarray): Integral gain vector
            - kd (np.ndarray): Derivative gain vector

        Raises:
            - TypeError: If kp, ki, or kd are not numpy arrays.
            - ValueError: If kp, ki, or kd do not have the correct shape.
        """
        # Parse inputs
        if not isinstance(kp, np.ndarray):
            raise TypeError(f"kp must be a numpy array, got {type(kp)}.")
        if not isinstance(ki, np.ndarray):
            raise TypeError(f"ki must be a numpy array, got {type(ki)}.")
        if not isinstance(kd, np.ndarray):
            raise TypeError(f"kd must be a numpy array, got {type(kd)}.")
        if kp.shape != (1,):
            raise ValueError(f"kp must be a (1, ) vector, got {kp.shape}.")
        if ki.shape != (1,):
            raise ValueError(f"ki must be a (1, ) vector, got {ki.shape}.")
        if kd.shape != (1,):
            raise ValueError(f"kd must be a (1, ) vector, got {kd.shape}.")

        # Set the PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_T(self, T: np.ndarray) -> None:
        """
        Set the thrust allocation matrix. The matrix maps from a (3, ) vector of forces
        on the USV center of mass to a (4, ) vector of thrusts on the four thrusters.

        Args:
            - T (np.ndarray): Thrust allocation matrix (4, 3).

        Raises:
            - TypeError: If T is not a numpy array.
            - ValueError: If T does not have the correct shape.
        """
        # Parse inputs
        if not isinstance(T, np.ndarray):
            raise TypeError(f"T must be a numpy array, got {type(T)}.")
        if T.shape != (4, 3):
            raise ValueError(f"T must be a (4, 3) matrix, got {T.shape}.")

        # Set the thrust allocation matrix
        self.T = T

    def set_desired_state(self, q_des: np.ndarray) -> None:
        """
        Set the desired state for the dynamic positioning controller.

        Args:
            - q_des (np.ndarray): Desired state vector (6, ) containing the USV desired
            position and velocity.

        Raises:
            - TypeError: If q_des is not a numpy array.
            - ValueError: If q_des does not have the correct shape.
        """
        # Parse inputs
        if not isinstance(q_des, np.ndarray):
            raise TypeError(f"q_des must be a numpy array, got {type(q_des)}.")
        if q_des.shape != (6,):
            raise ValueError(f"q_des must be a (6, ) vector, got {q_des.shape}.")

        # Set the desired state
        self.q_des = q_des

    def compute_control(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the control action based on the PID heading controller.

        Args:
            - q (np.ndarray): State vector (6, ) containing the USV measured position
            and velocity.

        Returns:
            - f (np.ndarray): Control action vector (4, ) for the thrusters.

        Raises:
            - ValueError: If error does not have the correct shape.
        """
        if q.shape != (6,):
            raise ValueError(f"State vector must be a (6, ) vector, got {q.shape}.")

        # Compute the error
        q_err = self.q_des - q

        # Compute the control action on the USV center of mass
        tau = np.zeros(3)  # TODO

        # Compute thrusters forces
        f = self.T @ tau
        return f
