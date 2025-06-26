import numpy as np


class PID:
    """
    PID class for implementing a Proportional-Integral-Derivative controller for the
    dynamic positioning of the SeaCat2 USV.
    """

    def __init__(self):

        self.kp: np.ndarray = np.zeros(3)
        self.ki: np.ndarray = np.zeros(3)
        self.kd: np.ndarray = np.zeros(3)
        self.T: np.ndarray = np.zeros((4, 3))  # Thrust allocation matrix

        self.integral: np.ndarray = np.zeros(3)
        self.prev_error: np.ndarray = np.zeros(3)
        self.prev_time: float = 0.0
        self.dt: float = 0.0

    def set_gains(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> None:
        """
        Set the PID gains.

        Args:
            - kp (np.ndarray): Proportional gain vector (3, ).
            - ki (np.ndarray): Integral gain vector (3, ).
            - kd (np.ndarray): Derivative gain vector (3, ).

        Raises:
            - TypeError: If kp, ki, or kd are not numpy arrays.
            - ValueError: If kp, ki, or kd do not have the correct shape.
        """
        # Parse inputs
        if not isinstance(kp, np.ndarray):
            raise TypeError(f"kp, must be numpy arrays, got {type(kp)}.")
        if kp.shape != (3,):
            raise ValueError(f"kp must be a 3-element vector, got {kp.shape}.")
        if not isinstance(ki, np.ndarray):
            raise TypeError(f"ki, must be numpy arrays, got {type(ki)}.")
        if ki.shape != (3,):
            raise ValueError(f"ki must be a 3-element vector, got {ki.shape}.")
        if not isinstance(kd, np.ndarray):
            raise TypeError(f"kd, must be numpy arrays, got {type(kd)}.")
        if kd.shape != (3,):
            raise ValueError(f"kd must be a 3-element vector, got {kd.shape}.")

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
