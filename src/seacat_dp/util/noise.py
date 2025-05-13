import numpy as np


class Noise:
    """
    A class for the noise generation for the SeaCat2.
    """

    def __init__(self):

        # Measurement noise
        self.sigma_x = 0.1  # std of the position measurement noise [m]
        self.sigma_psi = 0.1  # std of the heading measurement noise [rad]
        self.sigma_v = 0.1  # std of the speed measurement noise [m/s]
        self.sigma_r = 0.1  # std of the rotation speed measurement noise [rad/s]

        # Process noise
        self.sigma_tau_stern = 0.1  # std of the stern actuation process noise [N]
        self.sigma_tau_bow = 0.1  # std of the bow actuation process noise [N]

    def measurement_noise(self) -> np.ndarray:
        """
        Generates a noise vector with the same shape as the state vector q (6, 1).
        """
        noise = np.zeros((6, 1))
        noise[0] = np.random.normal(0, self.sigma_x)
        noise[1] = np.random.normal(0, self.sigma_x)
        noise[2] = np.random.normal(0, self.sigma_psi)
        noise[3] = np.random.normal(0, self.sigma_v)
        noise[4] = np.random.normal(0, self.sigma_v)
        noise[5] = np.random.normal(0, self.sigma_r)
        return noise

    def process_noise(self) -> np.ndarray:
        """
        Generates a noise vector with the same shape as the force vector f (4, 1).
        """
        noise = np.zeros((4, 1))
        noise[0] = np.random.normal(0, self.sigma_tau_stern)
        noise[1] = np.random.normal(0, self.sigma_tau_stern)
        noise[2] = np.random.normal(0, self.sigma_tau_bow)
        noise[3] = np.random.normal(0, self.sigma_tau_bow)
        return noise
