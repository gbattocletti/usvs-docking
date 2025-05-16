import numpy as np


class Disturbances:
    """
    A class for the noise generation for the SeaCat2.
    """

    def __init__(self):
        """
        Initializes an instance of the Disturbances class with the default values for
        all the parameters.
        """

        # Measurement noise
        self.sigma_x = 0.1  # std of the position measurement noise [m]
        self.sigma_psi = 0.1  # std of the heading measurement noise [rad]
        self.sigma_v = 0.1  # std of the speed measurement noise [m/s]
        self.sigma_r = 0.1  # std of the rotation speed measurement noise [rad/s]

        # Actuation noise
        self.sigma_tau_stern = 0.1  # std of the stern actuation process noise [N]
        self.sigma_tau_bow = 0.1  # std of the bow actuation process noise [N]

        # Water current
        self.current_angle = 0.0  # angle of the water current with respect to the x
        # axis of the inertial reference frame [rad]
        self.current_speed = 1.0  # speed of the water current measured in the
        # inertial reference frame [m/s]

        # Wind
        self.wind_angle = 0.0  # angle of the wind with respect to the x axis of the
        # inertial reference frame [rad]
        self.wind_speed = 1.0  # speed of the wind measured in the inertial reference
        # frame [m/s]

    def __str__(self):
        """
        Returns a string representation of the Disturbances class.
        """
        return (
            f"Disturbances:\n\t Measurement: sigma_x={self.sigma_x:.2f}, "
            f"sigma_psi={self.sigma_psi:.2f}, sigma_v={self.sigma_v:.2f}, "
            f"sigma_r={self.sigma_r:.2f}\n\t Actuation: "
            f"sigma_tau_stern={self.sigma_tau_stern:.2f}, "
            f"sigma_tau_bow={self.sigma_tau_bow:.2f}\n\t "
            f" Current: current_angle={self.current_angle:.2f} [rad], "
            f"current_speed={self.current_speed:.2f} [m/s]\n"
        )

    def set_current_direction(self, angle: float):
        """
        Sets the angle of the water current.

        Args:
            angle (float): angle of the water current with respect to the x axis of the
                    inertial reference frame [rad]
        """
        self.current_angle = angle

    def set_current_speed(self, speed: float):
        """
        Sets the speed of the water current.

        Args:
            speed (float): speed of the water current measured in the inertial reference
                    frame [m/s]
        """
        self.current_speed = speed

    def set_wind_direction(self, angle: float):
        """
        Sets the angle of the wind.

        Args:
            angle (float): angle of the wind with respect to the x axis of the inertial
            reference frame [rad]
        """
        self.wind_angle = angle

    def set_wind_speed(self, speed: float):
        """
        Sets the speed of the wind.

        Args:
            speed (float): speed of the wind measured in the inertial reference frame
                    [m/s]
        """
        self.wind_speed = speed

    def actuation_noise(self) -> np.ndarray:
        """
        Generates a noise vector with the same shape as the force vector f (4, ).
        """
        # TODO: consider moving to the model class (to be used in between the
        # computation of the forces and the state update)
        # TODO: check if needed (consider removing)
        noise = np.zeros(4)
        noise[0] = np.random.normal(0, self.sigma_tau_stern)
        noise[1] = np.random.normal(0, self.sigma_tau_stern)
        noise[2] = np.random.normal(0, self.sigma_tau_bow)
        noise[3] = np.random.normal(0, self.sigma_tau_bow)
        return noise

    def measurement_noise(self) -> np.ndarray:
        """
        Generates a noise vector with the same shape as the state vector q (6, ).
        """
        noise = np.zeros(6)
        noise[0] = np.random.normal(0, self.sigma_x)
        noise[1] = np.random.normal(0, self.sigma_x)
        noise[2] = np.random.normal(0, self.sigma_psi)
        noise[3] = np.random.normal(0, self.sigma_v)
        noise[4] = np.random.normal(0, self.sigma_v)
        noise[5] = np.random.normal(0, self.sigma_r)
        return noise

    def current(self) -> np.ndarray:
        """
        Generates the water current vector b_water (exogenous input), which has sape
        (3, ). The water current force is applied to the center of mass of the USV.

        In the current implementation, the water current is assumed to be a stationary
        current field with a constant speed and direction (in the inertial reference
        frame).

        Returns:
            b_current (np.ndarray): water current force vector [N]
        """

        f_water = 10 * self.current_speed  # TODO: Look for a better relation
        # between the water speed and the force acting on the USV to make the model more
        # realistic. For the time being, the force of the water will be assumed to be
        # known, directly proportional to the absolute water speed (rather than the
        # speed of the water relative to the boat), and its effect on the USV is
        # assumed to be independent of the heading of the USV.

        b_current = np.zeros(3)
        b_current[0] = f_water * np.cos(self.current_angle)  # x component
        b_current[1] = f_water * np.sin(self.current_angle)  # y component
        b_current[2] = 0.0  # z component (no rotation due to water current)
        return b_current

    def wind(self) -> np.ndarray:
        """
        Generates the wind vector b_wind (exogenous input), which has shape (3, 1).
        The wind force is applied to the center of mass of the USV.

        Returns:
            b_wind (np.ndarray): wind force vector [N]
        """
        b_wind = np.zeros(3)
        return b_wind
