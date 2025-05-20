class Parameters:
    """
    A class to hold the parameters for the SeaCat2.

    The parameters are defined as instance attributes in the __init__ method, so that
    multiple instances of the class can be created with different parameters if needed.
    This way, a model with model mismatch can be created by changing the parameters.
    """

    def __init__(self):

        # General parameters
        self.g: float = 9.81  # gravity [m/s^2]

        # Geometric data
        self.l_tot: float = 5.7  # total boat length [m]
        self.b_tot: float = 2  # total boat width [m]
        self.draft: float = 0.83  # boat draft [m]
        self.xg: float = 0.0  # location of the center of gravity with respect to the
        # center of the body reference frame along direction x
        self.l_pontoon: float = self.l_tot  # total length of the two pontoons [m]
        self.b_pontoon: float = 0.65  # width of the two pontoons [m]
        self.y_pontoon: float = 0.79  # distance from x axis to pontoon centerline [m]
        self.c_w_pontoon: float = 0.89  # waterline area coefficient (ratio between
        # waterline area and surrounding pontoon area) [-]
        self.c_b_pontoon: float = 0.56  # block coefficient (ratio between pontoon
        # volume and volume of box surrounding the pontoon) [-]

        # Inertia properties
        self.m: float = 1300  # total USV mass (localized in the center of gravity) [kg]
        self.I_xx: float = 868.06  # inertias (from SST Solidworks data) [kg*m^2]
        self.I_yy: float = 2710.34
        self.I_zz: float = 2822.16

        # Time constants
        self.t_sway: float = 1  # sway time constant [s]
        self.t_yaw: float = 1  # yaw time constant [s]

        # Thrusters geometric data
        self.alpha: float = 15  # Bow thrusters angle from y axis [deg]
        self.b_bow: float = 0.8  # Bow thrusters distance from x axis [m]
        self.b_stern: float = 0.8  # Rear thrusters distance from x axis [m]
        self.l_bow: float = 2.1  # Bow thrusters distance from y axis [m]
        self.l_stern: float = 2.4  # Rear thrusters distance from y axis [m]

        # Thrusters force and maximum speed
        self.u_max: float = 3.09  # Maximum forward speed corresponding to 6 knots
        self.max_bow_thrust_forward: float = 170  # bow thrust max forward force [N]
        self.max_bow_thrust_backward: float = -170  # bow thrust max backward force [N]
        self.max_stern_thrust_forward: float = 1000  # stern thrusters maximum forward
        # force at 0 knots (0 m/s) [N]
        self.max_stern_thrust_backward: float = -800  # stern thrusters maximum backward
        # force at 0 knots (0 m/s) [N]
        self.max_stern_thrust_forward_max_u: float = 500  # stern thrusters maximum
        # forward force at 5.5 knots (2.7m/s) [N]
        self.max_stern_thrust_backward_max_u: float = -400  # stern thrusters maximum
        # backward force at 5.5 knots (2.7m/s) [N]

        # Thrusters time constants
        self.t_bow: float = 1
        self.t_stern: float = 1
