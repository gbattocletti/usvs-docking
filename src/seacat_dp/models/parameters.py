class Parameters:
    """
    A class to hold the parameters for the SeaCat2.

    The parameters are defined as instance attributes in the __init__ method, so that
    multiple instances of the class can be created with different parameters if needed.
    This way, a model with model mismatch can be created by changing the parameters.
    """

    def __init__(self):

        # General parameters
        self.g = 9.81  # gravity [m/s^2]

        # Geometric data
        self.l_tot = 5.7  # total boat length [m]
        self.b_tot = 2  # total boat width [m]
        self.draft = 0.83  # boat draft [m]
        self.xg = 0.0  # location of the center of gravity with respect to the center of
        # the body reference frame along direction x
        self.l_pontoon = self.l_tot  # total length of the two pontoons [m]
        self.b_pontoon = 0.65  # width of the two pontoons [m]
        self.y_pontoon = 0.79  # distance between x axis and pontoon centerline [m]
        self.c_w_pontoon = 0.89  # waterline area coefficient (ratio between waterline
        # area and surrounding pontoon area) [-]
        self.c_b_pontoon = 0.56  # block coefficient (ratio between pontoon volume and
        # volume of box surrounding the pontoon) [-]

        # Inertia properties
        self.m = 1300  # Total vessel mass (localized in the center of gravity) [kg]
        self.I_xx = 868.06  # Inertias (from SST Solidworks data) [kg*m^2]
        self.I_yy = 2710.34
        self.I_zz = 2822.16

        # Time constants
        self.t_sway = 100  # sway time constant [s]
        self.t_yaw = 100  # yaw time constant [s]

        # Thrusters geometric data
        self.alpha = 15  # Bow thrusters angle from y axis [deg]
        self.d_bow = 0.8  # Bow thrusters distance from x axis [m]
        self.d_stern = 0.8  # Rear thrusters distance from x axis [m]
        self.l_bow = 2.1  # Bow thrusters distance from y axis [m]
        self.l_stern = 2.4  # Rear thrusters distance from y axis [m]

        # Thrusters force and maximum speed
        self.u_max = 3.09  # Maximum forward speed corresponding to 6 knots
        self.max_bow_thrust_forward = 170  # bow thrusters maximum forward force [N]
        self.max_bow_thrust_backward = -170  # bow thrusters maximum backward force [N]
        self.max_stern_thrust_forward = 1000  # stern thrusters maximum forward force at
        # 0 knots (0 m/s) [N]
        self.max_stern_thrust_backward = -800  # stern thrusters maximum backward force
        # at 0 knots (0 m/s) [N]
        self.max_stern_thrust_forward_max_u = 500  # stern thrusters maximum forward
        # force at 5.5 knots (2.7m/s) [N]
        self.max_stern_thrust_backward_max_u = -400  # stern thrusters maximum backward
        # force at 5.5 knots (2.7m/s) [N]
