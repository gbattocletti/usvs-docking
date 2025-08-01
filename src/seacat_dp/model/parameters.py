class Parameters:
    """
    A class to hold the general world parameters for the project.

    The class also provides a description of the USV-specific parameters.
    """

    def __init__(self):

        # General parameters
        self.g: float = 9.81  # gravity [m/s^2]
        self.rho: float = 1026  # density of water [kg/m^3]

        # Geometric data
        self.l_tot: float  # total boat length [m]
        self.b_tot: float  # total boat width [m]
        self.draft: float  # boat draft (height of submerged part) [m]
        self.xg: float  # dist from geometric center to center of gravity along x [m]
        self.l_pontoon: float  # total length of one pontoon [m]
        self.b_pontoon: float  # total width of one pontoon [m]
        self.y_pontoon: float  # distance from x axis to pontoon centerline [m]
        self.c_w_pontoon: float  # waterline area coefficient, i.e., ratio between the
        # 'real' area of the section of the pontoon at the waterline (draft distance
        # from the bottom of the pontoon) and the approximate pontoon area found as
        # l_pontoon*b_pontoon) [-]
        self.c_b_pontoon: float  # block coefficient, i.e., ratio between the real
        # submerged pontoon volume and the approximate submerged pontoon volume found as
        # l_pontoon*b_pontoon*draft) [-]

        # Inertia properties
        self.m: float  # total USV mass (localized in the center of gravity) [kg]
        self.I_xx: float  # inertia around x axis [kg*m^2]
        self.I_yy: float  # inertia around y axis [kg*m^2]
        self.I_zz: float  # inertia around z axis [kg*m^2]

        # Thrusters geometric data
        self.l_thrusters: float  # distance from y axis to thrusters [m]
        self.b_thrusters: float  # distance from x axis to thrusters [m]
        # The thrusters geometric data is USV-dependant and is defined in the specific
        # USV parameters class.

        # Thrusters force and maximum speed
        self.u_max: float  # Maximum forward speed [m/s]
        # The other thrusters force parameters are USV-dependant and are defined in the
        # specific USV parameters class.

        # Time constants
        self.t_sway: float  # sway time constant [s]
        self.t_yaw: float  # yaw time constant [s]
