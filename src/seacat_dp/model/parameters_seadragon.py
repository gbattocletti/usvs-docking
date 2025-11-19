from seacat_dp.model.parameters import Parameters


class SeaDragonParameters(Parameters):
    """
    Class to hold parameters for the SeaDragon model. See the Parameters class for a
    description of the parameters.

    The parameters are defined as instance attributes in the __init__ method, so that
    multiple instances of the class can be created with different parameters if needed.
    This way, a model with model mismatch can be created by changing the parameters.
    """

    def __init__(self):
        super().__init__()

        # Geometric data
        self.l_tot: float = 5.5
        self.b_tot: float = 2.1
        self.draft: float = 0.6  # placeholder value, TODO
        self.xg: float = 0.0
        self.l_pontoon: float = self.l_tot
        self.b_pontoon: float = 0.4
        self.y_pontoon: float = 0.85
        self.c_w_pontoon: float = 0.8  # placeholder value, TODO
        self.c_b_pontoon: float = 0.5  # placeholder value, TODO

        # Inertia properties
        self.m: float = 330
        self.I_xx: float = 250.0  # placeholder value, TODO
        self.I_yy: float = 650.0  # placeholder value, TODO
        self.I_zz: float = 700.0  # placeholder value, TODO

        # Time constants
        self.t_sway: float = 1.0
        self.t_yaw: float = 1.0

        # Thrusters geometric data
        self.l_thrusters: float = 1.52
        self.b_thrusters: float = self.y_pontoon

        # Thrusters force and maximum speed
        self.u_max: float = 3.4  # Maximum forward speed corresponding to 6.6 knots
        self.max_thrust: float = 100  # Max forward thrust [N]
        self.max_thrust_backward: float = -100  # Max backward thrust [N]
        self.max_thrust_angular_speed: float = 0.2  # Max thrusters rot speed [rad/s]

        # Thrusters time delay
        self.delay_thrusters: float = 1
