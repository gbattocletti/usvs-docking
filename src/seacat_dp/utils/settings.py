import numpy as np

from seacat_dp.model import seacat_pars


class SimSettings:
    """
    Base simulation settings class for the SeaCat DP project.
    """

    def __init__(self):

        # Simulation settings
        self.seed = 1312
        self.sim_dt = 0.001
        self.sim_t_end = 100.0  # [s] simulation time
        self.verbose = False
        self.show_plots = False
        self.save_plots = True
        self.save_anim = False

        # Controller
        self._controller = "nonlinear_mpc"
        self.controller_options = [
            "linear_mpc",
            "nonlinear_mpc",
            "pid",
        ]
        self._discretization = None
        self.ctrl_dt = 0.25  # [s] control time step
        self.ctrl_N = 20  # Prediction horizon

        # Cost function
        self.Q = np.diag(
            [
                10e3,  # x
                10e3,  # y
                10e3,  # yaw (heading)
                10e1,  # x velocity
                10e1,  # y velocity
                10e0,  # yaw rate
            ]
        )
        self.R = np.diag(
            [
                10e-3,  # stern left
                10e-3,  # stern right
                10e-2,  # bow left
                10e-2,  # bow right
            ]
        )
        self.P = self.Q = np.diag(
            [
                10e3,  # x
                10e3,  # y
                10e2,  # yaw (heading)
                10e1,  # x velocity
                10e1,  # y velocity
                10e-2,  # yaw rate
            ]
        )

        # Initial state and reference
        self.q_0 = np.array(
            [
                0.0,  # x position [m]
                0.0,  # y position [m]
                0 * np.pi,  # yaw angle [rad]
                0.0,  # x velocity [m/s]
                0.0,  # y velocity [m/s]
                0.0,  # yaw rate [rad/s]
            ]
        )

        self.q_ref = np.array(
            [
                0.0,  # x position [m]
                0.0,  # y position [m]
                0 * np.pi,  # yaw angle [rad]
                0.0,  # x velocity [m/s]
                0.0,  # y velocity [m/s]
                0.0,  # yaw rate [rad/s]
            ]
        )

        # Exogenous inputs
        self.v_curr = 0.0  # Current speed [m/s]
        self.h_curr = 0.0  # Current direction [rad]
        self.v_wind = 0.0  # Wind speed [m/s]
        self.h_wind = 0.0

        # Controller bounds
        params = seacat_pars.SeaCatParameters()
        self.u_max = np.array(
            [
                params.max_stern_thrust_forward,
                params.max_stern_thrust_forward,
                params.max_bow_thrust_forward,
                params.max_bow_thrust_forward,
            ]
        )  # [N]
        self.u_min = np.array(
            [
                params.max_stern_thrust_backward,
                params.max_stern_thrust_backward,
                params.max_bow_thrust_backward,
                params.max_bow_thrust_backward,
            ]
        )  # [N]
        # TODO: check how to make this vector more general (SeaDragon case)
        self.delta_u_max = np.array(
            [
                params.max_stern_thrust_forward / params.delay_stern * self.ctrl_dt,
                params.max_stern_thrust_forward / params.delay_stern * self.ctrl_dt,
                params.max_bow_thrust_forward / params.delay_bow * self.ctrl_dt,
                params.max_bow_thrust_forward / params.delay_bow * self.ctrl_dt,
            ]
        )  # [N/s]
        # TODO: check how to make this vector more general (SeaDragon case)
        self.delta_u_min = np.array(
            [
                params.max_stern_thrust_backward / params.delay_stern * self.ctrl_dt,
                params.max_stern_thrust_backward / params.delay_stern * self.ctrl_dt,
                params.max_bow_thrust_backward / params.delay_bow * self.ctrl_dt,
                params.max_bow_thrust_backward / params.delay_bow * self.ctrl_dt,
            ]
        )  # [N/s]

    @property
    def discretization(self):
        if self._discretization is None:
            if self.controller == "linear_mpc":
                self._discretization = "zoh"
            elif self.controller == "nonlinear_mpc":
                self._discretization = "rk4"
            elif self.controller == "pid":
                self._discretization = None
            else:
                raise ValueError(f"Unknown controller type: {self.controller}")
        return self._discretization

    @discretization.setter
    def discretization(self, value):
        if value not in ["euler", "zoh", "rk4"]:
            raise ValueError("Discretization must be 'euler', 'zoh', or 'rk4'.")
        self._discretization = value

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, value):
        if value not in self.controller_options:
            raise ValueError(f"Controller must be one of {self.controller_options}.")
        self._controller = value
        self._discretization = None  # Reset discretization when controller changes
