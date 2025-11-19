import numpy as np

from seacat_dp.model import parameters_seacat, parameters_seadragon


class SimSettings:
    """
    Settings class for single-agent simulations.
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
        ]
        self._discretization = None
        self.ctrl_dt = 0.25  # [s] control time step
        self.ctrl_N = 20  # Prediction horizon

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
        self.h_wind = 0.0  # Wind direction [rad]

        # USV
        self.usv_type = "seacat"  # USV model type: {"seacat", "seadragon"}

    @property
    def usv_type(self):
        return self._usv_type

    @usv_type.setter
    def usv_type(self, value):

        # Validate USV type
        if value not in ["seacat", "seadragon"]:
            raise ValueError("USV type must be 'seacat' or 'seadragon'.")
        self._usv_type = value

        # Set controller options for SeaCat
        if value == "seacat":
            self.controller_options = ["linear_mpc", "nonlinear_mpc"]
            self.controller = "nonlinear_mpc"
            params = parameters_seacat.SeaCatParameters()  # Load parameters

            # Input constraints
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

            # Input rate constraints
            self.delta_u_max = np.array(
                [
                    params.max_stern_thrust_forward / params.delay_stern * self.ctrl_dt,
                    params.max_stern_thrust_forward / params.delay_stern * self.ctrl_dt,
                    params.max_bow_thrust_forward / params.delay_bow * self.ctrl_dt,
                    params.max_bow_thrust_forward / params.delay_bow * self.ctrl_dt,
                ]
            )  # [N/s]
            self.delta_u_min = np.array(
                [
                    params.max_stern_thrust_backward
                    / params.delay_stern
                    * self.ctrl_dt,
                    params.max_stern_thrust_backward
                    / params.delay_stern
                    * self.ctrl_dt,
                    params.max_bow_thrust_backward / params.delay_bow * self.ctrl_dt,
                    params.max_bow_thrust_backward / params.delay_bow * self.ctrl_dt,
                ]
            )  # [N/s]

            # Cost function
            self.Q = np.diag(
                [
                    10e3,  # x position
                    10e3,  # y position
                    10e2,  # yaw (heading)
                    10e0,  # x velocity
                    10e0,  # y velocity
                    10e0,  # yaw rate
                ]
            )
            self.R = np.diag(
                [
                    10e-2,  # stern left
                    10e-2,  # stern right
                    10e-1,  # bow left
                    10e-1,  # bow right
                ]
            )
            self.P = self.Q  # Terminal cost

        # Set controller options for SeaDragon
        elif value == "seadragon":
            self.controller_options = ["nonlinear_mpc"]
            self.controller = "nonlinear_mpc"
            params = parameters_seadragon.SeaDragonParameters()  # Load parameters

            # Input constraints
            self.u_max = np.array(
                [
                    params.max_thrust,
                    params.max_thrust,
                    2 * np.pi,
                    2 * np.pi,
                ]
            )  # [N] and [rad]
            self.u_min = np.array(
                [
                    params.max_thrust_backward,
                    params.max_thrust_backward,
                    -2 * np.pi,
                    -2 * np.pi,
                ]
            )  # [N] and [rad]

            # Input rate constraints
            self.delta_u_max = np.array(
                [
                    params.max_thrust / params.delay_thrusters * self.ctrl_dt,
                    params.max_thrust / params.delay_thrusters * self.ctrl_dt,
                    params.max_thrust_angular_speed * self.ctrl_dt,
                    params.max_thrust_angular_speed * self.ctrl_dt,
                ]
            )  # [N] and [rad]
            self.delta_u_min = np.array(
                [
                    params.max_thrust_backward / params.delay_thrusters * self.ctrl_dt,
                    params.max_thrust_backward / params.delay_thrusters * self.ctrl_dt,
                    -params.max_thrust_angular_speed * self.ctrl_dt,
                    -params.max_thrust_angular_speed * self.ctrl_dt,
                ]
            )  # [N] and [rad]

            self.Q = np.diag(
                [
                    10e3,  # x position
                    10e3,  # y position
                    10e2,  # yaw (heading)
                    10e0,  # x velocity
                    10e0,  # y velocity
                    10e0,  # yaw rate
                ]
            )
            self.R = np.diag(
                [
                    10e-1,  # stern left
                    10e-1,  # stern right
                    10e-3,  # angle left
                    10e-3,  # angle right
                ]
            )
            self.P = self.Q  # Terminal cost

    @property
    def discretization(self):
        if self._discretization is None:
            if self.controller == "linear_mpc":
                self._discretization = "zoh"
            elif self.controller == "nonlinear_mpc":
                self._discretization = "rk4"
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
