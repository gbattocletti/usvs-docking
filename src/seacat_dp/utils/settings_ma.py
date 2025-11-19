import numpy as np

from seacat_dp.model import parameters_seacat, parameters_seadragon


class SimSettings:
    """
    Settings class for multi-agent simulations.
    """

    def __init__(self):

        # Simulation settings
        self.seed: int = 1312
        self.sim_dt: float = 0.001
        self.sim_t_end: float = 100.0  # [s] simulation time
        self.show_plots: bool = False
        self.save_plots: bool = True
        self.save_anim: bool = True

        # Controller
        self._controller: str = "nonlinear_mpc"
        self.discretization: str = (
            "rk4"  # Discretization method {'euler', 'rk4', 'zoh'}
        )
        self.ctrl_dt: float = 0.5  # [s] control time step
        self.ctrl_N: int = 20  # Prediction horizon

        # Initial state and reference
        self.q_0: np.ndarray = np.array(
            [
                0.0,  # x position SeaCat [m]
                0.0,  # y position SeaCat [m]
                0.0 * np.pi,  # yaw angle SeaCat [rad]
                0.0,  # x velocity SeaCat [m/s]
                0.0,  # y velocity SeaCat [m/s]
                0.0 * np.pi,  # yaw rate SeaCat [rad/s]
                0.0,  # x position SeaDragon [m]
                0.0,  # y position SeaDragon [m]
                0.0 * np.pi,  # yaw angle SeaDragon [rad]
                0.0,  # x velocity SeaDragon [m/s]
                0.0,  # y velocity SeaDragon [m/s]
                0.0 * np.pi,  # yaw rate SeaDragon [rad/s]
            ]
        )

        # Exogenous inputs
        self.v_curr: float = 0.0  # Current speed [m/s]
        self.h_curr: float = 0.0  # Current direction [rad]
        self.v_wind: float = 0.0  # Wind speed [m/s]
        self.h_wind: float = 0.0  # Wind direction [rad]

        # Disturbances
        self.enable_measurement_noise: bool = False  # enable/disable measurement noise
        self.enable_actuation_noise: bool = False  # enable/disable actuation noise

        # Load parameters
        params_sc = parameters_seacat.SeaCatParameters()
        params_sd = parameters_seadragon.SeaDragonParameters()

        # Input constraints
        self.u_max: np.ndarray = np.array(
            [
                params_sc.max_stern_thrust_forward,
                params_sc.max_stern_thrust_forward,
                params_sc.max_bow_thrust_forward,
                params_sc.max_bow_thrust_forward,
                params_sd.max_thrust,
                params_sd.max_thrust,
                2 * np.pi,
                2 * np.pi,
            ]
        )  # [N] and [rad]
        self.u_min: np.ndarray = np.array(
            [
                params_sc.max_stern_thrust_backward,
                params_sc.max_stern_thrust_backward,
                params_sc.max_bow_thrust_backward,
                params_sc.max_bow_thrust_backward,
                params_sd.max_thrust_backward,
                params_sd.max_thrust_backward,
                -2 * np.pi,
                -2 * np.pi,
            ]
        )  # [N] and [rad]

        # Input rate constraints
        self.delta_u_max: np.ndarray = np.array(
            [
                params_sc.max_stern_thrust_forward
                / params_sc.delay_stern
                * self.ctrl_dt,
                params_sc.max_stern_thrust_forward
                / params_sc.delay_stern
                * self.ctrl_dt,
                params_sc.max_bow_thrust_forward / params_sc.delay_bow * self.ctrl_dt,
                params_sc.max_bow_thrust_forward / params_sc.delay_bow * self.ctrl_dt,
                params_sd.max_thrust / params_sd.delay_thrusters * self.ctrl_dt,
                params_sd.max_thrust / params_sd.delay_thrusters * self.ctrl_dt,
                params_sd.max_thrust_angular_speed * self.ctrl_dt,
                params_sd.max_thrust_angular_speed * self.ctrl_dt,
            ]
        )  # [N] and [rad]
        self.delta_u_min: np.ndarray = np.array(
            [
                params_sc.max_stern_thrust_backward
                / params_sc.delay_stern
                * self.ctrl_dt,
                params_sc.max_stern_thrust_backward
                / params_sc.delay_stern
                * self.ctrl_dt,
                params_sc.max_bow_thrust_backward / params_sc.delay_bow * self.ctrl_dt,
                params_sc.max_bow_thrust_backward / params_sc.delay_bow * self.ctrl_dt,
                params_sd.max_thrust_backward
                / params_sd.delay_thrusters
                * self.ctrl_dt,
                params_sd.max_thrust_backward
                / params_sd.delay_thrusters
                * self.ctrl_dt,
                -params_sd.max_thrust_angular_speed * self.ctrl_dt,
                -params_sd.max_thrust_angular_speed * self.ctrl_dt,
            ]
        )  # [N] and [rad]

        # Cost function
        self.Q: np.ndarray = np.diag(
            [
                10e3,  # x position SeaCat
                10e3,  # y position SeaCat
                10e2,  # yaw (heading) SeaCat
                10e0,  # x velocity SeaCat
                10e0,  # y velocity SeaCat
                10e0,  # yaw rate SeaCat
                10e3,  # x position SeaDragon
                10e3,  # y position SeaDragon
                10e2,  # yaw (heading) SeaDragon
                10e0,  # x velocity SeaDragon
                10e0,  # y velocity SeaDragon
                10e0,  # yaw rate SeaDragon
            ]
        )
        self.R: np.ndarray = np.diag(
            [
                10e-2,  # stern left SeaCat
                10e-2,  # stern right SeaCat
                10e-1,  # bow left SeaCat
                10e-1,  # bow right SeaCat
                10e-1,  # stern left SeaDragon
                10e-1,  # stern right SeaDragon
                10e-3,  # angle left SeaDragon
                10e-3,  # angle right SeaDragon
            ]
        )
        self.P: np.ndarray = self.Q  # Terminal cost
