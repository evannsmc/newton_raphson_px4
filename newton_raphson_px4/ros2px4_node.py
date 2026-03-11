
from rclpy.node import Node
from rclpy.qos import(
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy
)
from px4_msgs.msg import(
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleRatesSetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleOdometry,
    RcChannels,
    BatteryStatus
)
from newton_raphson_px4_utils.px4_utils.core_funcs import (
    engage_offboard_mode,
    arm,
    land,
    disarm,
    publish_offboard_control_heartbeat_signal_position,
    publish_offboard_control_heartbeat_signal_bodyrate
)
from quad_platforms import (
    PlatformType,
    PlatformConfig,
    PLATFORM_REGISTRY
)
from quad_trajectories import (
    TrajContext,
    TrajectoryType,
    TRAJ_REGISTRY,
    generate_reference_trajectory,
    flat_to_x_u,
)
from newton_raphson_px4_utils.controller.newton_raphson_px4 import newton_raphson_standard


from newton_raphson_px4_utils.main_utils import BANNER
from newton_raphson_px4_utils.transformations.adjust_yaw import adjust_yaw
from newton_raphson_px4_utils.px4_utils.flight_phases import FlightPhase


import time
import jax
import math as m
import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.energy_meter import EnergyContext


from ros2_logger import LogType, VectorLogType # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

GRAVITY: float = 9.806

class OffboardControl(Node):
    def __init__(self, platform_type: PlatformType, trajectory: TrajectoryType = TrajectoryType.HOVER, hover_mode: int|None = None,
                double_speed: bool = True, short: bool = False, spin: bool = False,
                pyjoules: bool = False, csv_handler: CSVHandler|None = None, logging_enabled: bool = False,
                flight_period_: bool|None = None, feedforward: bool = False) -> None:

        super().__init__('offboard_control_node')
        self.get_logger().info(f"{BANNER}Initializing ROS 2 node: '{self.__class__.__name__}'{BANNER}")
        self.sim = platform_type==PlatformType.SIM
        self.platform_type = platform_type
        self.feedforward = feedforward
        self.trajectory_type = trajectory
        self.hover_mode = hover_mode
        self.double_speed = double_speed
        self.short = short
        self.spin = spin
        self.pyjoules_on = pyjoules
        self.logging_enabled = logging_enabled
        flight_period = flight_period_ if flight_period_ is not None else 30.0 if self.sim else 60.0
        
        if self.pyjoules_on:
            print("PyJoules energy monitoring ENABLED")
            self.csv_handler = csv_handler


        # Initialize platform configuration using dependency injection
        platform_class = PLATFORM_REGISTRY[self.platform_type]
        self.platform: PlatformConfig = platform_class()

        # Map trajectory string to TrajectoryType enum
        trajectory_map = {traj_type.value: traj_type for traj_type in TrajectoryType}
        if trajectory not in trajectory_map:
            raise ValueError(f"Unknown trajectory: {trajectory}. Available: {list(trajectory_map.keys())}")
        self.ref_type = trajectory_map[trajectory]
        print(f"\n[Trajectory] Main trajectory type: {self.ref_type.name}")


        # ----------------------- ROS2 Node Stuff --------------------------
        qos_profile = QoSProfile( # Configure QoS profile for publishing and subscribing
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ----------------------- Publishers --------------------------
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)


        # ----------------------- Subscribers --------------------------
        # Mocap variables
        self.mocap_k: int = -1
        self.full_rotations: int = 0
        self.mocap_initialized: bool = False
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback, qos_profile)

        self.in_offboard_mode: bool = False
        self.armed: bool = False
        self.in_land_mode: bool = False
        self.vehicle_status = None
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)

        self.offboard_mode_rc_switch_on = True if self.sim else False
        self.mode_channel = 5
        self.rc_channels_subscriber = self.create_subscription(
            RcChannels, '/fmu/out/rc_channels',
            self.rc_channel_callback, qos_profile)
        
        self.battery_status_subsriber = self.create_subscription(
            BatteryStatus, '/fmu/out/battery_status',
            self.battery_status_callback, qos_profile)

        # ----------------------- Set up Flight Phases and Time --------------------------
        self.T0 = time.time()
        self.program_time: float = 0.0
        self.cushion_period = 10.0
        self.flight_period = flight_period
        self.land_time = self.flight_period + 2 * self.cushion_period
        self.flight_phase = self.get_phase()
        print(f"Flight time: {self.land_time}s")

        # ----------------------- Run Timers --------------------------
        self.data_log_timer_period = 1.0 / 10.0  # 10 Hz data logging
        self.data_log_timer = self.create_timer(self.data_log_timer_period,
                                                self.data_log_timer_callback) if self.logging_enabled else None

        self.offboard_setpoint_counter = 0
        self.offboard_timer_period = 1.0 / 10.0  # 10 Hz offboard heartbeat
        self.timer = self.create_timer(self.offboard_timer_period,
                                      self.offboard_mode_timer_callback)

        # Separate controls publishing and computation loops
        self.publish_control_timer_period = 1.0 / 100.0  # 100 Hz publishing loop
        self.publish_control_timer = self.create_timer(self.publish_control_timer_period,
                                                       self.publish_control_timer_callback)

        self.compute_control_timer_period = 1.0 / 100.0  # 100 Hz NMPC computation (~10ms)
        self.compute_control_timer = self.create_timer(self.compute_control_timer_period,
                                            self.compute_control_timer_callback)


        # ----------------------- Initialize Control --------------------------
        self.HOVER_HEIGHT = 3.0 if self.sim else 0.7
        self.LAND_HEIGHT = 0.6 if self.sim else 0.45

        # Trajectory tracking
        self.trajectory_started: bool = False
        self.trajectory_time: float = 0.0
        self.reference_time: float = 0.0

        # JIT compilation test variables
        self.T_LOOKAHEAD = 1.2
        self.LOOKAHEAD_STATE_DT = 0.05

        self.first_thrust = self.platform.mass * GRAVITY
        self.last_input = np.array([self.first_thrust, 0.0, 0.0, 0.0])
        self.normalized_input = [self.platform.get_throttle_from_force(self.first_thrust), 0.0, 0.0, 0.0]
        self.x_ff = None      # feedforward state (set when F8_CONTRACTION is active)
        self.u_ff = None      # feedforward control (set when F8_CONTRACTION is active)
        self.u_dev = None     # accumulated NR correction relative to feedforward operating point
        self._ff_jit = None  # JIT-compiled flat_to_x_u (created in jit_compile_trajectories)
        self._traj_jit = None  # JIT-compiled trajectory generation (persists across mode switch)

        self.jit_compile_controller()

        # ----------------------- Initialize Trajectory --------------------------
        self.jit_compile_trajectories()
        print("[Offboard Control Node] Node initialized successfully!\n")
        time.sleep(3)  # Allow time to inspect JIT-compilation output

        self.T0 = time.time() # Reset program time after JIT compilation
        

        # ----------------------- Set up Logging Arrays --------------------------
        if self.logging_enabled:
            print("Data logging is ON")
            self.data_log_timer_period = .1
            self.first_log = True
            if self.first_log:
                self.first_log = False
                self.get_logger().info("Starting data logging.")
                # Metadata (constant, appended once)
                self.platform_logtype = LogType("platform", 0)
                self.trajectory_logtype = LogType("trajectory", 1)
                self.traj_double_logtype = LogType("traj_double", 2)
                self.traj_short_logtype = LogType("traj_short", 3)
                self.traj_spin_logtype = LogType("traj_spin", 4)
                self.lookahead_time_logtype = LogType("lookahead_time", 5)
                self.controller_logtype = LogType("controller", 6)

                self.platform_logtype.append(self.platform_type.value.upper())
                self.trajectory_logtype.append(self.ref_type.name)
                self.traj_double_logtype.append("DblSpd" if self.double_speed else "NormSpd")
                self.traj_short_logtype.append("Short" if self.short else "Not Short")
                self.traj_spin_logtype.append("Spin" if self.spin else "NoSpin")
                self.lookahead_time_logtype.append(self.T_LOOKAHEAD)
                self.controller_logtype.append("nr")

            # Timing
            self.program_time_logtype = LogType("time", 10)
            self.trajectory_time_logtype = LogType("traj_time", 11)
            self.reference_time_logtype = LogType("ref_time", 12)
            self.comp_time_logtype = LogType("comp_time", 13)

            # State
            self.x_logtype = LogType("x", 20)
            self.y_logtype = LogType("y", 21)
            self.z_logtype = LogType("z", 22)
            self.yaw_logtype = LogType("yaw", 23)
            self.vx_logtype = LogType("vx", 24)
            self.vy_logtype = LogType("vy", 25)
            self.vz_logtype = LogType("vz", 26)

            # Reference
            self.xref_logtype = LogType("x_ref", 30)
            self.yref_logtype = LogType("y_ref", 31)
            self.zref_logtype = LogType("z_ref", 32)
            self.yawref_logtype = LogType("yaw_ref", 33)
            self.vxref_logtype = LogType("vx_ref", 34)
            self.vyref_logtype = LogType("vy_ref", 35)
            self.vzref_logtype = LogType("vz_ref", 36)

            # Body rates (actual)
            self.p_logtype = LogType("p", 40)
            self.q_logtype = LogType("q", 41)
            self.r_logtype = LogType("r", 42)

            # Control inputs (normalized)
            self.throttle_input_logtype = LogType("throttle_input", 50)
            self.p_input_logtype = LogType("p_input", 51)
            self.q_input_logtype = LogType("q_input", 52)
            self.r_input_logtype = LogType("r_input", 53)

            # CBF: subnames produce columns cbf_v_throttle, cbf_v_p, cbf_v_q, cbf_v_r
            self.cbf_logtype = VectorLogType("cbf", 60, ['v_throttle', 'v_p', 'v_q', 'v_r'])




    def time_and_compare(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)

        # Force actual execution before stopping timer
        result = jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )

        end = time.perf_counter()
        elapsed = end - start

        return *result, elapsed

    def jit_compile_controller(self) -> None:
        """ Perform a dummy call to all JIT-compiled controller functions to trigger compilation
            We also time them before and after JIT to ensure performance"""
        self._state0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self._input0 = np.array([self.first_thrust, 0.1, 0.2, 0.3])
        self._ref0 = np.array([2.0, 2.0, -6.0, 0.0])


        print("\n[JIT Compilation] Pre-compiling controller function...")
        input, cbf_term, total_time1 = self.time_and_compare(
            newton_raphson_standard,
            jnp.array(self._state0),
            jnp.array(self._input0),
            jnp.array(self._ref0),
            jnp.array(self.T_LOOKAHEAD),
            jnp.array(self.LOOKAHEAD_STATE_DT),
            jnp.array(self.compute_control_timer_period),
            jnp.array(self.platform.mass),
        )
        print(f"  Result (NO JIT): {input=},\n  {cbf_term=}")
        print(f"  Time Taken (NO JIT): {total_time1:.4f}s")

        input, cbf_term, total_time2 = self.time_and_compare(
            newton_raphson_standard,
            jnp.array(self._state0),
            jnp.array(self._input0),
            jnp.array(self._ref0),
            jnp.array(self.T_LOOKAHEAD),
            jnp.array(self.LOOKAHEAD_STATE_DT),
            jnp.array(self.compute_control_timer_period),
            jnp.array(self.platform.mass),
        )
        print(f"  Result (JIT):\n  {input=},\n  {cbf_term=}")
        print(f"  Time Taken (JIT):\n  {total_time2:.4f}s")

        print(f"This is a speed up of {(total_time1)/(total_time2):.2f}x")
        print(f"Good for {(1.0/total_time2):.2f} Hz control loop")
        # exit(0)


    def jit_compile_trajectories(self) -> None:
        """ Perform a dummy call to all JIT-compiled trajectory functions to trigger compilation
            We also time them before and after JIT to ensure performance"""

        print("\n[JIT Compilation] Pre-compiling trajectory functions...")

        # Pre-compile hover trajectory
        print("  Compiling hover trajectory...")
        ref, ref_dot, hover_total_time_1 = self.time_and_compare(
            self.generate_ref_trajectory,
            TrajectoryType.HOVER,
            hover_mode=self.hover_mode
        )
        print(f"  Hover trajectory (NO JIT): {ref = }, {ref_dot = }")
        print(f"  Hover trajectory (NO JIT): {hover_total_time_1:.4f}s")


        print("  Compiling regular trajectory...")
        ref, ref_dot, regular_total_time_1 = self.time_and_compare(
            self.generate_ref_trajectory,
            self.ref_type,
        )
        print(f"  Regular trajectory (NO JIT): {ref = }, {ref_dot = }")
        print(f"  Regular trajectory (NO JIT): {regular_total_time_1:.4f}s")



        print(f"  Testing JIT-compiled trajectory functions...")
        ref, ref_dot, hover_total_time_2 = self.time_and_compare(
            self.generate_ref_trajectory,
            TrajectoryType.HOVER,
            hover_mode=self.hover_mode
        )
        print(f"  Hover trajectory (JIT): {ref = }, {ref_dot = }")
        print(f"  Hover trajectory (JIT): {hover_total_time_2:.4f}s")
        print(f"  Hover speed up: {(hover_total_time_1)/(hover_total_time_2):.2f}x")

        ref, ref_dot, regular_total_time_2 = self.time_and_compare(
            self.generate_ref_trajectory,
            self.ref_type,
        )
        print(f"  Regular trajectory (JIT): {ref = }, {ref_dot = }")
        print(f"  Regular trajectory (JIT): {regular_total_time_2:.4f}s")
        print(f"  Regular speed up: {(regular_total_time_1)/(regular_total_time_2):.2f}x")

        if self.ref_type == TrajectoryType.F8_CONTRACTION and self.feedforward:
            print("  Compiling feedforward (flat_to_x_u)...")
            ctx = TrajContext(sim=self.sim, hover_mode=self.hover_mode, spin=self.spin,
                              double_speed=False, short=self.short)
            flat_output = lambda t: TRAJ_REGISTRY[TrajectoryType.F8_CONTRACTION](t, ctx)
            self._ff_jit = jax.jit(lambda t: flat_to_x_u(t, flat_output))

            x_ff, u_ff, ff_time_1 = self.time_and_compare(self._ff_jit, 0.0)
            print(f"  Feedforward (NO JIT): {ff_time_1:.4f}s")
            x_ff, u_ff, ff_time_2 = self.time_and_compare(self._ff_jit, 0.0)
            print(f"  Feedforward (JIT): {ff_time_2:.4f}s")
            print(f"  Feedforward speed up: {ff_time_1 / ff_time_2:.2f}x")

        # Store a single JIT-compiled trajectory function for the main trajectory.
        # By capturing traj_fn and ctx in the closure here (they never change), JAX
        # traces and compiles the XLA program exactly ONCE during init and reuses it
        # on every control-loop call — eliminating retrace delays at mode switch.
        print("  Storing persistent JIT for main trajectory...")
        _ctx_main = TrajContext(
            sim=self.sim, hover_mode=self.hover_mode, spin=self.spin,
            double_speed=False if self.ref_type == TrajectoryType.F8_CONTRACTION else self.double_speed,
            short=self.short)
        _traj_fn_main = TRAJ_REGISTRY[self.ref_type]
        self._traj_jit = jax.jit(
            lambda t_start: generate_reference_trajectory(_traj_fn_main, t_start, 0.0, 1, _ctx_main))
        r, rd = self._traj_jit(0.0)
        jax.block_until_ready((r, rd))  # first call: compiles
        r, rd = self._traj_jit(0.0)
        jax.block_until_ready((r, rd))  # second call: confirms fast
        print(f"  Main trajectory JIT ready.")


    # ========== Subscriber Callbacks ==========
    def vehicle_odometry_callback(self, msg):
        """Process odometry and convert to quaternion state."""
        # Position and velocity
        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2]

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

        self.wx = msg.angular_velocity[0]
        self.wy = msg.angular_velocity[1]
        self.wz = msg.angular_velocity[2]

        self.qw = msg.q[0]
        self.qx = msg.q[1]
        self.qy = msg.q[2]
        self.qz = msg.q[3]

        orientation = R.from_quat(msg.q, scalar_first=True)
        self.roll, self.pitch, self._yaw = orientation.as_euler('xyz', degrees=False)
        self.yaw = adjust_yaw(self, self._yaw)

        # State vector blocks
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.euler_angle_raw = np.array([self.roll, self.pitch, self._yaw])
        self.euler_angle_total_yaw = np.array([self.roll, self.pitch, self.yaw])
        self.quat = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        self.angular_velocities = np.array([self.wx, self.wy, self.wz])

        self.state_output = np.hstack((self.position, self.yaw))
        self.nr_state = np.hstack((self.position, self.velocity, self.euler_angle_total_yaw))
        self.ROT = orientation

        self.get_logger().info(f'\nState output: {self.state_output}', throttle_duration_sec=0.3)



    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        self.in_offboard_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.armed = (self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_land_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND)

    def rc_channel_callback(self, rc_channels):
        flight_mode = rc_channels.channels[self.mode_channel - 1]
        self.offboard_mode_rc_switch_on = True if flight_mode >= 0.75 else False

    def battery_status_callback(self, battery_status):
        self.current_voltage = battery_status.voltage_v

    # ========== Timer Callbacks ==========
    def get_phase(self) -> FlightPhase:
        """Determine the current flight phase based on elapsed time."""
        if self.program_time < self.cushion_period:
            return FlightPhase.HOVER
        elif self.program_time < self.cushion_period + self.flight_period:
            return FlightPhase.CUSTOM
        elif self.program_time < self.land_time:
            return FlightPhase.RETURN
        else:
            return FlightPhase.LAND

    def time_before_next_phase(self, current_phase: FlightPhase) -> float:
        """Get the time remaining before the next flight phase."""
        if current_phase == FlightPhase.HOVER:
            return self.cushion_period - self.program_time
        elif current_phase == FlightPhase.CUSTOM:
            return (self.cushion_period + self.flight_period) - self.program_time
        elif current_phase == FlightPhase.RETURN:
            return self.land_time - self.program_time
        else:
            return 0.0



    def killswitch_and_flight_phase(self) -> bool:
        """Check kill switch and update flight phase."""

        if not self.offboard_mode_rc_switch_on:
            self.get_logger().warning(f"\nOffboard Callback: RC Flight Mode Channel {self.mode_channel} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land)")
            self.offboard_setpoint_counter = 0
            return False

        self.program_time = time.time() - self.T0
        self.flight_phase = self.get_phase()

        self.get_logger().warn(f"\n[{self.program_time:.2f}s] In {self.flight_phase.name} phase for the next {self.time_before_next_phase(self.flight_phase):.2f}s", throttle_duration_sec=0.5)

        return True

    def get_offboard_health(self) -> bool:
        """Check if vehicle is in offboard mode, armed, and has odometry data."""
        healthy = True

        if not self.in_offboard_mode:
            self.get_logger().warning("Vehicle is NOT in OFFBOARD mode.")
            healthy = False

        if not self.armed:
            self.get_logger().warning("Vehicle is NOT ARMED.")
            healthy = False

        if not self.mocap_initialized:
            self.get_logger().warning("Odometry is NOT received.")
            healthy = False

        return healthy




    def offboard_mode_timer_callback(self) -> None:
        """10Hz timer for offboard mode management."""
        """Callback function for the timer."""
        if not self.killswitch_and_flight_phase(): # Ensure we are in offboard mode and armed, and have odometry data
            return

        if self.offboard_setpoint_counter == 10:
            engage_offboard_mode(self)
            arm(self)
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        if self.flight_phase is FlightPhase.HOVER:
            publish_offboard_control_heartbeat_signal_position(self)

        elif self.flight_phase is FlightPhase.CUSTOM:
            publish_offboard_control_heartbeat_signal_bodyrate(self)

        elif self.flight_phase is FlightPhase.RETURN:
            publish_offboard_control_heartbeat_signal_position(self)

        elif self.flight_phase is FlightPhase.LAND:
            publish_offboard_control_heartbeat_signal_position(self)

        else:
            raise ValueError("Unknown flight phase")

    def data_log_timer_callback(self) -> None:
        """Callback function for the data logging timer."""
        if self.flight_phase is not FlightPhase.CUSTOM:
            return

        # Timing
        self.program_time_logtype.append(self.program_time)
        self.trajectory_time_logtype.append(self.trajectory_time)
        self.reference_time_logtype.append(self.reference_time)
        self.comp_time_logtype.append(self.compute_time)

        # State
        self.x_logtype.append(self.x)
        self.y_logtype.append(self.y)
        self.z_logtype.append(self.z)
        self.yaw_logtype.append(self.yaw)
        self.vx_logtype.append(self.vx)
        self.vy_logtype.append(self.vy)
        self.vz_logtype.append(self.vz)

        # Reference
        self.xref_logtype.append(self.ref[0])
        self.yref_logtype.append(self.ref[1])
        self.zref_logtype.append(self.ref[2])
        self.yawref_logtype.append(self.ref[3])
        self.vxref_logtype.append(self.ref_dot[0])
        self.vyref_logtype.append(self.ref_dot[1])
        self.vzref_logtype.append(self.ref_dot[2])

        # Body rates (actual)
        self.p_logtype.append(self.wx)
        self.q_logtype.append(self.wy)
        self.r_logtype.append(self.wz)

        # Control inputs
        self.throttle_input_logtype.append(self.normalized_input[0])
        self.p_input_logtype.append(self.normalized_input[1])
        self.q_input_logtype.append(self.normalized_input[2])
        self.r_input_logtype.append(self.normalized_input[3])

        # CBF
        self.cbf_logtype.append(*self.cbf_term)


    # --- Setpoint Publisher Functions --- #
    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float):
        """Publish the position setpoint."""
        log_throttle_val = 1.0
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z, yaw]}", throttle_duration_sec=log_throttle_val)


    def publish_rates_setpoint(self, thrust: float, roll: float, pitch: float, yaw: float) -> None:
        """Publish the thrust and body rate setpoint."""
        log_throttle_val = 1.0
        msg = VehicleRatesSetpoint()
        msg.roll = float(roll)
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -1* float(thrust)

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_setpoint_publisher.publish(msg)

        self.get_logger().info(f"Publishing rates setpoints [thrust, r,p,y]: {[thrust, roll, pitch, yaw]}", throttle_duration_sec=log_throttle_val)



    def publish_control_timer_callback(self) -> None:

        if self.in_land_mode:
            self.get_logger().info("In land mode...")
            if abs(self.z) < 0.71 if self.sim else abs(self.z) < 0.64:
                self.get_logger().info("Landed, disarming...")
                disarm(self)
                exit(0)

        if not self.killswitch_and_flight_phase(): # Ensure we are in offboard mode and armed, and have odometry data
            return
        if not self.get_offboard_health():
            return

        if self.flight_phase is FlightPhase.HOVER:
            hover_pose = [0., 0., -self.HOVER_HEIGHT, 0.]
            self.publish_position_setpoint(*hover_pose)

        elif self.flight_phase is FlightPhase.CUSTOM:
            self.publish_rates_setpoint(*self.normalized_input)

        elif self.flight_phase is FlightPhase.RETURN:
            hover_pose = [0., 0., -self.HOVER_HEIGHT, 0.]
            self.publish_position_setpoint(*hover_pose)

        elif self.flight_phase is FlightPhase.LAND:
            land_pose = [0., 0., -self.LAND_HEIGHT, 0.]
            self.publish_position_setpoint(*land_pose)
            if abs(self.z) < 0.64 :
                land(self)




    def compute_control_timer_callback(self) -> None:

        if not self.killswitch_and_flight_phase(): # Ensure we are in offboard mode and armed, and have odometry data
            return
        if not self.get_offboard_health():
            return
        if self.get_phase() is not FlightPhase.CUSTOM:
            return

        throttle_val = 0.3

        if not self.trajectory_started:
            self.trajectory_T0 = time.time()
            self.trajectory_time = 0.0
            self.trajectory_started = True

        self.trajectory_time = time.time() - self.trajectory_T0
        self.reference_time = self.trajectory_time + self.T_LOOKAHEAD
        # self.get_logger().warning(f"\nTrajectory time: {self.trajectory_time:.2f}s, Reference time: {self.reference_time:.2f}s", throttle_duration_sec=throttle_val)
        # self.get_logger().warning(f"[{self.program_time:.2f}s] Computing control. Trajectory time: {self.trajectory_time:.2f}s", throttle_duration_sec=throttle_val)

        if self._traj_jit is not None:
            ref, ref_dot = self._traj_jit(self.reference_time)
        else:
            ref, ref_dot = self.generate_ref_trajectory(self.ref_type)
        self.ref = np.array(ref).flatten()
        self.ref_dot = np.array(ref_dot).flatten()

        if self.ref_type == TrajectoryType.F8_CONTRACTION and self.feedforward and self._ff_jit is not None:
            x_ff, u_ff = self._ff_jit(self.reference_time)
            self.x_ff = x_ff  # [px,py,pz, vx,vy,vz, f_specific, phi, th, psi]
            self.u_ff = u_ff  # [df, dphi, dth, dpsi]
        else:
            self.x_ff = None
            self.u_ff = None
            self.u_dev = None


        t0 = time.time()
        self.controller_handler()
        self.compute_time = time.time() - t0
        self.last_input = self.new_input
        # self.get_logger().warning(f"\nNew control input: {self.new_input}", throttle_duration_sec=throttle_val)
        self.get_logger().warning(f"\nControl computation time: {self.compute_time:.4f}s, Good for {1.0/self.compute_time:.2f} Hz control loop", throttle_duration_sec=throttle_val)

        # NOW CONVERT TO NORMALIZED INPUTS for PX4
        new_force = float(self.new_input[0])
        new_throttle_raw = float(self.platform.get_throttle_from_force(new_force))

        battery_compensation = 1 - 0.0779 * (self.current_voltage - 16.0)
        new_throttle = new_throttle_raw * battery_compensation
    

        new_roll_rate = float(self.new_input[1])
        new_pitch_rate = float(self.new_input[2])
        new_yaw_rate = float(self.new_input[3])

        self.normalized_input = [new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate]
        # self.get_logger().warning(f"\nNormalized control input (throttle, p, q, r): {self.normalized_input}", throttle_duration_sec=throttle_val)

    def controller_handler(self):
        """Wrapper for controller computation."""
        if self.pyjoules_on:
            with EnergyContext(handler=self.csv_handler, domains=[RaplPackageDomain(0)]): # type: ignore #
                self.controller()
        else:
            self.controller()

    def controller(self):
        """Compute control input using Newton-Raphson with optional feedforward."""
        if self.u_ff is not None:
            # u_ff[1:4] = [dphi, dth, dpsi] are Euler angle rates (world-frame).
            # NR and the drone dynamics work in body rates [p, q, r].
            # We must invert the kinematic transformation T before injecting:
            #
            #   [phi_dot, theta_dot, psi_dot] = T(roll, pitch) @ [p, q, r]
            #   =>  [p, q, r] = T^{-1} @ [dphi, dth, dpsi]
            #
            # T is the same matrix used in nr_utils.body2world_angular_rates.
            roll  = float(self.nr_state[6])
            pitch = float(self.nr_state[7])
            sr, cr = m.sin(roll),  m.cos(roll)
            sp, cp = m.sin(pitch), m.cos(pitch)
            tp     = sp / cp  # tan(pitch)
            T = np.array([
                [1.,  sr * tp,  cr * tp],
                [0.,  cr,      -sr     ],
                [0.,  sr / cp,  cr / cp],
            ])
            euler_rates_ff = np.array(self.u_ff[1:4])          # [dphi, dth, dpsi]
            body_rates_ff  = np.linalg.solve(T, euler_rates_ff) # [p_ff, q_ff, r_ff]
            thrust_ff      = self.platform.mass * float(self.x_ff[6])  # f_specific → F (N)
            u_ff_vec = np.array([thrust_ff, body_rates_ff[0], body_rates_ff[1], body_rates_ff[2]])

            # Preserve the accumulated NR correction on top of the moving feedforward
            # operating point instead of resetting back to u_ff every control step.
            if self.u_dev is None:
                self.u_dev = np.array(self.last_input) - u_ff_vec

            last_input = jnp.array(u_ff_vec + self.u_dev)
        else:
            self.u_dev = None
            last_input = jnp.array(self.last_input)

        new_input, cbf_term = newton_raphson_standard(
            jnp.array(self.nr_state),
            last_input,
            jnp.array(self.ref),
            jnp.array(self.T_LOOKAHEAD),
            jnp.array(self.LOOKAHEAD_STATE_DT),
            jnp.array(self.compute_control_timer_period),
            jnp.array(self.platform.mass),
        )
        if self.u_ff is not None:
            self.u_dev = np.array(new_input) - u_ff_vec
        self.new_input = new_input
        self.cbf_term = cbf_term



    def generate_ref_trajectory(self, traj_type: TrajectoryType, **ctx_overrides):
        """Generate reference trajectory."""
        ctx_dict = {
            'sim': self.sim,
            'hover_mode': ctx_overrides.get('hover_mode', self.hover_mode),
            'spin': ctx_overrides.get('spin', self.spin),
            'double_speed': ctx_overrides.get('double_speed', True),
            'short': ctx_overrides.get('short', self.short)
        }
        ctx = TrajContext(**ctx_dict)
        traj_func = TRAJ_REGISTRY[traj_type]

        return generate_reference_trajectory(
            traj_func=traj_func,
            t_start=self.reference_time,
            horizon=0.0,
            num_steps=1,
            ctx=ctx
        )
