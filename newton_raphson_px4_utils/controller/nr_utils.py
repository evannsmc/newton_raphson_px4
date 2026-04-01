import jax.numpy as jnp
from jax import jacfwd, lax

GRAVITY = 9.8  # Match Gazebo world (Tools/simulation/gz/worlds/default.sdf)
USING_CBFS = True
C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])


def body2world_angular_rates(roll, pitch, body_rates):
    # body_rates = [p,q,r]
    T = jnp.array([
        [1, jnp.sin(roll)*jnp.tan(pitch),  jnp.cos(roll)*jnp.tan(pitch)],
        [0, jnp.cos(roll),                -jnp.sin(roll)],
        [0, jnp.sin(roll)/jnp.cos(pitch),  jnp.cos(roll)/jnp.cos(pitch)]
    ])
    return T @ body_rates  # [roll_dot, pitch_dot, yaw_dot] in world frame

# TODO: IMPLEMENT RK4 and other better integration methods for ENHANCED WARDI CTRL
def f_quad(state, input, mass):
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    curr_thrust = input[0]
    body_rates = input[1:]

    curr_rolldot, curr_pitchdot, curr_yawdot = body2world_angular_rates(roll, pitch, body_rates)

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / mass) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / mass) * (cr * sy * sp - cy * sr)
    vzdot = GRAVITY - (curr_thrust / mass) * (cr * cp)

    xdot = jnp.array([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])
    return xdot

# --- Output prediction and related functions ---
def dynamics(state, input, mass):
    """Put the dynamics function here."""
    xdot = f_quad(state, input, mass)
    return xdot


def interpolate_input(u_prev, u_next, progress, use_foh):
    """Select the control used during prediction.

    Baseline mode uses the classic zero-order hold on the candidate input.
    Workshop mode uses a first-order hold from the current input toward the
    candidate input to reduce the structural mismatch from assuming a 1.2 s
    constant control command.
    """
    progress = jnp.clip(progress, 0.0, 1.0)
    return lax.cond(
        use_foh,
        lambda _: u_prev + (u_next - u_prev) * progress,
        lambda _: u_next,
        operand=None,
    )


def rk4_pred(state, u_prev, u_next, lookahead_step, integrations_int, mass, use_foh):
    total_steps = jnp.maximum(1, integrations_int).astype(state.dtype)

    def input_at(stage_index, stage_fraction):
        progress = (stage_index.astype(state.dtype) + stage_fraction) / total_steps
        return interpolate_input(u_prev, u_next, progress, use_foh)

    def for_function(i, current_state):
        i_state = jnp.asarray(i, dtype=state.dtype)
        u_k1 = input_at(i_state, 0.0)
        u_k23 = input_at(i_state, 0.5)
        u_k4 = input_at(i_state, 1.0)

        k1 = dynamics(current_state, u_k1, mass)
        k2 = dynamics(current_state + k1 * lookahead_step / 2, u_k23, mass)
        k3 = dynamics(current_state + k2 * lookahead_step / 2, u_k23, mass)
        k4 = dynamics(current_state + k3 * lookahead_step, u_k4, mass)
        return current_state + (k1 + 2*k2 + 2*k3 + k4) * lookahead_step / 6

    pred_state = lax.fori_loop(0, integrations_int, for_function, state)
    return pred_state


def predict_state(state, u_prev, u_next, T_lookahead, lookahead_step, mass, use_foh):
    """Predict the state at t+T using a configurable hold assumption."""
    integrations_int = (T_lookahead / lookahead_step).astype(int)
    pred_state = rk4_pred(state, u_prev, u_next, lookahead_step, integrations_int, mass, use_foh)

    return pred_state


def predict_output(state, u_prev, u_next, T_lookahead, lookahead_step, mass, use_foh):
    """Take output from the predicted states."""
    pred_state = predict_state(state, u_prev, u_next, T_lookahead, lookahead_step, mass, use_foh)
    return C @ pred_state


def get_jac_pred_u(state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh):
    """Get the jacobian of the predicted output with respect to the candidate control input."""
    raw_val = jacfwd(predict_output, 2)(
        state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh
    )
    return raw_val.reshape((4,4))


def get_inv_jac_pred_u(state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh):
    """Get the inverse of the jacobian of the predicted output with respect to the control input."""
    return jnp.linalg.pinv(
        get_jac_pred_u(
            state,
            last_input,
            candidate_input,
            T_lookahead,
            lookahead_step,
            mass,
            use_foh,
        ).reshape((4,4))
    )


# --- Integral Control Barrier Function (I-CBF) related functions ---
def execute_cbf(current, phi, max_value, min_value, gamma, switch_value = 0.0):
    """Execute the control barrier function."""
    zeta_max = gamma * (max_value - current) - phi
    zeta_min = gamma * (min_value - current) - phi
    v = jnp.where(current >= switch_value,
                  jnp.minimum(0, zeta_max),
                  jnp.maximum(0, zeta_min))
    return v

def get_integral_cbf(last_input, phi):
    """Integral control barrier function set-up for all inputs."""
    # Extract values from input
    curr_thrust, curr_roll_rate, curr_pitch_rate, curr_yaw_rate = last_input
    phi_thrust, phi_roll_rate, phi_pitch_rate, phi_yaw_rate = phi

    # CBF parameters
    thrust_gamma = 1.0  # CBF parameter
    thrust_max = 27.0  # max thrust (force) value
    thrust_min = 15.0  # min thrust (force) value
    switch_value = (thrust_max + thrust_min) / 2.0  # mid-point between max and min thrust
    v_thrust = execute_cbf(curr_thrust, phi_thrust, thrust_max, thrust_min, thrust_gamma, switch_value)

    # CBF for rates
    rates_max_abs = 0.8  # max absolute value of roll, pitch, and yaw rates
    rates_max = rates_max_abs
    rates_min = -rates_max_abs
    gamma_rates = 1.0  # CBF parameter

    v_roll = execute_cbf(curr_roll_rate, phi_roll_rate, rates_max, rates_min, gamma_rates)
    v_pitch = execute_cbf(curr_pitch_rate, phi_pitch_rate, rates_max, rates_min, gamma_rates)
    v_yaw = execute_cbf(curr_yaw_rate, phi_yaw_rate, rates_max, rates_min, gamma_rates)

    v = jnp.array([v_thrust, v_roll, v_pitch, v_yaw])
    return v



# --- Error functions with yaw error using quaternions ---
def quaternion_from_yaw(yaw):
    """Converts a yaw angle to a quaternion."""
    half_yaw = yaw / 2.0
    return jnp.array([jnp.cos(half_yaw), 0, 0, jnp.sin(half_yaw)])


def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def yaw_error_from_quaternion(q):
    """Returns the yaw error from the quaternion of angular error."""
    return 2 * jnp.arctan2(q[3], q[0])


def quaternion_normalize(q):
    """Normalizes a quaternion."""
    return q / jnp.linalg.norm(q)


def shortest_path_yaw_quaternion(current_yaw, desired_yaw):
    """Returns the shortest path between two yaw angles with quaternions."""
    q_current = quaternion_normalize(quaternion_from_yaw(current_yaw))
    q_desired = quaternion_normalize(quaternion_from_yaw(desired_yaw))
    q_error = quaternion_multiply(q_desired, quaternion_conjugate(q_current))
    q_error_normalized = quaternion_normalize(q_error)
    return yaw_error_from_quaternion(q_error_normalized)


def get_tracking_error(ref, pred):
    """Calculates the tracking error between the reference and predicted outputs with yaw error handled by quaternions."""
    err = ref - pred
    err = err.at[3].set(shortest_path_yaw_quaternion(pred[3], ref[3]))
    return err
