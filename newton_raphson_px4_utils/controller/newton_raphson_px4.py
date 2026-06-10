import jax
from jax import lax
import jax.numpy as jnp

from newton_raphson_px4_utils.jax_utils import jit
from newton_raphson_px4_utils.controller.nr_utils import(
    predict_output, get_tracking_error, get_inv_jac_pred_u, get_integral_cbf
)

# ALPHA = jnp.array([20.0, 30.0, 30.0, 30.0])

ALPHA = jnp.array([50.0, 60.0, 60.0, 60.0])
USE_CBF: bool = True



@jit
def newton_raphson_standard(state,
                            last_input,
                            reference,
                            lookahead_horizon_s,
                            lookahead_stage_dt,
                            integration_dt,
                            mass,
                            ):

    y_pred = predict_output(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
    error = get_tracking_error(reference, y_pred)
    dgdu_inv = get_inv_jac_pred_u(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
    NR = dgdu_inv @ error
    v = get_integral_cbf(last_input, NR) if USE_CBF else jnp.zeros_like(NR)
    udot = NR + v
    change_u = udot * integration_dt

    u = last_input + ALPHA * change_u

    return u, v
