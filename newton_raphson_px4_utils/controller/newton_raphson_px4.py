from dataclasses import dataclass

from jax import lax
import jax.numpy as jnp
import numpy as np

from newton_raphson_px4_utils.jax_utils import jit
from newton_raphson_px4_utils.controller.nr_utils import (
    predict_output,
    get_tracking_error,
    get_inv_jac_pred_u,
    get_integral_cbf,
)

USE_CBF: bool = True


@dataclass(frozen=True)
class NRProfileConfig:
    name: str
    lookahead_horizon_s: float
    alpha: np.ndarray
    integral_gain: np.ndarray
    integral_limit: np.ndarray
    num_iterations: int
    iteration_damping: float
    use_foh: bool


def build_nr_profile(profile_name: str) -> NRProfileConfig:
    """Return a documented controller profile for repeatable experiments."""
    profile = profile_name.strip().lower()
    if profile == "baseline":
        return NRProfileConfig(
            name="baseline",
            lookahead_horizon_s=1.2,
            alpha=np.array([50.0, 60.0, 60.0, 60.0], dtype=np.float64),
            integral_gain=np.zeros(4, dtype=np.float64),
            integral_limit=np.zeros(4, dtype=np.float64),
            num_iterations=1,
            iteration_damping=1.0,
            use_foh=False,
        )
    if profile == "workshop":
        return NRProfileConfig(
            name="workshop",
            lookahead_horizon_s=0.8,
            alpha=np.array([45.0, 55.0, 55.0, 45.0], dtype=np.float64),
            integral_gain=np.array([0.35, 0.35, 0.50, 0.12], dtype=np.float64),
            integral_limit=np.array([0.75, 0.75, 0.50, 0.30], dtype=np.float64),
            num_iterations=2,
            iteration_damping=0.65,
            use_foh=True,
        )
    raise ValueError(f"Unknown Newton-Raphson profile: {profile_name}")


@jit
def newton_raphson_standard(state,
                            last_input,
                            reference,
                            error_integral,
                            lookahead_horizon_s,
                            lookahead_stage_dt,
                            integration_dt,
                            mass,
                            alpha,
                            integral_gain,
                            integral_limit,
                            num_iterations,
                            iteration_damping,
                            use_foh,
                            ):
    clipped_integral = jnp.clip(error_integral, -integral_limit, integral_limit)

    def nr_iteration(_, carry):
        candidate_input, _ = carry

        y_pred = predict_output(
            state,
            last_input,
            candidate_input,
            lookahead_horizon_s,
            lookahead_stage_dt,
            mass,
            use_foh,
        )
        error = get_tracking_error(reference, y_pred) + integral_gain * clipped_integral
        dgdu_inv = get_inv_jac_pred_u(
            state,
            last_input,
            candidate_input,
            lookahead_horizon_s,
            lookahead_stage_dt,
            mass,
            use_foh,
        )
        nr_step = dgdu_inv @ error
        cbf_term = get_integral_cbf(candidate_input, nr_step) if USE_CBF else jnp.zeros_like(nr_step)
        delta_u = alpha * (nr_step + cbf_term) * integration_dt
        updated_input = candidate_input + iteration_damping * delta_u
        return updated_input, cbf_term

    return lax.fori_loop(
        0,
        num_iterations,
        nr_iteration,
        (last_input, jnp.zeros_like(last_input)),
    )
