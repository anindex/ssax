from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "is_jax_array", "default_prng_key", "default_progress_fn"
]


def is_jax_array(obj: Any) -> bool:
  """Check if an object is a Jax array."""
  if hasattr(jax, "array"):
    return isinstance(obj, jnp.array)
  return isinstance(obj, jnp.Devicearray)


def default_prng_key(
    rng: Optional[jax.random.PRNGKeyArray] = None,
) -> jax.random.PRNGKeyArray:
    """Return a default PRNG key."""
    return jax.random.PRNGKey(0) if rng is None else rng


def default_progress_fn(
    status: Tuple[np.array, np.array, np.array, NamedTuple], *args: Any
) -> None:
    """Default progress function."""
    iteration, inner_iterations, total_iter, state = status
    iteration = int(iteration) + 1
    inner_iterations = int(inner_iterations)
    total_iter = int(total_iter)
    errors = np.array(state.errors).ravel()

    # Avoid reporting error on each iteration,
    # because errors are only computed every `inner_iterations`.
    if iteration % inner_iterations == 0:
        error_idx = max(0, iteration // inner_iterations - 1)
        error = errors[error_idx]

        print(f"{iteration} / {total_iter} -- {error}")  # noqa: T201
