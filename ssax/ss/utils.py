from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

__all__ = [
    "is_jax_array", "default_prng_key"
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
