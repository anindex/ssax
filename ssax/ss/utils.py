from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


def default_prng_key(rng: jax.Array = None) -> jax.Array:
    """Return a default PRNG key."""
    return jax.random.PRNGKey(0) if rng is None else rng
