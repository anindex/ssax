import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, random

from typing import Any

from .utils import default_prng_key


@jit
def rotation_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    theta = theta[..., jnp.newaxis, jnp.newaxis]
    dim1 = jnp.concatenate([jnp.cos(theta), -jnp.sin(theta)], axis=-2)
    dim2 = jnp.concatenate([jnp.sin(theta), jnp.cos(theta)], axis=-2)
    mat = jnp.concatenate([dim1, dim2], axis=-1)
    return mat


@jit
def get_random_maximal_torus_matrix(origin: jnp.ndarray, 
                                    angle_range=[0, 2 * jnp.pi],
                                    rng: Any = None,
                                    **kwargs) -> jnp.ndarray:
    batch, dim = origin.shape
    assert dim % 2 == 0, 'Only work with even dim for random rotation for now.'

    theta = random.uniform(default_prng_key(rng), shape=(batch, dim // 2), minval=angle_range[0], maxval=angle_range[1], dtype=origin.dtype)
    rot_mat = vmap(rotation_matrix)(theta)
    # make batch block diag
    max_torus_mat = vmap(lambda mats: jsp.linalg.block_diag(*mats))(rot_mat)
    return max_torus_mat
