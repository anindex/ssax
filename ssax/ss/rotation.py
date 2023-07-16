import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, random


@jit
def rotation_matrix(theta):
    theta = theta[..., jnp.newaxis, jnp.newaxis]
    dim1 = jnp.concatenate([jnp.cos(theta), -jnp.sin(theta)], axis=-2)
    dim2 = jnp.concatenate([jnp.sin(theta), jnp.cos(theta)], axis=-2)
    mat = jnp.concatenate([dim1, dim2], axis=-1)
    return mat


@jit
def get_random_maximal_torus_matrix(origin, angle_range=[0, jnp.pi/2], **kwargs):
    batch, dim = origin.shape
    assert dim % 2 == 0, 'Only work with even dim for random rotation for now.'
    rand_key = kwargs.get('rand_key', random.PRNGKey(0))
    theta = random.uniform(rand_key, shape=(batch, dim // 2), minval=angle_range[0], maxval=angle_range[1], dtype=origin.dtype)
    rot_mat = vmap(rotation_matrix)(theta)
    # make batch block diag
    max_torus_mat = vmap(lambda mats: jsp.linalg.block_diag(*mats))(rot_mat)
    return max_torus_mat
