import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, random
from functools import partial
from typing import Any

from ssax.ss.utils import default_prng_key


@jit
def rotation_matrix(theta: jnp.array) -> jnp.array:
    theta = theta[..., jnp.newaxis, jnp.newaxis]
    dim1 = jnp.concatenate([jnp.cos(theta), -jnp.sin(theta)], axis=-2)
    dim2 = jnp.concatenate([jnp.sin(theta), jnp.cos(theta)], axis=-2)
    mat = jnp.concatenate([dim1, dim2], axis=-1)
    return mat


@jit
def get_random_maximal_torus_matrix(origin: jnp.array, 
                                    angle_range=[0, 2 * jnp.pi],
                                    rng: Any = None,
                                    **kwargs) -> jnp.array:
    # NOTE: this does not work for odd dim.
    # NOTE: this function does not draw a uniform random rotation matrix on SO(dim). But it still works in practice.
    batch, dim = origin.shape
    assert dim % 2 == 0, 'Only work with even dim for random rotation for maximal torus rot matrix.'

    theta = random.uniform(default_prng_key(rng), shape=(batch, dim // 2), minval=angle_range[0], maxval=angle_range[1], dtype=origin.dtype)
    rot_mat = vmap(rotation_matrix)(theta)
    # make batch block diag
    max_torus_mat = vmap(lambda mats: jsp.linalg.block_diag(*mats))(rot_mat)
    return max_torus_mat


@partial(jit, static_argnames='dim')
def get_random_uniform_rot_matrix(dim: int,
                                  rng: Any = None,
                                  **kwargs) -> jnp.array:
    """Compute a uniformly random rotation matrix drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    See: Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization"""

    def body_fun(i, vals):
        rng, H, D = vals
        rng, subkey = random.split(rng)
        v = random.normal(subkey, shape=(dim,))
        D = D.at[i - 1].set(jnp.sign(v[0]))
        v = v.at[0].add(-D[i - 1] * jnp.sqrt((v * v).sum()))
        # Householder transformation
        Hx = jnp.eye(dim) - 2 * jnp.outer(v, v) / (v * v).sum()
        H = jnp.dot(H, Hx)
        return rng, H, D

    rng = default_prng_key(rng)
    H = jnp.eye(dim)
    D = jnp.ones((dim,))
    _, H, D = jax.lax.fori_loop(1, dim, body_fun, (rng, H, D))
    # Fix the last sign such that the determinant is 1
    D = D.at[-1].set((-1)**(1 - dim % 2) * jnp.prod(D))
    R = (D * H.T).T
    return R


if __name__ == '__main__':
    print(jnp.linalg.det(get_random_uniform_rot_matrix(10)))
