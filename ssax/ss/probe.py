import jax
from jax import jit, random
import jax.numpy as jnp

from functools import partial

from .utils import default_prng_key


@partial(jit, static_argnames=['num_probe'])
def get_random_probe_points(origin: jax.Array, 
                            points: jax.Array, 
                            probe_radius: float = 2., 
                            num_probe: int = 5, 
                            rng: jax.Array = None) -> jax.Array:
    batch, num_points, dim = points.shape
    alpha = random.uniform(default_prng_key(rng), shape=(batch, num_points, num_probe, 1))
    probe_points = points * probe_radius
    probe_points = probe_points[..., jnp.newaxis, :] * alpha  + origin[..., jnp.newaxis, jnp.newaxis]  # [batch, num_points, num_probe, dim]
    return probe_points


@jit
def get_probe_points(origin: jax.Array, 
                     points: jax.Array,
                     probes: jax.Array,
                     probe_radius: float = 2.) -> jax.Array:
    alpha = probes[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    probe_points = points * probe_radius
    probe_points = probe_points[..., jnp.newaxis, :] * alpha  + origin[..., jnp.newaxis, jnp.newaxis, :]  # [batch, num_points, num_probe, dim]
    return probe_points


@jit
def get_shifted_points(new_origins: jax.Array, points: jax.Array):
    '''
    Args:
        new_origins: [no, dim]
        points: [nb, dim]
    Returns:
        shifted_points: [no, nb, dim]
    '''
    # asumming points has centroid at origin
    shifted_points = points + new_origins[..., jnp.newaxis, :]
    return shifted_points


@partial(jit, static_argnames='num_probe')
def get_projecting_points(X1: jax.Array, 
                          X2: jax.Array, 
                          probe_step_size: float, 
                          num_probe: int = 5) -> jax.Array:
    '''
    X1: [nb1 x dim]
    X2: [nb2 x dim] or [nb1 x nb2 x dim]
    return [nb1 x nb2 x num_probe x dim]
    '''
    X1 = X1[..., jnp.newaxis, jnp.newaxis, :]
    if X2.ndim == 2:
        X2 = X2[..., jnp.newaxis, :, jnp.newaxis, :]
    elif X2.ndim == 3:
        assert X2.shape[0] == X1.shape[0]
        X2 = X2[..., jnp.newaxis, :]
    alpha = jnp.arange(1, num_probe + 1)[jnp.newaxis, jnp.newaxis, :, jnp.newaxis] * probe_step_size
    points = X1 + (X2 - X1) * alpha
    return points
