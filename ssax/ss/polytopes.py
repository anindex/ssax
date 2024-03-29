import jax
from jax import jit, random, vmap
import jax.numpy as jnp

from itertools import product
from functools import partial
from typing import Tuple

from .probe import get_random_probe_points, get_probe_points
from .rotation import get_random_uniform_rot_matrix
from .utils import default_prng_key


@jit
def get_cube_vertices(origin: jax.Array, 
                      radius: float = 1., 
                      **kwargs) -> jax.Array:
    dim = origin.shape[-1]
    points = jnp.array(list(product([1, -1], repeat=dim)), dtype=origin.dtype) / jnp.sqrt(dim)
    points = points * radius + origin
    return points


@jit
def get_orthoplex_vertices(origin: jax.Array, 
                           radius: float = 1., 
                           **kwargs) -> jax.Array:
    dim = origin.shape[-1]
    points = jnp.zeros((2 * dim, dim), dtype=origin.dtype)
    first = jnp.arange(0, dim)
    second = jnp.arange(dim, 2 * dim)
    points = points.at[first, first].set(radius)
    points = points.at[second, first].set(-radius)
    points = points + origin
    return points


@jit
def get_simplex_vertices(origin: jax.Array, radius: float = 1., **kwargs) -> jax.Array:
    '''
    Simplex coordinates: https://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_a_regular_n-dimensional_simplex_in_Rn
    '''
    dim = origin.shape[-1]
    points = jnp.sqrt(1 + 1/dim) * jnp.eye(dim) - ((jnp.sqrt(dim + 1) + 1) / jnp.sqrt(dim ** 3)) * jnp.ones((dim, dim))
    points = jnp.concatenate([points, (1 / jnp.sqrt(dim)) * jnp.ones((1, dim))], axis=0)
    points = points * radius + origin
    return points


@jit
def get_sampled_polytope_vertices(origin: jax.Array,
                                  probes: jax.Array,
                                  polytope_vertices: jax.Array, 
                                  step_radius: float = 1., 
                                  probe_radius: float = 2.,
                                  rng: jax.Array = None,
                                  **kwargs) -> Tuple[jax.Array]:
    if origin.ndim == 1:
        origin = origin[jnp.newaxis, ...]
    batch, dim = origin.shape
    polytope_vertices = polytope_vertices[jnp.newaxis, ...].repeat(batch, axis=0)  # [batch, num_vertices, dim]

    # batch split key
    rng = default_prng_key(rng)
    batch_rng = random.split(rng, batch)
    uniform_rot_mat = vmap(get_random_uniform_rot_matrix, in_axes=(None, 0))(dim, batch_rng)  # [batch, dim, dim]
    polytope_vertices = polytope_vertices @ uniform_rot_mat
    step_points = polytope_vertices * step_radius + origin[:, jnp.newaxis, ...]  # [batch, num_vertices, dim]
    probe_points = get_probe_points(origin, polytope_vertices, probes, probe_radius)  # [batch, num_vertices, num_probe, dim]
    return step_points, probe_points, polytope_vertices


def get_sampled_points_on_sphere(origin: jax.Array,
                                 step_radius: float = 1., 
                                 probe_radius: float = 2., 
                                 num_probe: int = 5, 
                                 num_sphere_point: int = 50,
                                 rng: jax.Array = None,
                                 **kwargs) -> Tuple[jax.Array]:
    if origin.ndim == 1:
        origin = origin[jnp.newaxis, :]
    batch, dim = origin.shape
    # marsaglia method
    rng = default_prng_key(rng)
    points = random.normal(rng, shape=(batch, num_sphere_point, dim), dtype=origin.dtype)  # [batch, num_points, dim]
    points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)
    step_points = points * step_radius + origin[:, jnp.newaxis, :] # [batch, num_points, dim]
    probe_points = get_probe_points(origin, points, probe_radius, num_probe)  # [batch, 2 * dim, num_probe, dim]
    return step_points, probe_points, points


POLYTOPE_MAP = {
    'cube': get_cube_vertices,
    'orthoplex': get_orthoplex_vertices,
    'simplex': get_simplex_vertices,
}

POLYTOPE_NUM_VERTICES_MAP = {
    'cube': lambda dim: 2 ** dim,
    'orthoplex': lambda dim: 2 * dim,
    'simplex': lambda dim: dim + 1,
}

SAMPLE_POLYTOPE_MAP = {
    'polytope': get_sampled_polytope_vertices,
    'random': get_sampled_points_on_sphere,
}
