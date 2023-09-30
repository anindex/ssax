import jax
from jax import jit, random
import jax.numpy as jnp

from itertools import product
from functools import partial
from typing import Tuple

from .probe import get_random_probe_points, get_probe_points
from .rotation import get_random_maximal_torus_matrix
from .utils import default_prng_key


@jit
def get_cube_vertices(origin: jnp.array, 
                      radius: float = 1., 
                      **kwargs) -> jnp.array:
    dim = origin.shape[-1]
    points = jnp.array(list(product([1, -1], repeat=dim)), dtype=origin.dtype) / jnp.sqrt(dim)
    points = points * radius + origin
    return points


@jit
def get_orthoplex_vertices(origin: jnp.array, 
                           radius: float = 1., 
                           **kwargs) -> jnp.array:
    dim = origin.shape[-1]
    points = jnp.zeros((2 * dim, dim), dtype=origin.dtype)
    first = jnp.arange(0, dim)
    second = jnp.arange(dim, 2 * dim)
    points = points.at[first, first].set(radius)
    points = points.at[second, first].set(-radius)
    points = points + origin
    return points


@jit
def get_simplex_vertices(origin: jnp.array, radius: float = 1., **kwargs) -> jnp.array:
    '''
    Simplex coordinates: https://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_a_regular_n-dimensional_simplex_in_Rn
    '''
    dim = origin.shape[-1]
    points = jnp.sqrt(1 + 1/dim) * jnp.eye(dim) - ((jnp.sqrt(dim + 1) + 1) / jnp.sqrt(dim ** 3)) * jnp.ones((dim, dim))
    points = jnp.concatenate([points, (1 / jnp.sqrt(dim)) * jnp.ones((1, dim))], axis=0)
    points = points * radius + origin
    return points


# @jit
def get_sampled_polytope_vertices(origin: jnp.array,
                                  probes: jnp.array,
                                  polytope_vertices: jnp.array, 
                                  step_radius: float = 1., 
                                  probe_radius: float = 2.,
                                  rng: jax.random.PRNGKey = None,
                                  **kwargs) -> Tuple[jnp.array]:
    if origin.ndim == 1:
        origin = origin[jnp.newaxis, ...]
    batch, dim = origin.shape
    polytope_vertices = polytope_vertices[jnp.newaxis, ...].repeat(batch, axis=0)  # [batch, num_vertices, dim]

    max_torus_mat = get_random_maximal_torus_matrix(origin, rng=rng)
    polytope_vertices = polytope_vertices @ max_torus_mat
    step_points = polytope_vertices * step_radius + origin[:, jnp.newaxis, ...]  # [batch, num_vertices, dim]
    probe_points = get_probe_points(origin, polytope_vertices, probes, probe_radius)  # [batch, num_vertices, num_probe, dim]
    return step_points, probe_points, polytope_vertices


def get_sampled_points_on_sphere(origin: jnp.array,
                                 step_radius: float = 1., 
                                 probe_radius: float = 2., 
                                 num_probe: int = 5, 
                                 num_sphere_point: int = 50,
                                 rng: jax.random.PRNGKey = None,
                                 **kwargs) -> Tuple[jnp.array]:
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
