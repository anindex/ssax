import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from functools import partial
from typing import Tuple, Dict, Callable, Optional, Union

from .probe import get_probe_points, get_random_probe_points
from .rotation import get_random_uniform_rot_matrix
from .utils import default_prng_key

@jit
def get_cube_vertices(origin: jax.Array, 
                      radius: float = 1.0, 
                      **kwargs) -> jax.Array:
    """Generates hypercube vertices using bitwise logic."""
    dim = origin.shape[-1]
    n_vertices = 2 ** dim
    
    # 0 to 2^dim - 1
    indices = jnp.arange(n_vertices, dtype=jnp.uint32)[:, None]
    shifts = jnp.arange(dim, dtype=jnp.uint32)[None, :]
    
    # Bits to signs mapping (0 -> -1, 1 -> 1)
    bits = (indices >> shifts) & 1
    signs = 1.0 - 2.0 * bits.astype(origin.dtype)
    
    points = signs / jnp.sqrt(dim)
    return points * radius + origin


@jit
def get_orthoplex_vertices(origin: jax.Array, 
                           radius: float = 1.0, 
                           **kwargs) -> jax.Array:
    """Generates orthoplex (cross-polytope) vertices."""
    dim = origin.shape[-1]
    eye = jnp.eye(dim, dtype=origin.dtype)
    points = jnp.concatenate([eye, -eye], axis=0)
    return points * radius + origin


@jit
def get_simplex_vertices(origin: jax.Array, 
                         radius: float = 1.0, 
                         **kwargs) -> jax.Array:
    """Generates regular simplex vertices."""
    dim = origin.shape[-1]
    
    points = jnp.sqrt(1 + 1/dim) * jnp.eye(dim, dtype=origin.dtype)
    points = points - ((jnp.sqrt(dim + 1) + 1) / jnp.sqrt(dim ** 3))
    
    last_vertex = (1 / jnp.sqrt(dim)) * jnp.ones((1, dim), dtype=origin.dtype)
    points = jnp.concatenate([points, last_vertex], axis=0)
    
    return points * radius + origin


@jit
def get_sampled_polytope_vertices(origin: jax.Array,
                                  probes: jax.Array,
                                  polytope_vertices: jax.Array, 
                                  step_radius: float = 1.0, 
                                  probe_radius: float = 2.0,
                                  rng: Optional[jax.Array] = None,
                                  **kwargs) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Rotates a polytope and generates deterministic probes.
    
    Args:
        probes: Array of scalar scales (e.g. [0.5, 1.0]).
                Passed to deterministic get_probe_points.
    """
    origin = jnp.atleast_2d(origin)
    batch, dim = origin.shape
    
    rng = default_prng_key(rng)
    batch_keys = random.split(rng, batch)

    # 1. Generate Rotation Matrices [batch, dim, dim]
    rot_mats = vmap(get_random_uniform_rot_matrix, in_axes=(None, 0))(dim, batch_keys)
    
    # 2. Apply Rotation using Einsum
    # 'bji' = batch, row(j), col(i) of Rotation Matrix
    # 'ni'  = num_verts(n), col(i) of Vertices (treated as row vectors)
    # Result 'bnj' = batch, num_verts, dim
    # This is equivalent to v_new = R @ v_old
    rotated_vertices = jnp.einsum('bji, ni -> bnj', rot_mats, polytope_vertices)
    
    # 3. Translate Step Points
    step_points = rotated_vertices * step_radius + origin[:, None, :]
    
    # 4. Generate Probes (Deterministic scaling based on `probes` array)
    # rotated_vertices acts as the "direction" relative to origin
    probe_points = get_probe_points(origin, rotated_vertices, probes, probe_radius)
    
    return step_points, probe_points, rotated_vertices


@partial(jit, static_argnames=("num_probe", "num_sphere_point"))
def get_sampled_points_on_sphere(origin: jax.Array,
                                 step_radius: float = 1.0, 
                                 probe_radius: float = 2.0, 
                                 num_probe: int = 5, 
                                 num_sphere_point: int = 50,
                                 rng: Optional[jax.Array] = None,
                                 **kwargs) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Samples random points on a sphere and generates random probes.
    """
    origin = jnp.atleast_2d(origin)
    batch, dim = origin.shape
    
    rng = default_prng_key(rng)
    rng_sphere, rng_probe = random.split(rng)
    
    # 1. Sample Points on Sphere (Marsaglia)
    points = random.normal(rng_sphere, shape=(batch, num_sphere_point, dim), dtype=origin.dtype)
    points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)
    
    # 2. Translate Step Points
    step_points = points * step_radius + origin[:, None, :]
    
    # 3. Generate Probes (Random scaling using num_probe)
    # Note: We pass 'points' as directions.
    probe_points = get_random_probe_points(origin, points, probe_radius, num_probe, rng_probe)
    
    return step_points, probe_points, points


POLYTOPE_MAP: Dict[str, Callable] = {
    'cube': get_cube_vertices,
    'orthoplex': get_orthoplex_vertices,
    'simplex': get_simplex_vertices,
}

POLYTOPE_NUM_VERTICES_MAP: Dict[str, Callable[[int], int]] = {
    'cube': lambda dim: 2 ** dim,
    'orthoplex': lambda dim: 2 * dim,
    'simplex': lambda dim: dim + 1,
}

SAMPLE_POLYTOPE_MAP: Dict[str, Callable] = {
    'polytope': get_sampled_polytope_vertices,
    'random': get_sampled_points_on_sphere,
}