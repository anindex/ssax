import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from typing import Optional

from .utils import default_prng_key


@partial(jit, static_argnames=['num_probe'])
def get_random_probe_points(origin: jax.Array, 
                            points: jax.Array, 
                            probe_radius: float = 2.0, 
                            num_probe: int = 5, 
                            rng: Optional[jax.Array] = None) -> jax.Array:
    """
    Generates probe points randomly sampled along the line segments 
    connecting 'origin' to 'origin + points * probe_radius'.

    Args:
        origin: Center [batch, dim]
        points: Direction vectors [batch, num_points, dim]
        probe_radius: Maximum distance from origin.
        num_probe: Number of random samples per direction.
        rng: Random key.

    Returns:
        Array of shape [batch, num_points, num_probe, dim]
    """
    rng = default_prng_key(rng)
    batch, num_points, dim = points.shape
    
    # Generate random scaling factors in [0, 1]
    # Shape: [batch, num_points, num_probe, 1]
    alpha = random.uniform(rng, shape=(batch, num_points, num_probe, 1))
    
    # Prepare shapes for broadcasting:
    # origin: [batch, 1, 1, dim]
    origin_exp = origin[:, None, None, :]
    # points: [batch, num_points, 1, dim]
    points_exp = points[:, :, None, :]
    
    # P = Origin + (Vector * Radius) * RandomScale
    probe_points = origin_exp + (points_exp * probe_radius) * alpha
    return probe_points


@jit
def get_probe_points(origin: jax.Array, 
                     points: jax.Array,
                     scales: jax.Array,
                     probe_radius: float = 2.0) -> jax.Array:
    """
    Generates probe points at fixed scale intervals along the line segments.

    Args:
        origin: Center [batch, dim]
        points: Direction vectors [batch, num_points, dim]
        scales: Array of scalar coefficients (e.g., [0.5, 1.0]) of shape [num_probe].
        probe_radius: Maximum distance.

    Returns:
        Array of shape [batch, num_points, num_probe, dim]
    """
    # origin: [batch, 1, 1, dim]
    origin_exp = origin[:, None, None, :]
    
    # points: [batch, num_points, 1, dim]
    points_exp = points[:, :, None, :]
    
    # scales: [1, 1, num_probe, 1]
    scales_exp = scales[None, None, :, None]
    
    # P = Origin + (Vector * Radius) * FixedScale
    probe_points = origin_exp + (points_exp * probe_radius) * scales_exp
    return probe_points


@jit
def get_shifted_points(new_origins: jax.Array, points: jax.Array) -> jax.Array:
    """
    Shifts a generic point cloud by multiple new origins (broadcasting).

    Args:
        new_origins: [num_origins, dim]
        points: [num_points, dim] (usually centered at 0)
        
    Returns:
        [num_origins, num_points, dim]
    """
    # [num_origins, 1, dim] + [1, num_points, dim]
    return new_origins[:, None, :] + points[None, :, :]


@partial(jit, static_argnames='num_probe')
def get_projecting_points(X1: jax.Array, 
                          X2: jax.Array, 
                          probe_step_size: float, 
                          num_probe: int = 5) -> jax.Array:
    """
    Generates points projecting from X1 towards X2.
    
    Can handle two cases:
    1. All-pairs: X1 [N, D] vs X2 [M, D] -> [N, M, P, D]
    2. Batched-pairs: X1 [N, D] vs X2 [N, M, D] -> [N, M, P, D]

    Args:
        X1: Source points [nb1, dim]
        X2: Target points [nb2, dim] or [nb1, nb2, dim]
        probe_step_size: Step size multiplier.
        num_probe: Number of steps.

    Returns:
        [nb1, nb2, num_probe, dim]
    """
    # X1: [nb1, 1, 1, dim]
    X1_exp = X1[:, None, None, :]
    
    # Handle X2 shapes
    if X2.ndim == 2:
        # Case: All-pairs
        # X2: [nb2, dim] -> [1, nb2, 1, dim]
        X2_exp = X2[None, :, None, :]
    else:
        # Case: Batched specific targets
        # X2: [nb1, nb2, dim] -> [nb1, nb2, 1, dim]
        X2_exp = X2[:, :, None, :]
    
    # Generate scales: [1, 1, num_probe, 1]
    # creates [step, 2*step, ..., num_probe*step]
    scales = (jnp.arange(1, num_probe + 1, dtype=X1.dtype) * probe_step_size)
    scales = scales[None, None, :, None]
    
    # Vector P = A + (B - A) * t
    points = X1_exp + (X2_exp - X1_exp) * scales
    return points