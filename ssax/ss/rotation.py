import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, random
from functools import partial
from typing import Optional, Tuple

from .utils import default_prng_key


@jit
def get_rotation_matrix_2d(theta: jax.Array) -> jax.Array:
    """
    Creates 2x2 rotation matrices from angles.
    
    Args:
        theta: Angles array of shape [...]
        
    Returns:
        Array of shape [..., 2, 2]
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    # Stack to create [[cos, -sin], [sin, cos]]
    # shape: [..., 2, 2]
    row1 = jnp.stack([c, -s], axis=-1)
    row2 = jnp.stack([s, c], axis=-1)
    return jnp.stack([row1, row2], axis=-2)


@partial(jit, static_argnames=['dim'])
def get_random_uniform_rot_matrix(dim: int,
                                  rng: Optional[jax.Array] = None,
                                  **kwargs) -> jax.Array:
    """
    Generates a uniformly random rotation matrix on SO(dim) using the QR method.
    
    Ref: F. Mezzadri, "How to generate random matrices from the classical compact groups"
    (arXiv:math-ph/0609050).
    
    Args:
        dim: Dimension of the matrix (N).
        rng: JAX PRNG key.
        
    Returns:
        Rotation matrix of shape [dim, dim] with det(R) = 1.
    """
    rng = default_prng_key(rng)
    
    # 1. Sample from standard normal distribution
    Z = random.normal(rng, (dim, dim))
    
    # 2. QR Decomposition
    Q, R = jnp.linalg.qr(Z)
    
    # 3. Ensure uniqueness by fixing the sign of the diagonal of R
    # The diagonal of R must be positive for the decomposition to be unique 
    # and map to the Haar measure correctly.
    d = jnp.diagonal(R)
    phases = jnp.sign(d)
    
    # Apply phases to Q
    Q = Q * phases
    
    # 4. Ensure determinant is +1 (Restriction to SO(n))
    # Currently Q is in O(n), meaning det is +1 or -1.
    det = jnp.linalg.det(Q)
    
    # If det is negative, flip the sign of the first column
    # (flipping one column changes determinant sign)
    det_sign = jnp.sign(det)
    Q = Q.at[:, 0].multiply(det_sign)
    
    return Q


@partial(jit, static_argnames=['angle_range'])
def get_random_maximal_torus_matrix(origin: jax.Array, 
                                    angle_range: Tuple[float, float] = (0, 2 * jnp.pi),
                                    rng: Optional[jax.Array] = None,
                                    **kwargs) -> jax.Array:
    """
    Generates a block-diagonal rotation matrix (maximal torus).
    Only supports even dimensions.
    
    Args:
        origin: Used only for shape inference [batch, dim].
        angle_range: Min and max angles for rotation.
        
    Returns:
        Block diagonal matrices [batch, dim, dim]
    """
    batch, dim = origin.shape
    
    if dim % 2 != 0:
        raise ValueError(f'Maximal torus rotation requires even dimension, got {dim}.')

    rng = default_prng_key(rng)
    
    # 1. Sample angles: [batch, dim // 2]
    # One angle per 2x2 block
    n_blocks = dim // 2
    theta = random.uniform(rng, shape=(batch, n_blocks), 
                           minval=angle_range[0], maxval=angle_range[1], 
                           dtype=origin.dtype)
    
    # 2. Convert angles to 2x2 matrices: [batch, n_blocks, 2, 2]
    blocks = get_rotation_matrix_2d(theta)
    
    # 3. Form block diagonal matrix [batch, dim, dim]
    
    def _assemble_blocks(block_list):
        # block_list is [n_blocks, 2, 2]
        return jsp.linalg.block_diag(*block_list)

    max_torus_mat = vmap(_assemble_blocks)(blocks)
    
    return max_torus_mat