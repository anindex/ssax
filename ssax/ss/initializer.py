import abc
from typing import Any, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
from jax import random
from flax import struct
from ott.tools.gaussian_mixture.gaussian import Gaussian

from .utils import default_prng_key

__all__ = ["GaussianInitializer", "UniformInitializer"]


@struct.dataclass
class Initializer(abc.ABC):
    """Base abstract class for Sinkhorn Step initializers."""
    
    rng: jax.Array = struct.field(default=None, pytree_node=True)

    @abc.abstractmethod
    def init_points(self, num_points: int, rng: Optional[jax.Array] = None) -> jax.Array:
        """Initialize points for Sinkhorn Step.

        Args:
            num_points: Number of points to sample.
            rng: Optional random key. If None, uses the stored `self.rng`.
                 Note: Without passing a new key, the output is deterministic.

        Returns:
            Array of shape [num_points, dim].
        """
        pass

    def __call__(self, num_points: int, rng: Optional[jax.Array] = None) -> jax.Array:
        """Convenience wrapper for init_points."""
        return self.init_points(num_points, rng)


@struct.dataclass
class GaussianInitializer(Initializer):
    """Multivariate Gaussian initializer using OTT's Gaussian tool."""

    dist: Gaussian = struct.field(default=None, pytree_node=True)

    @classmethod
    def create(cls, 
               mean: Sequence[float], 
               cov: Union[Sequence[float], float, jax.Array], 
               rng: Optional[jax.Array] = None,
               **kwargs: Any) -> "GaussianInitializer":
        """Factory method to create a Gaussian initializer.

        Args:
            mean: Mean vector of shape [dim].
            cov: Covariance. Can be a scalar (isotropic), vector (diagonal), 
                 or matrix (full covariance).
            rng: Random seed.
        """
        mean = jnp.array(mean)
        dim = mean.shape[0]
        
        # Handle covariance broadcasting
        if isinstance(cov, (float, int)):
            # Isotropic: scalar * I
            cov_matrix = cov * jnp.eye(dim)
        else:
            cov = jnp.array(cov)
            if cov.ndim == 1:
                # Diagonal covariance provided as vector
                if cov.shape[0] != dim:
                    raise ValueError(f"Covariance vector dim {cov.shape[0]} != mean dim {dim}")
                cov_matrix = jnp.diag(cov)
            elif cov.ndim == 2:
                # Full covariance matrix
                if cov.shape != (dim, dim):
                    raise ValueError(f"Covariance shape {cov.shape} != ({dim}, {dim})")
                cov_matrix = cov
            else:
                raise ValueError("Covariance must be scalar, 1D (diag), or 2D (full).")

        rng = default_prng_key(rng)
        dist = Gaussian.from_mean_and_cov(mean, cov_matrix)
        
        return cls(rng=rng, dist=dist, **kwargs)

    def init_points(self, 
                    num_points: int,
                    rng: Optional[jax.Array] = None) -> jax.Array:
        """Sample points from the Gaussian distribution."""
        # Use passed rng, otherwise fallback to stored rng
        rng = rng if rng is not None else self.rng
        if rng is None:
            raise ValueError("No RNG key provided either in init or call.")
            
        return self.dist.sample(rng, num_points)


@struct.dataclass
class UniformInitializer(Initializer):
    """Uniform distribution initializer within hyper-rectangle bounds."""

    dim: int = struct.field(default=2, pytree_node=False)
    bounds: jax.Array = struct.field(default=None, pytree_node=True)

    @classmethod
    def create(cls, 
               bounds: List[Tuple[float, float]], 
               rng: Optional[jax.Array] = None,
               **kwargs: Any) -> "UniformInitializer":
        """Factory method for Uniform initializer.

        Args:
            bounds: List of (min, max) tuples for each dimension. 
                    Shape [dim, 2].
            rng: Random seed.
        """
        bounds_arr = jnp.array(bounds)
        if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
            raise ValueError(f"Bounds must be shape [dim, 2], got {bounds_arr.shape}")
            
        dim = bounds_arr.shape[0]
        rng = default_prng_key(rng)
        
        return cls(rng=rng, dim=dim, bounds=bounds_arr, **kwargs)

    def init_points(self, 
                    num_points: int,
                    rng: Optional[jax.Array] = None) -> jax.Array:
        """Sample points uniformly within bounds."""
        rng = rng if rng is not None else self.rng
        if rng is None:
            raise ValueError("No RNG key provided either in init or call.")

        # bounds shape: [dim, 2]
        minval = self.bounds[:, 0]  # [dim]
        maxval = self.bounds[:, 1]  # [dim]
        
        # JAX handles broadcasting: [num_points, dim]
        return random.uniform(rng, shape=(num_points, self.dim), 
                              minval=minval, maxval=maxval)
