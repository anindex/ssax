import abc
from typing import Any, Dict, Optional, Sequence, Tuple, List, Union

import jax
import jax.numpy as jnp
from jax import random, jit
from flax import struct
from ott.tools.gaussian_mixture.gaussian import Gaussian

from ssax.ss.utils import default_prng_key


__all__ = [
    "GaussianInitializer", "UniformInitializer"
]


@struct.dataclass
class Initializer(abc.ABC):
    """Base class for Sinkhorn Step initializers."""
    rng: random.PRNGKeyArray = None

    @abc.abstractmethod
    def init_points(self, num_points: int) -> jnp.array:
        """Initialize points for Sinkhorn Step.

        Args:
          rng: Random number generator for stochastic initializers.

        Returns:
          Points for Sinkhorn Step.
        """
        pass

    def __call__(self, num_points: int) -> jnp.array:
        return self.init_points(num_points)


@struct.dataclass
class GaussianInitializer(Initializer):
    """Gaussian Sinkhorn Step initializer."""

    dist: Gaussian = None

    @classmethod
    def create(cls, 
               mean: List[float], 
               var: Union[List[float], float], 
               rng: random.PRNGKeyArray = None,
               **kwargs: Any) -> "GaussianInitializer":
        mean = jnp.array(mean)
        if isinstance(var, float):
            var = var * jnp.eye(mean.shape[0])
        else:
            var = jnp.array(var)
        rng = default_prng_key(rng)
        dist = Gaussian.from_mean_and_cov(mean, var)
        return cls(rng=rng, dist=dist, **kwargs)

    def init_points(self, 
                    num_points: int,
                    rng: Optional[random.PRNGKeyArray] = None) -> jnp.array:
        if rng is None:
            rng = self.rng
        return self.dist.sample(rng, num_points)


@struct.dataclass
class UniformInitializer(Initializer):
    """Uniform Sinkhorn Step initializer."""

    dim: int = None
    bounds: jnp.array = None

    @classmethod
    def create(cls, 
               bounds: List[Tuple[float, float]],  # [d, 2]
               rng: random.PRNGKeyArray = None,
               **kwargs: Any) -> "UniformInitializer":
        bounds = jnp.array(bounds)
        dim = bounds.shape[0]
        rng = default_prng_key(rng)
        return cls(rng=rng, dim=dim, bounds=bounds, **kwargs)

    def init_points(self, 
                    num_points: int,
                    rng: Optional[random.PRNGKeyArray] = None) -> jnp.array:
        if rng is None:
            rng = self.rng
        samples = random.uniform(rng, shape=(num_points, self.dim), minval=self.bounds[:, 0], maxval=self.bounds[:, 1])
        return samples
