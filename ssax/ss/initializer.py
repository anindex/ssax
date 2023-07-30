import abc
from typing import Any, Dict, Optional, Sequence, Tuple, List, Union

import jax
import jax.numpy as jnp
from jax import random, jit

from ott.tools.gaussian_mixture.gaussian import Gaussian

from ssax.ss.utils import default_prng_key


__all__ = [
    "SSGaussianInitializer", "SSUniformInitializer"
]


@jax.tree_util.register_pytree_node_class
class SinkhornStepInitializer(abc.ABC):
    """Base class for Sinkhorn Step initializers."""

    def __init__(self, 
                 rng: random.PRNGKeyArray = None,
                 **kwargs: Any) -> None:
        self.rng = default_prng_key(rng)

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

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  
        return [], {}

    @classmethod
    def tree_unflatten(  
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "SinkhornStepInitializer":
        return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class SSGaussianInitializer(SinkhornStepInitializer):
    """Gaussian Sinkhorn Step initializer."""

    def __init__(self, 
                 mean: List[float], 
                 var: Union[List[float], float], 
                 rng: random.PRNGKeyArray = None,
                 **kwargs: Any) -> None:
        self.mean = jnp.array(mean)
        if isinstance(var, float):
            self.var = var * jnp.eye(self.mean.shape[0])
        else:
            self.var = jnp.array(var)
        self.dist = Gaussian.from_mean_and_cov(mean, var)
        super().__init__(rng, **kwargs)

    def init_points(self, 
                    num_points: int,
                    rng: Optional[random.PRNGKeyArray] = None) -> jnp.array:
        if rng is None:
            rng = self.rng
        return self.dist.sample(rng, num_points)

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  
        return [
            self.mean,
            self.var
        ], {}

    @classmethod
    def tree_unflatten(  
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "SinkhornStepInitializer":
        mean, var = children
        return cls(mean, var, **aux_data)


@jax.tree_util.register_pytree_node_class
class SSUniformInitializer(SinkhornStepInitializer):
    """Uniform Sinkhorn Step initializer."""

    def __init__(self, 
                 bounds: List[Tuple[float, float]],  # [d, 2]
                 rng: random.PRNGKeyArray = None,
                 **kwargs: Any) -> None:
        super().__init__(rng, **kwargs)
        self.bounds = jnp.array(bounds)
        self.dim = self.bounds.shape[0]

    def init_points(self, 
                    num_points: int,
                    rng: Optional[random.PRNGKeyArray] = None) -> jnp.array:
        if rng is None:
            rng = self.rng
        samples = random.uniform(rng, shape=(num_points, self.dim), minval=self.bounds[:, 0], maxval=self.bounds[:, 1])
        return samples

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  
        return [
            self.bounds
        ], {}

    @classmethod
    def tree_unflatten(  
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "SinkhornStepInitializer":
        bounds = children
        return cls(bounds, **aux_data)
