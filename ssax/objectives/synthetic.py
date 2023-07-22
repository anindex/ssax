import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from typing import Any, Callable, Dict, Optional, Tuple, Union, List

from .base import ObjectiveFn


@jax.tree_util.register_pytree_node_class
class Ackley(ObjectiveFn):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = jnp.array([(-32.768, 32.768) for _ in range(dim)])
        self._optimizers = jnp.zeros(self.dim)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.a = 20
        self.b = 0.2
        self.c = 2 * jnp.pi

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        a, b, c = self.a, self.b, self.c
        part1 = -a * jnp.exp(-b / jnp.sqrt(self.dim) * jnp.linalg.norm(X, axis=-1))
        part2 = -(jnp.exp(jnp.mean(jnp.cos(c * X), axis=-1)))
        return part1 + part2 + a + jnp.e
    
    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self._bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class Beale(ObjectiveFn):

    _optimal_value = 0.0
    _optimizers = jnp.array([(3.0, 0.5)])

    def __init__(self, 
                 noise_std: float | None = None, 
                 negate: bool = False, 
                 bounds: List[Tuple[float, float]] | None = None, 
                 **kwargs: Any):
        self.dim = 2
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2**2) ** 2
        part3 = (2.625 - x1 + x1 * x2**3) ** 2
        return part1 + part2 + part3
