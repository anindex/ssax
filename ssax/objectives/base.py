import abc
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
class ObjectiveFn(abc.ABC):
    """Base class for all costs."""

    dim: int
    _optimal_value: float
    _optimizers: Optional[jnp.ndarray] = None
    _bounds: jnp.ndarray

    def __init__(self,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 bounds: Optional[List[Tuple[float, float]]] = None, 
                 **kwargs: Any):
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.noise_std = noise_std
        self.negate = negate
        self._bounds = bounds
        if self._bounds is not None and self._bounds.shape[0] != self.dim:
            raise ValueError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {self._bounds.shape=}."
            )

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self._optimal_value if self.negate else self._optimal_value

    @abc.abstractmethod
    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        """Compute cost

        Args:
          X: Array.

        Returns:
          The cost array.
        """

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        cost = self.evaluate(X)
        return cost

    def tree_flatten(self):  # noqa: D102
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        del aux_data
        return cls(*children)
