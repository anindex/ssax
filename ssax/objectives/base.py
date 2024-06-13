import abc
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class ObjectiveFn(abc.ABC):
    """Base class for all costs."""

    dim: int = struct.field(default=2, pytree_node=False)
    optimal_value: float = None
    optimizers: Optional[jax.Array] = None
    noise_std: Optional[float] = None
    negate: bool = False
    bounds: jax.Array = struct.field(default=None, pytree_node=False)

    @abc.abstractmethod
    def evaluate(self, X: jax.Array) -> jax.Array:
        """Compute cost

        Args:
          X: array.

        Returns:
          The cost array.
        """

    def __call__(self, X: jax.Array) -> jax.Array:
        cost = self.evaluate(X)
        return cost
