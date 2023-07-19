import abc
import functools
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

from ott.math import fixed_point_loop, matrix_square_root
from ott.math import utils as mu



@jax.tree_util.register_pytree_node_class
class ObjectiveFn(abc.ABC):
  """Base class for all costs.
  """

  @abc.abstractmethod
  def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.

    Returns:
      The cost array.
    """

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.
      y: Array.

    Returns:
      The cost, optionally including the :attr:`norms <norm>` of
      :math:`x`/:math:`y`.
    """
    cost = self.evaluate(x)
    return cost

  def tree_flatten(self):  # noqa: D102
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)