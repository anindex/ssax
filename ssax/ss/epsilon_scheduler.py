from typing import Any, Optional

import jax
import jax.numpy as jnp

from ott.geometry.epsilon_scheduler import Epsilon


__all__ = ["LinearEpsilon"]


@jax.tree_util.register_pytree_node_class
class LinearEpsilon(Epsilon):

    def __init__(self, target: float | None = None, scale_epsilon: float | None = None, init: float = 1, decay: float = 1):
        super().__init__(target, scale_epsilon, init, decay)

    def at(self, iteration: Optional[int] = 1) -> float:
        if iteration is None:
            return self.target
        
        eps = jnp.minimum(self._init - (self._decay * iteration), self.target)
        return eps
