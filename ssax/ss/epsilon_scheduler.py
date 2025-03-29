from typing import Any, Optional

import jax
import jax.numpy as jnp

from ott.geometry.epsilon_scheduler import Epsilon


__all__ = ["LinearEpsilon"]


@jax.tree_util.register_pytree_node_class
class LinearEpsilon(Epsilon):

    def __init__(self, 
                 target: float = None, 
                 init: float = 1, 
                 decay: float = 1,
                 **kwargs: Any):
        super().__init__(target, init, decay)

    def at(self, iteration: Optional[int] = 1) -> float:
        if iteration is None:
            return self.target

        eps = jnp.maximum(self.init - (self.decay * iteration), self.target)
        return eps
