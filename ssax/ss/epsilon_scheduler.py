from typing import Any, Optional, Tuple, Dict

import jax
import jax.numpy as jnp

from ott.geometry.epsilon_scheduler import Epsilon

__all__ = ["LinearEpsilon"]


@jax.tree_util.register_pytree_node_class
class LinearEpsilon(Epsilon):
    """Linearly decaying epsilon scheduler.
    
    Computes epsilon at iteration `t` as:
    eps_t = max(init - (decay * t), target)

    Args:
        target: The floor value for epsilon.
        init: The starting value at iteration 0.
        decay: The amount to subtract per iteration.
    """

    def __init__(self, 
                 target: float = 1e-3, 
                 init: float = 1.0, 
                 decay: float = 0.01,
                 **kwargs: Any):
        target = target if target is not None else 1e-3
        super().__init__(target=target, init=init, decay=decay, **kwargs)

    def at(self, iteration: Optional[int] = 1) -> float:
        """Calculate epsilon at the given iteration."""
        if iteration is None:
            return self.target

        # Linear decay with a floor at self.target
        eps = self.init - (self.decay * iteration)
        return jnp.maximum(eps, self.target)

    def tree_flatten(self) -> Tuple[Tuple[()], Dict[str, Any]]:
        """Flatten the object for JAX transformations.
        """
        children = () 
        aux_data = {
            "target": self.target,
            "init": self.init,
            "decay": self.decay
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Tuple[()]) -> "LinearEpsilon":
        """Reconstruct the object from flattened data."""
        return cls(**aux_data)