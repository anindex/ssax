from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import geometry

__all__ = ["GenericCost"]


@jax.tree_util.register_pytree_node_class
class GenericCost(geometry.Geometry):
    """Generic cost function evaluated at polytope vertices for Sinkhorn Step."""

    def __init__(
        self,
        objective_fn: Any,
        X: jax.Array,
        **kwargs: Any
        ):
        super().__init__(**kwargs)

        self.objective_fn = objective_fn
        
        assert X.ndim == 4  # polytope vertices [batch, num_vertices, num_probe, d]
        self.X = X

    @property
    def cost_matrix(self) -> Optional[jax.Array]:  
        if self._cost_matrix is None:
            self._compute_cost_matrix()
        return self._cost_matrix * self.inv_scale_cost

    @property
    def kernel_matrix(self) -> Optional[jax.Array]:  
        return jnp.exp(-self.cost_matrix / self.epsilon)

    @property
    def shape(self) -> Tuple[int, int]:
        # in the process of flattening/unflattening in vmap, `__init__`
        # can be called with dummy objects
        # we optionally access `shape` in order to get the batch size
        return self.X.shape[:2]

    @property
    def is_symmetric(self) -> bool:  
        return self.X.shape[0] == self.X.shape[1]

    def _compute_cost_matrix(self) -> jax.Array:
        costs = self.objective_fn(self.X)
        self._cost_matrix = costs.mean(axis=-1)  # [batch, num_vertices]

    def evaluate(self, X: jax.Array) -> jax.Array:
        """Evaluate cost function at given points."""
        return self.objective_fn(X)

    def tree_flatten(self):  
        return (
            self.objective_fn,
            self.X,
            self._src_mask,
            self._tgt_mask,
            self._epsilon_init,
        ), {
            "scale_cost": self._scale_cost,
            "relative_epsilon": self._relative_epsilon
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        objective_fn, X, src_mask, tgt_mask, epsilon = children
        return cls(
            objective_fn=objective_fn,
            X=X,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            epsilon=epsilon,
            **aux_data
        )
