from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from ott.geometry import geometry

__all__ = ["GenericCost"]


@jax.tree_util.register_pytree_node_class
class GenericCost(geometry.Geometry):
    """Generic cost function evaluated at polytope vertices for Sinkhorn Step.
    
    This geometry computes the cost matrix by evaluating an objective function
    on the provided vertices and averaging over the probe dimension.
    
    It integrates with the OTT library by instantiating a Geometry object
    defined by the computed [batch, num_vertices] cost matrix.
    """

    def __init__(
        self,
        objective_fn: Callable[[jax.Array], jax.Array],
        X: jax.Array,
        epsilon: Optional[Union[float, jax.Array]] = None,
        relative_epsilon: Optional[bool] = None,
        scale_cost: Optional[Union[float, str]] = None,
        **kwargs: Any
    ):
        """
        Args:
            objective_fn: A function that takes the vertex array `X` and returns 
                          costs. Expected output shape is compatible with averaging 
                          over the last axis (probes).
            X: The polytope vertices of shape [batch, num_vertices, num_probe, dim].
            epsilon: Regularization parameter for the Sinkhorn algorithm.
            relative_epsilon: If True, epsilon is relative to the mean cost.
            scale_cost: Strategy to scale the cost matrix (e.g. 'mean', 'max_cost').
            **kwargs: Additional arguments passed to ott.geometry.Geometry.
        """
        # Validate Input
        if X.ndim != 4:
            raise ValueError(f"Expected X to be 4D [batch, vert, probe, dim], got {X.shape}")

        self.objective_fn = objective_fn
        self.X = X

        raw_costs = objective_fn(X)
        cost_matrix = jnp.mean(raw_costs, axis=-1)

        super().__init__(
            cost_matrix=cost_matrix,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            scale_cost=scale_cost,
            **kwargs
        )

    def evaluate(self, X: jax.Array) -> jax.Array:
        """Evaluate the objective function at given points (convenience method)."""
        return self.objective_fn(X)

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Flatten the object for JAX transformations."""
        children = (
            self.objective_fn,
            self.X,
            self._epsilon_init,
        )
        
        aux_data = {
            "relative_epsilon": self._relative_epsilon,
            "scale_cost": self._scale_cost,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Tuple[Any, ...]) -> "GenericCost":
        """Reconstruct the object from flattened data."""
        objective_fn, X, epsilon = children
        return cls(
            objective_fn=objective_fn,
            X=X,
            epsilon=epsilon,
            **aux_data
        )