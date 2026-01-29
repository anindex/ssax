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
        chunk_size: Optional[int] = None,
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
            chunk_size: If set, evaluate objective_fn in chunks of this size via
                        jax.lax.map to bound peak memory. None means evaluate all
                        at once (original behavior).
            **kwargs: Additional arguments passed to ott.geometry.Geometry.
        """
        # Validate Input
        if X.ndim != 4:
            raise ValueError(f"Expected X to be 4D [batch, vert, probe, dim], got {X.shape}")

        self.objective_fn = objective_fn
        self._x_shape = X.shape
        self._chunk_size = chunk_size

        # Vectorize evaluation by flattening to [batch*vert*probe, dim]
        batch, num_verts, num_probe, dim = X.shape
        flat_X = X.reshape(-1, dim)

        if chunk_size is not None and chunk_size > 0:
            # Chunked evaluation: process chunk_size samples at a time
            raw_costs_flat = _chunked_eval(objective_fn, flat_X, chunk_size)
        else:
            raw_costs_flat = objective_fn(flat_X)

        raw_costs = raw_costs_flat.reshape(batch, num_verts, num_probe)
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
            self._epsilon_init,
        )

        aux_data = {
            "relative_epsilon": self._relative_epsilon,
            "scale_cost": self._scale_cost,
            "x_shape": self._x_shape,
            "chunk_size": self._chunk_size,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Tuple[Any, ...]) -> "GenericCost":
        """Reconstruct the object from flattened data."""
        objective_fn, epsilon = children
        x_shape = aux_data.pop("x_shape")
        chunk_size = aux_data.pop("chunk_size", None)

        dummy_X = jnp.zeros(x_shape)
        instance = cls(
            objective_fn=objective_fn,
            X=dummy_X,
            epsilon=epsilon,
            chunk_size=chunk_size,
            **aux_data
        )
        return instance


def _chunked_eval(fn: Callable, X: jax.Array, chunk_size: int) -> jax.Array:
    """Evaluate fn on X in chunks to limit peak memory.

    Pads X to be divisible by chunk_size, evaluates via lax.map,
    then trims back to the original size.

    Args:
        fn: Function mapping (chunk_size, dim) -> (chunk_size,) or scalar per row.
        X: Array of shape (N, dim).
        chunk_size: Number of rows to evaluate at once.

    Returns:
        Array of shape (N,) with fn evaluated on each row.
    """
    N = X.shape[0]
    # Pad to make divisible by chunk_size
    remainder = N % chunk_size
    if remainder != 0:
        pad_size = chunk_size - remainder
        X = jnp.pad(X, ((0, pad_size), (0, 0)))
    else:
        pad_size = 0

    N_padded = X.shape[0]
    num_chunks = N_padded // chunk_size
    X_chunks = X.reshape(num_chunks, chunk_size, -1)

    # Use lax.map for sequential, memory-efficient evaluation
    results = jax.lax.map(fn, X_chunks)
    results = results.reshape(-1)

    if pad_size > 0:
        results = results[:N]
    return results
