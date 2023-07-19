import math
from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott.geometry import geometry

__all__ = ["GenericCost"]


@jax.tree_util.register_pytree_node_class
class GenericCost(geometry.Geometry):
    """Generic cost function evaluated at polytope vertices for Sinkhorn Step."""

    def __init__(
        self,
        X: jnp.ndarray,
        objective_fn: Any,
        **kwargs: Any
        ):
        super().__init__(**kwargs)
        self._X = X  # polytope vertices [batch, num_vertices, d]
        self.objective_fn = objective_fn

    @property.setter
    def X(self, new_x: jnp.ndarray):  # noqa: D102
        assert new_x.ndim == 3
        self._X = new_x
        self._compute_cost_matrix()

    @property
    def X(self) -> jnp.ndarray:  # noqa: D102
        return self._X

    @property
    def cost_matrix(self) -> Optional[jnp.ndarray]:  # noqa: D102
        if self._cost_matrix is None:
            self._compute_cost_matrix()
        return self._cost_matrix * self.inv_scale_cost

    @property
    def kernel_matrix(self) -> Optional[jnp.ndarray]:  # noqa: D102
        return jnp.exp(-self.cost_matrix / self.epsilon)

    @property
    def shape(self) -> Tuple[int, int, int]:
        # in the process of flattening/unflattening in vmap, `__init__`
        # can be called with dummy objects
        # we optionally access `shape` in order to get the batch size
        if self._X is None:
            return 0
        return self._X.shape

    @property
    def is_symmetric(self) -> bool:  # noqa: D102
        return self._X.shape[0] == self._X.shape[1]

    def _compute_cost_matrix(self) -> jnp.ndarray:
        self._cost_matrix = self.objective_fn(self._X)

    def barycenter(self, weights: jnp.ndarray) -> jnp.ndarray:
        """Compute barycenter of points in self._X using weights."""
        return jnp.average(self._X, weights=weights, axis=1)  # [batch, d]

    def tree_flatten(self):  # noqa: D102
        return (
            self._X,
            self._src_mask,
            self._tgt_mask,
            self._epsilon_init,
            self.objective_fn,
        ), {
            "scale_cost": self._scale_cost
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        X, src_mask, tgt_mask, epsilon, objective_fn = children
        return cls(
            X,
            objective_fn=objective_fn,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            epsilon=epsilon,
            **aux_data
        )

