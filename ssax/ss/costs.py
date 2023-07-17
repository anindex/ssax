import math
from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import costs, geometry, low_rank
from ott.math import utils as mu

__all__ = ["GenericCost"]


@jax.tree_util.register_pytree_node_class
class GenericCost(geometry.Geometry):
    """Generic cost function evaluated at polytope vertices for Sinkhorn Step."""

    def __init__(
        self,
        x: jnp.ndarray,
        cost_fn: costs.CostFn,
        is_online: Optional[bool] = True,
        scale_cost: Union[bool, int, float,
                            Literal["mean", "max_norm", "max_bound", "max_cost",
                                    "median"]] = 1.0,
        **kwargs: Any
        ):
        super().__init__(**kwargs)
        self.x = x  # polytope vertices [b, n, d]

        self.cost_fn = cost_fn
        self._axis_norm = 0 if callable(self.cost_fn.norm) else None

        self.is_online = is_online
        self._scale_cost = "mean" if scale_cost is True else scale_cost

    @property
    def _norm_x(self) -> Union[float, jnp.ndarray]:
        if self._axis_norm == 0:
            return self.cost_fn.norm(self.x)
        return 0.

    @property
    def cost_matrix(self) -> Optional[jnp.ndarray]:  # noqa: D102
        if self.is_online:
            return None
        cost_matrix = self._compute_cost_matrix()
        return cost_matrix * self.inv_scale_cost

    @property
    def kernel_matrix(self) -> Optional[jnp.ndarray]:  # noqa: D102
        if self.is_online:
            return None
        return jnp.exp(-self.cost_matrix / self.epsilon)

    @property
    def shape(self) -> Tuple[int, int, int]:
        # in the process of flattening/unflattening in vmap, `__init__`
        # can be called with dummy objects
        # we optionally access `shape` in order to get the batch size
        if self.x is None:
            return 0
        return self.x.shape

    @property
    def is_symmetric(self) -> bool:  # noqa: D102
        return self.x.shape[0] == self.x.shape[1]

    @property
    def inv_scale_cost(self) -> float:  # noqa: D102
        if isinstance(self._scale_cost,
                     (int, float)) or utils.is_jax_array(self._scale_cost):
            return 1.0 / self._scale_cost
        self = self._masked_geom()
        if self._scale_cost == "max_cost":
            if self.is_online:
                return 1.0 / self._compute_summary_online(self._scale_cost)
            return 1.0 / jnp.max(self._compute_cost_matrix())
        if self._scale_cost == "mean":
            if self.is_online:
                return 1.0 / self._compute_summary_online(self._scale_cost)
            if self.shape[0] > 0:
                geom = self._masked_geom(mask_value=jnp.nan)._compute_cost_matrix()
                return 1.0 / jnp.nanmean(geom)
            return 1.0
        if self._scale_cost == "median":
            if not self.is_online:
                geom = self._masked_geom(mask_value=jnp.nan)
                return 1.0 / jnp.nanmedian(geom._compute_cost_matrix())
            raise NotImplementedError(
                "Using the median as scaling factor for "
                "the cost matrix with the online mode is not implemented."
            )
        if self._scale_cost == "max_norm":
            if self.cost_fn.norm is not None:
                return 1.0 / jnp.maximum(self._norm_x.max(), self._norm_y.max())
            return 1.0
        if self._scale_cost == "max_bound":
            if self.is_squared_euclidean:
                x_argmax = jnp.argmax(self._norm_x)
                y_argmax = jnp.argmax(self._norm_y)
                max_bound = (
                    self._norm_x[x_argmax] + self._norm_y[y_argmax] +
                    2 * jnp.sqrt(self._norm_x[x_argmax] * self._norm_y[y_argmax])
                )
                return 1.0 / max_bound
            raise NotImplementedError(
                "Using max_bound as scaling factor for "
                "the cost matrix when the cost is not squared euclidean "
                "is not implemented."
            )
        raise ValueError(f"Scaling {self._scale_cost} not implemented.")

    def _compute_cost_matrix(self) -> jnp.ndarray:
        cost_matrix = ...
        return cost_matrix

    def apply_cost(
        self,
        arr: jnp.ndarray,
        axis: int = 0,
        fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        is_linear: bool = False,
    ) -> jnp.ndarray:

        if self.is_squared_euclidean and (fn is None or is_linear):
            return self.vec_apply_cost(arr, axis, fn=fn)

        return self._apply_cost(arr, axis, fn=fn)

    def _apply_cost(
        self, arr: jnp.ndarray, axis: int = 0, fn=None
    ) -> jnp.ndarray:
        """See :meth:`apply_cost`."""
        if not self.is_online:
            return super().apply_cost(arr, axis, fn)

        app = jax.vmap(
            _apply_cost_xy,
            in_axes=[None, 0, None, self._axis_norm, None, None, None, None]
        )
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if axis == 0:
            return app(
                self.x, self.y, self._norm_x, self._norm_y, arr, self.cost_fn,
                self.inv_scale_cost, fn
            )
        return app(
            self.y, self.x, self._norm_y, self._norm_x, arr, self.cost_fn,
            self.inv_scale_cost, fn
        )

    def barycenter(self, weights: jnp.ndarray) -> jnp.ndarray:
        """Compute barycenter of points in self.x using weights."""
        return self.cost_fn.barycenter(self.x, weights)[0]

    def tree_flatten(self):  # noqa: D102
        return (
            self.x,
            self.y,
            self._src_mask,
            self._tgt_mask,
            self._epsilon_init,
            self.cost_fn,
        ), {
            "batch_size": self._batch_size,
            "scale_cost": self._scale_cost
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        x, y, src_mask, tgt_mask, epsilon, cost_fn = children
        return cls(
            x,
            y,
            cost_fn=cost_fn,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            epsilon=epsilon,
            **aux_data
        )

