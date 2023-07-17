from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ott.solvers.linear import continuous_barycenter, sinkhorn, sinkhorn_lr

__all__ = ["SinkhornStep"]

State = Union["sinkhorn.SinkhornState", "sinkhorn_lr.LRSinkhornState",
              "continuous_barycenter.FreeBarycenterState"]


@jax.tree_util.register_pytree_node_class
class SinkhornStep:
    """Sinkhorn Step solver for problems that use a linear problem in inner loop."""

    def __init__(
        self,
        epsilon: Optional[float] = None,
        rank: int = -1,
        linear_ot_solver: Optional[Union["sinkhorn.Sinkhorn",
                                        "sinkhorn_lr.LRSinkhorn"]] = None,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        store_inner_errors: bool = False,
        **kwargs: Any,
    ):
        default_epsilon = 1.0

        self.epsilon = epsilon if epsilon is not None else default_epsilon
        self.rank = rank
        self.linear_ot_solver = linear_ot_solver
        if self.linear_ot_solver is None:
            if self.is_low_rank:
                if epsilon is None:
                    # Use default entropic regularization in LRSinkhorn if None was passed
                    self.linear_ot_solver = sinkhorn_lr.LRSinkhorn(
                        rank=self.rank, **kwargs
                    )
                else:
                    # If epsilon is passed, use it to replace the default LRSinkhorn value
                    self.linear_ot_solver = sinkhorn_lr.LRSinkhorn(
                        rank=self.rank, epsilon=self.epsilon, **kwargs
                    )
        else:
            self.linear_ot_solver = sinkhorn.Sinkhorn(**kwargs)

        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.store_inner_errors = store_inner_errors
        self._kwargs = kwargs

    @property
    def is_low_rank(self) -> bool:
        """Whether the solver is low-rank."""
        return self.rank > 0

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        return ([self.epsilon, self.linear_ot_solver, self.threshold], {
            "min_iterations": self.min_iterations,
            "max_iterations": self.max_iterations,
            "rank": self.rank,
            "store_inner_errors": self.store_inner_errors,
            **self._kwargs
        })

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "SinkhornStep":
        epsilon, linear_ot_solver, threshold = children
        return cls(
            epsilon=epsilon,
            linear_ot_solver=linear_ot_solver,
            threshold=threshold,
            **aux_data
        )

    def _converged(self, state: State, iteration: int) -> bool:
        costs, i, tol = state.costs, iteration, self.threshold
        return jnp.logical_and(
            i >= 2, jnp.isclose(costs[i - 2], costs[i - 1], rtol=tol)
        )

    def _diverged(self, state: State, iteration: int) -> bool:
        return jnp.logical_not(jnp.isfinite(state.costs[iteration - 1]))

    def _continue(self, state: State, iteration: int) -> bool:
        """Continue while not(converged) and not(diverged)."""
        return jnp.logical_or(
            iteration <= 2,
            jnp.logical_and(
                jnp.logical_not(self._diverged(state, iteration)),
                jnp.logical_not(self._converged(state, iteration))
            )
        )
