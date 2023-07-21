from typing import Any, Dict, Optional, Sequence, Tuple, Union, NamedTuple

import jax
from jax import jit
import jax.numpy as jnp

from ott.solvers.linear import continuous_barycenter, sinkhorn, sinkhorn_lr
from ott.problems.linear import barycenter_problem, linear_problem
from ott.geometry.epsilon_scheduler import Epsilon

from ssax.ss.costs import GenericCost
from ssax.ss.utils import default_prng_key
from ssax.objectives.base import ObjectiveFn
from ssax.ss.polytopes import POLYTOPE_MAP, SAMPLE_POLYTOPE_MAP


__all__ = ["SinkhornStep"]

State = Union["sinkhorn.SinkhornState", "sinkhorn_lr.LRSinkhornState",
              "continuous_barycenter.FreeBarycenterState"]


class SinkhornStepState(NamedTuple):
    """Holds the state of the Wasserstein barycenter solver.

    Args:
        costs: Holds the sequence of regularized GW costs seen through the outer
        loop of the solver.
        linear_convergence: Holds the sequence of bool convergence flags of the
        inner Sinkhorn iterations.
        errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
        at each iteration.
        X: optimizing points.
        a: weights of the barycenter. (not using)
    """

    costs: Optional[jnp.ndarray] = None
    linear_convergence: Optional[jnp.ndarray] = None
    errors: Optional[jnp.ndarray] = None
    X: Optional[jnp.ndarray] = None
    a: Optional[jnp.ndarray] = None

    def set(self, **kwargs: Any) -> "SinkhornStepState":
        """Return a copy of self, possibly with overwrites."""
        return self._replace(**kwargs)



@jax.tree_util.register_pytree_node_class
class SinkhornStep:
    """Sinkhorn Step solver for problems that use a linear problem in inner loop."""

    def __init__(
        self,
        objective_fn: ObjectiveFn,
        polytope_type: str = "orthoplex",
        epsilon: Optional[Union[Epsilon, float]] = None,
        step_radius: float = 1.,
        probe_radius: float = 2.,
        random_probe: bool = False,
        num_sphere_point: int = 50,
        num_probe: int = 5,
        rank: int = -1,
        linear_ot_solver: Optional[Union["sinkhorn.Sinkhorn",
                                        "sinkhorn_lr.LRSinkhorn"]] = None,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        store_inner_errors: bool = False,
        rng: Optional[jax.random.PRNGKeyArray] = None,
        **kwargs: Any,
    ):
        default_epsilon = 1.0

        self.objective_fn = objective_fn
        self.cost = None  # type: GenericCost

        # Sinkhorn Step params
        self.polytope_type = polytope_type
        if self.polytope_type in POLYTOPE_MAP:
            self.direction_set = 'polytope'
        else:
            self.direction_set = 'random'
        self.polytope_vertices = None
        self.epsilon = epsilon if epsilon is not None else default_epsilon
        self.step_radius = step_radius
        self.probe_radius = probe_radius
        self.random_probe = random_probe
        self.num_sphere_point = num_sphere_point
        self.num_probe = num_probe

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
        else:  # NOTE: current implementation does not support low-rank solvers
            self.linear_ot_solver = sinkhorn.Sinkhorn(**kwargs)

        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.store_inner_errors = store_inner_errors
        self.rng = default_prng_key(rng)
        self._kwargs = kwargs

    @property
    def is_low_rank(self) -> bool:
        """Whether the solver is low-rank."""
        return self.rank > 0
    
    def init_state(
        self,
        X_init: Optional[jnp.ndarray] = None,
    ) -> SinkhornStepState:
        """Initialize the state of the Wasserstein barycenter iterations.

        Args:
        bar_prob: The barycenter problem.
        bar_size: Size of the barycenter.
        X_init: Initial barycenter estimate of shape ``[bar_size, ndim]``.
            If `None`, ``bar_size`` points will be sampled from the input
            measures according to their weights
            :attr:`~ott.problems.linear.barycenter_problem.FreeBarycenterProblem.flattened_y`.
        rng: Random key for seeding.

        Returns:
        The initial barycenter state.
        """
        num_points, dim = X_init.shape
        a = jnp.ones((num_points,)) / num_points
        num_iter = self.max_iterations
        if self.store_inner_errors:
            errors = -jnp.ones((
                self.linear_ot_solver.outer_iterations,
                num_iter,
            ))
        else:
            errors = None

        # init polytope vertices
        self.polytope_vertices = POLYTOPE_MAP[self.polytope_type](jnp.zeros((dim,)))

        # init cost
        self.cost = GenericCost(self.objective_fn)

        # init uniform weights
        # TODO: support non-uniform weights for conditional sinkhorn step
        self.a = a
        self.b = jnp.ones((self.polytope_vertices.shape[0],)) / self.polytope_vertices.shape[0]

        return SinkhornStepState(
            -jnp.ones((num_iter,)), -jnp.ones((num_iter,)), errors, X_init, a
        )
    
    @jit
    def _step(self, state: State, iteration: int) -> State:
        """Run one iteration of the Sinkhorn algorithm."""
        X = state.X

        eps = self.epsilon.at(iteration) if isinstance(self.epsilon, Epsilon) else self.epsilon
        step_radius = self.step_radius * eps
        probe_radius = self.probe_radius * eps
        
        # compute sampled polytope vertices
        X_vertices, X_probe, vertices = SAMPLE_POLYTOPE_MAP[self.direction_set](X,
                                                                                polytope_vertices=self.polytope_vertices,
                                                                                step_radius=step_radius,
                                                                                probe_radius=probe_radius,
                                                                                num_probe=self.num_probe,
                                                                                random_probe=self.random_probe,
                                                                                num_sphere_point=self.num_sphere_point,
                                                                                rng=self.rng)

        # solve Sinkhorn
        self.cost.X = X_probe
        res = self.linear_ot_solver(
            linear_problem.LinearProblem(
                self.cost, self.a, self.b
            )
        )

        # barycentric projection
        X_new = jnp.einsum('bik,bi->bk', X_vertices, res.matrix / self.a[..., jnp.newaxis])

        # cost = jnp.sum(res.reg_ot_costs)
        # updated_costs = self.costs.at[iteration].set(cost)
        # converged = jnp.all(convergeds)
        # linear_convergence = self.linear_convergence.at[iteration].set(converged)

        # if store_errors and self.errors is not None:
        # errors = self.errors.at[iteration, :, :].set(errors)
        # else:
        # errors = None


    
    def iterations(self, X_init: jnp.ndarray) -> State:
        """Run the Sinkhorn iterations.

        Args:
        X_init: Initial barycenter estimate of shape ``[bar_size, ndim]``.
            If `None`, ``bar_size`` points will be sampled from the input
            measures according to their weights
            :attr:`~ott.problems.linear.barycenter_problem.FreeBarycenterProblem.flattened_y`.

        Returns:
        The final barycenter state.
        """
        state = self.init_state(X_init)
        iteration = 0
        while self._continue(state, iteration):
            state = self._step(state, iteration)
            iteration += 1
        return state
        

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
