from typing import Any, Dict, Optional, Sequence, Tuple, Union, NamedTuple

import jax
from jax import jit, random
import jax.numpy as jnp

from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.problems.linear import linear_problem
from ott.geometry.epsilon_scheduler import Epsilon
from ott.math import fixed_point_loop

from ssax.ss.costs import GenericCost
from ssax.ss.utils import default_prng_key
from ssax.ss.polytopes import POLYTOPE_MAP, SAMPLE_POLYTOPE_MAP


__all__ = ["SinkhornStepState", "SinkhornStep"]


class SinkhornStepState(NamedTuple):
    """Holds the state of the Wasserstein barycenter solver.

    Args:
        costs: Holds the sequence of regularized GW costs seen through the outer
        loop of the solver.
        linear_convergence: Holds the sequence of bool convergence flags of the
        inner Sinkhorn iterations.
        sinkhorn_errors: Holds sequence of vectors of sinkhorn_errors of the Sinkhorn algorithm
        at each iteration.
        X: optimizing points.
        a: weights of the barycenter. (not using)
    """

    costs: Optional[jnp.array] = None
    linear_convergence: Optional[jnp.array] = None
    sinkhorn_errors: Optional[jnp.array] = None
    objective_vals: Optional[jnp.array] = None
    displacement_sqnorms: Optional[jnp.array] = None
    X: Optional[jnp.array] = None
    X_history: Optional[jnp.array] = None
    a: Optional[jnp.array] = None

    def set(self, **kwargs: Any) -> "SinkhornStepState":
        """Return a copy of self, possibly with overwrites."""
        return self._replace(**kwargs)


State = SinkhornStepState


@jax.tree_util.register_pytree_node_class
class SinkhornStep:
    """Sinkhorn Step solver for problems that use a linear problem in inner loop."""

    def __init__(
        self,
        objective_fn: Any,
        linear_ot_solver: Optional[Union["sinkhorn.Sinkhorn",
                                         "sinkhorn_lr.LRSinkhorn"]] = None,
        epsilon: Optional[Union[Epsilon, float]] = None,
        ent_epsilon: Optional[Union[Epsilon, float]] = None,
        ent_relative_epsilon: Optional[bool] = None,
        scale_cost: Optional[Union[str, float]] = 1.0,
        step_radius: float = 1.,
        probe_radius: float = 2.,
        random_probe: bool = False,
        num_sphere_point: int = 50,
        num_probe: int = 5,
        polytope_type: str = 'orthoplex',
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        rank: int = -1,
        store_inner_errors: bool = False,
        store_outer_evals: bool = False,
        store_history: bool = False,
        rng: Optional[jax.random.PRNGKeyArray] = None,
        **kwargs: Any,
    ):
        default_epsilon = 0.1

        self.objective_fn = objective_fn
        self.dim = self.objective_fn.dim

        # Sinkhorn Step params
        self.polytope_type = polytope_type
        if self.polytope_type in POLYTOPE_MAP:
            self.direction_set = 'polytope'
        else:
            self.direction_set = 'random'
        self.polytope_vertices = POLYTOPE_MAP[self.polytope_type](jnp.zeros((self.dim,)))
        self.epsilon = epsilon if epsilon is not None else default_epsilon
        self.ent_epsilon = ent_epsilon
        self.ent_relative_epsilon = ent_relative_epsilon
        self.scale_cost = scale_cost
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
        self.store_outer_evals = store_outer_evals
        self.store_history = store_history
        self.rng = default_prng_key(rng)
        self._kwargs = kwargs

        # init uniform weights
        # TODO: support non-uniform weights for conditional sinkhorn step
        # self.a = jnp.ones((num_points,)) / num_points
        # self.b = jnp.ones((self.polytope_vertices.shape[0],)) / self.polytope_vertices.shape[0]

    @property
    def is_low_rank(self) -> bool:
        """Whether the solver is low-rank."""
        return self.rank > 0
    
    def init_state(
        self,
        X_init: jnp.array,
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
        assert dim == self.dim

        num_iter = self.max_iterations
        if self.store_inner_errors:
            sinkhorn_errors = -jnp.ones((
                num_iter,
                self.linear_ot_solver.outer_iterations,
            ))
        else:
            sinkhorn_errors = None
        
        if self.store_outer_evals:
            objective_vals = -jnp.ones((num_iter, num_points))
        else:
            objective_vals = None
        
        if self.store_history:
            X_history = jnp.zeros((num_iter, num_points, dim))
        else:
            X_history = None

        # NOTE: uniform weights for now
        a = jnp.ones((num_points,)) / num_points

        return SinkhornStepState(
            -jnp.ones((num_iter,)), -jnp.ones((num_iter,)), 
            sinkhorn_errors, objective_vals, -jnp.ones((num_iter,)), X_init, X_history, a
        )

    @jit
    def step(self, state: State, iteration: int) -> State:
        """Run one iteration of the Sinkhorn algorithm."""
        X = state.X

        eps = self.epsilon.at(iteration) if isinstance(self.epsilon, Epsilon) else self.epsilon
        step_radius = self.step_radius * eps
        probe_radius = self.probe_radius * eps
        rng, subkey = random.split(self.rng)

        # compute sampled polytope vertices
        X_vertices, X_probe, vertices = SAMPLE_POLYTOPE_MAP[self.direction_set](X,
                                                                                polytope_vertices=self.polytope_vertices,
                                                                                step_radius=step_radius,
                                                                                probe_radius=probe_radius,
                                                                                num_probe=self.num_probe,
                                                                                random_probe=self.random_probe,
                                                                                num_sphere_point=self.num_sphere_point,
                                                                                rng=subkey)

        # solve Sinkhorn
        cost = GenericCost(self.objective_fn, X_probe, 
                           epsilon=self.ent_epsilon,
                           relative_epsilon=self.ent_relative_epsilon,
                           scale_cost=self.scale_cost,)
        ot_prob = linear_problem.LinearProblem(cost)
        res = self.linear_ot_solver(ot_prob)

        # barycentric projection
        X_new = jnp.einsum('bik,bi->bk', X_vertices, res.matrix / state.a[..., jnp.newaxis])

        updated_costs = state.costs.at[iteration].set(res.ent_reg_cost)
        linear_convergence = state.linear_convergence.at[iteration].set(res.converged)

        if self.store_inner_errors and state.sinkhorn_errors is not None:
            sinkhorn_errors = state.sinkhorn_errors.at[iteration, :].set(res.errors)
        else:
            sinkhorn_errors = None

        if self.store_outer_evals and state.objective_vals is not None:
            objective_vals = state.objective_vals.at[iteration, :].set(self.cost.evaluate(X_new))
        else:
            objective_vals = None
        
        if self.store_history and state.X_history is not None:
            X_history = state.X_history.at[iteration, :, :].set(X_new)
        else:
            X_history = None

        displacement_sqnorms = state.displacement_sqnorms.at[iteration].set(jnp.sum((X_new - X)**2, axis=-1).mean())

        return state.set(
            costs=updated_costs,
            linear_convergence=linear_convergence,
            sinkhorn_errors=sinkhorn_errors,
            objective_vals=objective_vals,
            displacement_sqnorms=displacement_sqnorms,
            X=X_new,
            X_history=X_history
        )

    @jit
    def iterations(self, X_init: jnp.array) -> State:
        """Jittable Sinkhorn Step outer loop.

        Args:
        X_init: Initial points of shape ``[batch, ndim]``.

        Returns:
        The final state.
        """
        def cond_fn(
            iteration: int,
            constants: Tuple[SinkhornStep,
                             linear_problem.LinearProblem],
            state: SinkhornStepState
        ) -> bool:
            solver, lin_prob = constants
            return solver._continue(state, iteration)

        def body_fn(
            iteration, constants: Tuple[SinkhornStep,
                                        linear_problem.LinearProblem],
            state: SinkhornStepState, compute_error: bool
        ) -> SinkhornStepState:
            del compute_error  # Always assumed True
            solver, _ = constants
            return solver.step(state, iteration)

        state = fixed_point_loop.fixpoint_iter(
            cond_fn=cond_fn,
            body_fn=body_fn,
            min_iterations=self.min_iterations,
            max_iterations=self.max_iterations,
            inner_iterations=1,
            constants=(self, None),
            state=self.init_state(X_init)
        )

        return self.output_from_state(state)

    def output_from_state(self, state: SinkhornStepState) -> SinkhornStepState:
        return state

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        return ([self.objective_fn, self.linear_ot_solver, self.epsilon, self.ent_epsilon], {
            "ent_relative_epsilon": self.ent_relative_epsilon,
            "scale_cost": self.scale_cost,
            "polytope_type": self.polytope_type,
            "step_radius": self.step_radius,
            "probe_radius": self.probe_radius,
            "random_probe": self.random_probe,
            "num_sphere_point": self.num_sphere_point,
            "num_probe": self.num_probe,
            "threshold": self.threshold,
            "min_iterations": self.min_iterations,
            "max_iterations": self.max_iterations,
            "rank": self.rank,
            "store_inner_errors": self.store_inner_errors,
            "store_outer_evals": self.store_outer_evals,
            "store_history": self.store_history,
            **self._kwargs
        })

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "SinkhornStep":
        objective_fn, linear_ot_solver, epsilon, ent_epsilon = children
        return cls(
            objective_fn=objective_fn,
            linear_ot_solver=linear_ot_solver,
            epsilon=epsilon,
            ent_epsilon=ent_epsilon,
            **aux_data
        )

    def _converged(self, state: State, iteration: int) -> bool:
        dqsnorm, i, tol = state.displacement_sqnorms, iteration, self.threshold
        return jnp.logical_and(
            i >= 3, jnp.isclose(dqsnorm[i - 2], dqsnorm[i - 1], rtol=tol)
        )

    def _diverged(self, state: State, iteration: int) -> bool:
        return jnp.logical_not(jnp.isfinite(state.costs[iteration - 1]))

    def _continue(self, state: State, iteration: int) -> bool:
        """Continue while not(converged) and not(diverged)."""
        return jnp.logical_or(
            iteration <= 3,
            jnp.logical_and(
                jnp.logical_not(self._diverged(state, iteration)),
                jnp.logical_not(self._converged(state, iteration))
            )
        )
