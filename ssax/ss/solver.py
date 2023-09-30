from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable

import jax
from jax import jit, random
import jax.numpy as jnp
from flax import struct
from functools import partial
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.problems.linear import linear_problem
from ott.geometry.epsilon_scheduler import Epsilon
from ott.math import fixed_point_loop

from ssax.ss.costs import GenericCost
from ssax.ss.utils import default_prng_key
from ssax.ss.polytopes import POLYTOPE_MAP, get_sampled_polytope_vertices, get_sampled_points_on_sphere


__all__ = ["SinkhornStepState", "SinkhornStep"]


def outer_loop(
    cond_fn: Callable[[int, Any, Any], bool],
    body_fn: Callable[[Any, Any, Any, Any], Any], min_iterations: int,
    max_iterations: int, inner_iterations: int, constants: Any, state: Any
):
    """Fixed point iteration with early stopping."""
    compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1

    def max_cond_fn(iteration_state):
        iteration, state = iteration_state
        return jnp.logical_and(
            iteration < max_iterations,
            jnp.logical_or(
                iteration < min_iterations, cond_fn(iteration, constants, state)
            )
        )

    def unrolled_body_fn(iteration_state):

        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None

        iteration_state, _ = jax.lax.scan(
            one_iteration, iteration_state, compute_error_flags
        )
        return iteration_state

    _, state = jax.lax.while_loop(max_cond_fn, unrolled_body_fn, (0, state))
    return state


@struct.dataclass
class SinkhornStepState():
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

    costs: jnp.array = None
    X: jnp.array = None
    linear_convergence: Optional[jnp.array] = None
    sinkhorn_errors: Optional[jnp.array] = None
    objective_vals: Optional[jnp.array] = None
    displacement_sqnorms: Optional[jnp.array] = None
    X_history: Optional[jnp.array] = None
    a: Optional[jnp.array] = None
    rng: Optional[jax.random.PRNGKeyArray] = None

    def set(self, **kwargs: Any) -> "SinkhornStepState":
        """Return a copy of self, possibly with overwrites."""
        return self.replace(**kwargs)


State = SinkhornStepState


@struct.dataclass
class SinkhornStep():
    """Sinkhorn Step solver for problems that use a linear problem in inner loop."""

    objective_fn: Any = None
    dim: int = None
    polytope_vertices: jnp.array = None
    linear_ot_solver: Union["sinkhorn.Sinkhorn", "sinkhorn_lr.LRSinkhorn"] = None
    epsilon: Union[Epsilon, float] = 0.1
    ent_epsilon: Union[Epsilon, float] = None
    ent_relative_epsilon: bool = False
    scale_cost: float = 1.0
    step_radius: float = 1.
    probe_radius: float = 2.
    probes: jnp.array = None
    rank: int = -1
    min_iterations: int = 5
    max_iterations: int = 50
    threshold: float = 1e-3
    store_inner_errors: bool = False
    store_outer_evals: bool = False
    store_history: bool = False

    @classmethod
    def create(
        cls,
        objective_fn: Any,
        linear_ot_solver: Optional[Union["sinkhorn.Sinkhorn",
                                         "sinkhorn_lr.LRSinkhorn"]] = None,
        epsilon: Optional[Union[Epsilon, float]] = 0.1,
        ent_epsilon: Optional[Union[Epsilon, float]] = None,
        ent_relative_epsilon: Optional[bool] = None,
        scale_cost: Optional[Union[str, float]] = 1.0,
        step_radius: float = 1.,
        probe_radius: float = 2.,
        num_probe: int = 5,
        polytope_type: str = 'orthoplex',
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        rank: int = -1,
        store_inner_errors: bool = False,
        store_outer_evals: bool = False,
        store_history: bool = False,
        **kwargs: Any,
    ) -> "SinkhornStep":
        is_low_rank = rank > 0
        dim = objective_fn.dim

        # Sinkhorn Step params
        use_polytope = polytope_type in POLYTOPE_MAP
        polytope_vertices = POLYTOPE_MAP[polytope_type](jnp.zeros((dim,)))
        probes = jnp.linspace(0, 1, num_probe + 2)[1:num_probe + 1]
        if linear_ot_solver is None:
            if is_low_rank:
                if epsilon is None:
                    # Use default entropic regularization in LRSinkhorn if None was passed
                    linear_ot_solver = sinkhorn_lr.LRSinkhorn(
                        rank=rank, **kwargs
                    )
                else:
                    # If epsilon is passed, use it to replace the default LRSinkhorn value
                    linear_ot_solver = sinkhorn_lr.LRSinkhorn(
                        rank=rank, epsilon=epsilon, **kwargs
                    )
        else:  # NOTE: current implementation does not support low-rank solvers
            linear_ot_solver = sinkhorn.Sinkhorn(**kwargs)

        # init uniform weights
        # TODO: support non-uniform weights for conditional sinkhorn step
        # self.a = jnp.ones((num_points,)) / num_points
        # self.b = jnp.ones((self.polytope_vertices.shape[0],)) / self.polytope_vertices.shape[0]
        return cls(
            objective_fn=objective_fn,
            dim=dim,
            polytope_vertices=polytope_vertices,
            linear_ot_solver=linear_ot_solver,
            epsilon=epsilon,
            ent_epsilon=ent_epsilon,
            ent_relative_epsilon=ent_relative_epsilon,
            scale_cost=scale_cost,
            step_radius=step_radius,
            probe_radius=probe_radius,
            probes=probes,
            rank=rank,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            store_inner_errors=store_inner_errors,
            store_outer_evals=store_outer_evals,
            store_history=store_history,
            **kwargs
        )

    @property
    def is_low_rank(self) -> bool:
        """Whether the solver is low-rank."""
        return self.rank > 0

    def init_state(
        self,
        X_init: jnp.array,
        rng: Optional[jax.random.PRNGKeyArray] = None,
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

        num_iter = self.max_iterations
        sinkhorn_errors = -jnp.ones((
            num_iter,
            self.linear_ot_solver.outer_iterations,
        ))
        objective_vals = -jnp.ones((num_iter, num_points))
        X_history = jnp.zeros((num_iter, num_points, dim))

        # NOTE: uniform weights for now
        a = jnp.ones((num_points,)) / num_points

        return SinkhornStepState(
            costs=-jnp.ones((num_iter,)), X=X_init, linear_convergence=-jnp.ones((num_iter,)),
            sinkhorn_errors=sinkhorn_errors, objective_vals=objective_vals, 
            displacement_sqnorms=-jnp.ones((num_iter,)), X_history=X_history, a=a,
            rng=default_prng_key(rng)
        )

    @jit
    def step(self, state: State, iteration: int) -> State:
        """Run one iteration of the Sinkhorn algorithm."""
        X = state.X

        eps = self.epsilon.at(iteration) if isinstance(self.epsilon, Epsilon) else self.epsilon
        step_radius = self.step_radius * eps
        probe_radius = self.probe_radius * eps
        rng, sub_rng = random.split(state.rng)

        # compute sampled polytope vertices
        X_vertices, X_probe, vertices = get_sampled_polytope_vertices(X,
                                                                      polytope_vertices=self.polytope_vertices,
                                                                      step_radius=step_radius,
                                                                      probe_radius=probe_radius,
                                                                      probes=self.probes,
                                                                      rng=sub_rng)

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

        # optimization statistics NOTE: comment those if not needed
        sinkhorn_errors = state.sinkhorn_errors.at[iteration, :].set(res.errors)
        objective_vals = state.objective_vals.at[iteration, :].set(self.objective_fn(X_new))
        X_history = state.X_history.at[iteration, :, :].set(X_new)

        displacement_sqnorms = state.displacement_sqnorms.at[iteration].set(jnp.sum((X_new - X)**2, axis=-1).mean())

        return state.set(
            costs=updated_costs,
            linear_convergence=linear_convergence,
            sinkhorn_errors=sinkhorn_errors,
            objective_vals=objective_vals,
            displacement_sqnorms=displacement_sqnorms,
            X=X_new,
            X_history=X_history,
            rng=rng,
        )

    def warm_start(self, X_init: jnp.array, rng: random.PRNGKeyArray = None) -> State:
        """
        Warm start the Sinkhorn Step solver by compiling the step function.
        This returns the initial state of the solver.
        """
        init_state = self.init_state(X_init, rng=rng)
        self.step(self.init_state(X_init), 0)
        return init_state

    def iterations(self, init_state: State) -> State:
        """JITable Sinkhorn Step outer loop.

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

        state = outer_loop(
            cond_fn=cond_fn,
            body_fn=body_fn,
            min_iterations=self.min_iterations,
            max_iterations=self.max_iterations,
            inner_iterations=1,
            constants=(self, None),
            state=init_state
        )

        return self.output_from_state(state)

    def output_from_state(self, state: SinkhornStepState) -> SinkhornStepState:
        return state

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
