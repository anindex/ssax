from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random, grad, vmap
from flax import struct

from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.problems.linear import linear_problem
from ott.geometry.epsilon_scheduler import Epsilon

from .costs import GenericCost
from .utils import default_prng_key
from .polytopes import POLYTOPE_MAP, get_sampled_polytope_vertices

__all__ = ["SinkhornStepState", "SinkhornStep"]


# -----------------------------------------------------------------------------
# Outer Loop (Fixed Point Iteration)
# -----------------------------------------------------------------------------

def outer_loop(
    cond_fn: Callable[[int, Any, Any], bool],
    body_fn: Callable[[Any, Any, Any, Any], Any], 
    min_iterations: int,
    max_iterations: int, 
    inner_iterations: int, 
    constants: Any, 
    state: Any
):
    """Fixed point iteration with early stopping and unrolled inner loops."""
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


# -----------------------------------------------------------------------------
# Solver State
# -----------------------------------------------------------------------------

@struct.dataclass
class SinkhornStepState:
    """Holds the state of the Sinkhorn Step solver."""
    costs: jax.Array
    X: jax.Array
    linear_convergence: jax.Array
    displacement_sqnorms: jax.Array
    a: jax.Array
    rng: jax.Array
    iteration_count: int = 0

    # Dual potentials for warm-starting OT solver
    f: Optional[jax.Array] = None
    g: Optional[jax.Array] = None

    # Optional tracking fields
    sinkhorn_errors: Optional[jax.Array] = None
    objective_vals: Optional[jax.Array] = None
    cosin_similarity: Optional[jax.Array] = None
    X_history: Optional[jax.Array] = None

    def set(self, **kwargs: Any) -> "SinkhornStepState":
        """Return a copy of self, possibly with overwrites."""
        return self.replace(**kwargs)

State = SinkhornStepState


# -----------------------------------------------------------------------------
# Sinkhorn Step Solver
# -----------------------------------------------------------------------------

@struct.dataclass
class SinkhornStep:
    """Batch gradient-free solver for non-convex objectives using Entropic OT."""

    objective_fn: Any = struct.field(pytree_node=False)
    dim: int = struct.field(pytree_node=False)
    
    # Geometry configurations (Arrays must be pytree nodes)
    polytope_vertices: jax.Array = struct.field(pytree_node=True)
    probes: jax.Array = struct.field(pytree_node=True)
    
    # Solver configurations
    linear_ot_solver: Union["sinkhorn.Sinkhorn", "sinkhorn_lr.LRSinkhorn"] = struct.field(pytree_node=True)
    epsilon: Union[Epsilon, float] = struct.field(default=0.1, pytree_node=True)
    ent_epsilon: Union[Epsilon, float] = struct.field(default=None, pytree_node=True)
    ent_relative_epsilon: Optional[bool] = struct.field(default=None, pytree_node=False)
    
    # Hyperparameters
    scale_cost: float = struct.field(default=1.0, pytree_node=False)
    step_radius: float = struct.field(default=1.0, pytree_node=False)
    probe_radius: float = struct.field(default=2.0, pytree_node=False)
    rank: int = struct.field(default=-1, pytree_node=False)
    
    # Loop controls
    min_iterations: int = struct.field(default=5, pytree_node=False)
    max_iterations: int = struct.field(default=50, pytree_node=False)
    threshold: float = struct.field(default=1e-3, pytree_node=False)
    
    # Logging flags
    store_inner_errors: bool = struct.field(default=False, pytree_node=False)
    store_outer_evals: bool = struct.field(default=False, pytree_node=False)
    store_history: bool = struct.field(default=False, pytree_node=False)
    store_cosine_similarity: bool = struct.field(default=False, pytree_node=False)

    @classmethod
    def create(
        cls,
        objective_fn: Any,
        linear_ot_solver: Optional[Union["sinkhorn.Sinkhorn", "sinkhorn_lr.LRSinkhorn"]] = None,
        epsilon: Optional[Union[Epsilon, float]] = 0.1,
        ent_epsilon: Optional[Union[Epsilon, float]] = None,
        ent_relative_epsilon: Optional[bool] = None,
        scale_cost: Union[str, float] = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        num_probe: int = 5,
        polytope_type: str = 'orthoplex',
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        rank: int = -1,
        store_inner_errors: bool = False,
        store_outer_evals: bool = False,
        store_history: bool = False,
        store_cosine_similarity: bool = False,
        **kwargs: Any,
    ) -> "SinkhornStep":
        
        dim = objective_fn.dim
        is_low_rank = rank > 0

        # Generate template vertices
        polytope_vertices = POLYTOPE_MAP[polytope_type](jnp.zeros((dim,)), radius=1.0)
        
        # Generate probe scales
        probes = jnp.linspace(0, 1, num_probe + 2)[1:num_probe + 1]

        # Initialize OTT Solver
        if linear_ot_solver is None:
            if is_low_rank:
                # For Low-Rank, we need a safe default epsilon if none provided
                solver_eps = epsilon if epsilon is not None else 0.1
                linear_ot_solver = sinkhorn_lr.LRSinkhorn(
                    rank=rank, epsilon=solver_eps, **kwargs
                )
            else:
                linear_ot_solver = sinkhorn.Sinkhorn(**kwargs)

        return cls(
            objective_fn=objective_fn,
            dim=dim,
            polytope_vertices=polytope_vertices,
            probes=probes,
            linear_ot_solver=linear_ot_solver,
            epsilon=epsilon,
            ent_epsilon=ent_epsilon,
            ent_relative_epsilon=ent_relative_epsilon,
            scale_cost=scale_cost,
            step_radius=step_radius,
            probe_radius=probe_radius,
            rank=rank,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            store_inner_errors=store_inner_errors,
            store_outer_evals=store_outer_evals,
            store_history=store_history,
            store_cosine_similarity=store_cosine_similarity
        )

    def init_state(
        self,
        X_init: jax.Array,
        rng: Optional[jax.Array] = None,
    ) -> SinkhornStepState:
        """Initialize the solver state."""
        num_points, _ = X_init.shape
        num_iter = self.max_iterations

        def init_log(shape, enabled):
            return jnp.full(shape, -1.0) if enabled else None

        sinkhorn_errors = init_log(
            (num_iter, self.linear_ot_solver.outer_iterations), self.store_inner_errors
        )
        objective_vals = init_log((num_iter, num_points), self.store_outer_evals)
        X_history = init_log((num_iter, num_points, self.dim), self.store_history)
        if self.store_history:
             X_history = X_history.at[0].set(X_init)
        
        cosin_similarity = init_log((num_iter, num_points), self.store_cosine_similarity)

        # Uniform weights
        a = jnp.ones((num_points,)) / num_points

        return SinkhornStepState(
            costs=jnp.full((num_iter,), -1.0),
            X=X_init,
            linear_convergence=jnp.full((num_iter,), -1.0),
            sinkhorn_errors=sinkhorn_errors,
            objective_vals=objective_vals,
            displacement_sqnorms=jnp.full((num_iter,), -1.0),
            cosin_similarity=cosin_similarity,
            X_history=X_history,
            a=a,
            rng=default_prng_key(rng)
        )

    @partial(jit, static_argnames=['iteration'])
    def step(self, state: State, iteration: int) -> State:
        """Run one iteration of the Sinkhorn Step algorithm."""
        X = state.X
        rng, sub_rng = random.split(state.rng)

        # 1. Resolve Epsilon & Radii
        current_eps = self.epsilon.at(iteration) if isinstance(self.epsilon, Epsilon) else self.epsilon
        step_radius = self.step_radius * current_eps
        probe_radius = self.probe_radius * current_eps

        # 2. Sample Polytope & Probes
        X_vertices, X_probe, _ = get_sampled_polytope_vertices(
            origin=X,
            probes=self.probes,
            polytope_vertices=self.polytope_vertices,
            step_radius=step_radius,
            probe_radius=probe_radius,
            rng=sub_rng
        )

        # 3. Define Cost & Solve OT
        cost_geometry = GenericCost(
            self.objective_fn,
            X_probe,
            epsilon=self.ent_epsilon,
            relative_epsilon=self.ent_relative_epsilon,
            scale_cost=self.scale_cost
        )

        ot_prob = linear_problem.LinearProblem(cost_geometry)

        # Warm-start from previous dual potentials if available
        init_potentials = None if state.f is None else (state.f, state.g)
        ot_out = self.linear_ot_solver(ot_prob, init=init_potentials)

        # 4. Extract dual potentials for next iteration warm-start
        new_f = ot_out.f
        new_g = ot_out.g

        # 5. Barycentric Projection
        transport_matrix = ot_out.matrix
        weights = transport_matrix / state.a[..., None]

        # X_new[b, d] = sum_k (X_vertices[b, k, d] * weights[b, k])
        X_new = jnp.einsum('bkd,bk->bd', X_vertices, weights)

        # 5. Update Statistics
        updated_costs = state.costs.at[iteration].set(ot_out.ent_reg_cost)
        linear_convergence = state.linear_convergence.at[iteration].set(ot_out.converged)
        displacement_sqnorms = state.displacement_sqnorms.at[iteration].set(
            jnp.mean(jnp.sum((X_new - X)**2, axis=-1))
        )

        # Optional Logs
        sinkhorn_errors = state.sinkhorn_errors
        if sinkhorn_errors is not None:
            sinkhorn_errors = sinkhorn_errors.at[iteration, :].set(ot_out.errors)

        objective_vals = state.objective_vals
        if objective_vals is not None:
            objective_vals = objective_vals.at[iteration, :].set(self.objective_fn(X_new))

        X_history = state.X_history
        if X_history is not None:
            X_history = X_history.at[iteration, :, :].set(X_new)

        cosin_similarity = state.cosin_similarity
        if cosin_similarity is not None:
            grad_fn = grad(self.objective_fn)
            true_grad = -vmap(grad_fn)(X_new)
            ss_step = X_new - X
            
            tg_norm = jnp.linalg.norm(true_grad, axis=-1)
            ss_norm = jnp.linalg.norm(ss_step, axis=-1)
            
            safe_denom = jnp.where(tg_norm * ss_norm == 0, 1.0, tg_norm * ss_norm)
            cosin = jnp.einsum('bi,bi->b', true_grad, ss_step) / safe_denom
            cosin_similarity = cosin_similarity.at[iteration, :].set(cosin)

        return state.set(
            costs=updated_costs,
            linear_convergence=linear_convergence,
            sinkhorn_errors=sinkhorn_errors,
            objective_vals=objective_vals,
            displacement_sqnorms=displacement_sqnorms,
            cosin_similarity=cosin_similarity,
            X=X_new,
            X_history=X_history,
            rng=rng,
            iteration_count=iteration + 1,
            f=new_f,
            g=new_g
        )

    def trim_state(self, state: State) -> State:
        """Trim tracking arrays to actual iteration count to save memory."""
        actual_iters = state.iteration_count

        if actual_iters == 0 or actual_iters >= self.max_iterations:
            return state

        trimmed_costs = state.costs[:actual_iters]
        trimmed_convergence = state.linear_convergence[:actual_iters]
        trimmed_displacement = state.displacement_sqnorms[:actual_iters]

        trimmed_sinkhorn = None
        if state.sinkhorn_errors is not None:
            trimmed_sinkhorn = state.sinkhorn_errors[:actual_iters]

        trimmed_obj = None
        if state.objective_vals is not None:
            trimmed_obj = state.objective_vals[:actual_iters]

        trimmed_cosin = None
        if state.cosin_similarity is not None:
            trimmed_cosin = state.cosin_similarity[:actual_iters]

        trimmed_history = None
        if state.X_history is not None:
            trimmed_history = state.X_history[:actual_iters]

        return state.set(
            costs=trimmed_costs,
            linear_convergence=trimmed_convergence,
            displacement_sqnorms=trimmed_displacement,
            sinkhorn_errors=trimmed_sinkhorn,
            objective_vals=trimmed_obj,
            cosin_similarity=trimmed_cosin,
            X_history=trimmed_history
        )

    def warm_start(self, X_init: jax.Array, rng: Optional[jax.Array] = None) -> State:
        """Run one step to compile functions and return initial state."""
        init_state = self.init_state(X_init, rng=rng)
        return self.step(init_state, 0)

    def iterations(self, init_state: State) -> State:
        """Run the full Sinkhorn Step outer loop."""
        def cond_fn(iteration, constants, state):
            solver, _ = constants
            return solver._continue(state, iteration)

        def body_fn(iteration, constants, state, compute_error):
            del compute_error
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
        return self.trim_state(state)

    def _converged(self, state: State, iteration: int) -> bool:
        dqsnorm = state.displacement_sqnorms
        return jnp.logical_and(
            iteration >= 3, 
            jnp.isclose(dqsnorm[iteration - 2], dqsnorm[iteration - 1], rtol=self.threshold)
        )

    def _diverged(self, state: State, iteration: int) -> bool:
        return jnp.logical_not(jnp.isfinite(state.costs[iteration - 1]))

    def _continue(self, state: State, iteration: int) -> bool:
        return jnp.logical_or(
            iteration <= 3,
            jnp.logical_and(
                jnp.logical_not(self._diverged(state, iteration)),
                jnp.logical_not(self._converged(state, iteration))
            )
        )
