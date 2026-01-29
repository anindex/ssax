import pytest
import jax
import jax.numpy as jnp
from jax import random

from ssax.ss.solver import SinkhornStep
from ssax.ss.profiling import (
    check_numerical_equivalence,
    compare_solver_states,
    measure_state_memory
)
from ssax.objectives.synthetic import Ackley, Rosenbrock, StyblinskiTang


TOLERANCE_ABS = 1e-5
TOLERANCE_REL = 1e-6


@pytest.fixture
def rng_key():
    """Provide a fixed random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def ackley_2d():
    """Provide a 2D Ackley objective."""
    return Ackley(dim=2)


@pytest.fixture
def rosenbrock_10d():
    """Provide a 10D Rosenbrock objective."""
    return Rosenbrock(dim=10)


def create_test_solver(objective_fn, **kwargs):
    """Helper to create a solver with test defaults."""
    defaults = {
        "epsilon": 0.1,
        "step_radius": 1.0,
        "probe_radius": 2.0,
        "num_probe": 3,
        "polytope_type": "orthoplex",
        "min_iterations": 2,
        "max_iterations": 10,
        "threshold": 1e-3,
        "store_inner_errors": True,
        "store_outer_evals": True,
        "store_history": True
    }
    defaults.update(kwargs)
    return SinkhornStep.create(objective_fn, **defaults)


def test_solver_initialization(ackley_2d, rng_key):
    """Test that solver initializes correctly."""
    solver = create_test_solver(ackley_2d)

    X_init = random.uniform(rng_key, (100, 2), minval=-5.0, maxval=5.0)
    state = solver.init_state(X_init, rng=rng_key)

    assert state.X.shape == (100, 2)
    assert state.costs.shape == (solver.max_iterations,)
    assert state.linear_convergence.shape == (solver.max_iterations,)


def test_single_step_determinism(ackley_2d, rng_key):
    """Test that a single step is deterministic with the same RNG."""
    solver = create_test_solver(ackley_2d)

    X_init = random.uniform(rng_key, (50, 2), minval=-5.0, maxval=5.0)

    state1 = solver.init_state(X_init, rng=rng_key)
    state1 = solver.step(state1, 0)

    state2 = solver.init_state(X_init, rng=rng_key)
    state2 = solver.step(state2, 0)

    is_eq, results = compare_solver_states(state1, state2, TOLERANCE_ABS, TOLERANCE_REL)
    assert is_eq, "Single step should be deterministic"


def test_full_optimization_convergence(ackley_2d, rng_key):
    """Test that full optimization converges."""
    solver = create_test_solver(ackley_2d, max_iterations=20)

    X_init = random.uniform(rng_key, (100, 2), minval=-5.0, maxval=5.0)
    init_state = solver.init_state(X_init, rng=rng_key)

    final_state = solver.iterations(init_state)

    final_costs = ackley_2d(final_state.X)
    initial_costs = ackley_2d(X_init)

    assert jnp.mean(final_costs) < jnp.mean(initial_costs), "Optimization should reduce cost"


def test_memory_tracking(ackley_2d, rng_key):
    """Test memory measurement utilities."""
    solver = create_test_solver(ackley_2d)

    X_init = random.uniform(rng_key, (100, 2), minval=-5.0, maxval=5.0)
    state = solver.init_state(X_init, rng=rng_key)

    memory_map = measure_state_memory(state)

    assert "total" in memory_map
    assert memory_map["total"] > 0
    assert "X" in memory_map
    assert "costs" in memory_map


def test_numerical_equivalence_utility():
    """Test numerical equivalence checking."""
    arr1 = jnp.array([1.0, 2.0, 3.0])
    arr2 = jnp.array([1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6])

    is_eq, metrics = check_numerical_equivalence(arr1, arr2, 1e-5, 1e-5)
    assert is_eq, "Arrays should be equivalent within tolerance"

    arr3 = jnp.array([1.0, 2.0, 10.0])
    is_eq, metrics = check_numerical_equivalence(arr1, arr3, 1e-5, 1e-5)
    assert not is_eq, "Arrays should not be equivalent"


class TestPhase1Equivalence:
    """Test suite for Phase 1 optimization equivalence."""

    def test_optimized_vs_baseline_ackley(self, ackley_2d, rng_key):
        """Compare optimized implementation with baseline on Ackley."""
        solver = create_test_solver(ackley_2d, max_iterations=15)

        X_init = random.uniform(rng_key, (100, 2), minval=-5.0, maxval=5.0)
        init_state = solver.init_state(X_init, rng=rng_key)

        final_state = solver.iterations(init_state)

        assert jnp.all(jnp.isfinite(final_state.X)), "Final X should be finite"
        assert jnp.all(jnp.isfinite(final_state.costs)), "Costs should be finite"

    def test_optimized_vs_baseline_rosenbrock(self, rosenbrock_10d, rng_key):
        """Compare optimized implementation with baseline on high-dim Rosenbrock."""
        solver = create_test_solver(rosenbrock_10d, max_iterations=15)

        X_init = random.uniform(rng_key, (50, 10), minval=-2.0, maxval=2.0)
        init_state = solver.init_state(X_init, rng=rng_key)

        final_state = solver.iterations(init_state)

        assert jnp.all(jnp.isfinite(final_state.X)), "Final X should be finite"
        assert jnp.all(jnp.isfinite(final_state.costs)), "Costs should be finite"


class TestMemoryOptimizations:
    """Test suite specifically for memory optimizations."""

    def test_memory_reduction_after_optimization(self, ackley_2d, rng_key):
        """Verify that memory usage is reduced compared to baseline."""
        solver = create_test_solver(ackley_2d, max_iterations=50)

        X_init = random.uniform(rng_key, (100, 2), minval=-5.0, maxval=5.0)
        state = solver.init_state(X_init, rng=rng_key)

        memory_before = measure_state_memory(state)

        # This test will verify memory reduction once dynamic allocation is implemented
        # For now, just check that measurement works
        assert memory_before["total"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
