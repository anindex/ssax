import time
import functools
from typing import Callable, Any, Dict, Tuple, Optional
import jax
import jax.numpy as jnp


def memory_tracker(func: Callable) -> Callable:
    """Decorator to track peak memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import tracemalloc
            tracemalloc.start()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1024 / 1024
            print(f"{func.__name__}: Peak memory = {peak_mb:.2f} MB")
            return result
        except ImportError:
            return func(*args, **kwargs)
    return wrapper


def time_tracker(func: Callable) -> Callable:
    """Decorator to track execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: Elapsed time = {elapsed:.4f} s")
        return result
    return wrapper


def profile_solver_step(solver, state, iteration: int) -> Dict[str, Any]:
    """Profile a single solver step and return metrics."""
    start = time.perf_counter()
    new_state = solver.step(state, iteration)
    jax.block_until_ready(new_state)
    elapsed = time.perf_counter() - start

    return {
        "elapsed_time": elapsed,
        "cost": float(new_state.costs[iteration]),
        "converged": bool(new_state.linear_convergence[iteration]),
        "displacement": float(new_state.displacement_sqnorms[iteration])
    }


def measure_array_memory(arr: Optional[jax.Array]) -> float:
    """Return memory usage of a JAX array in MB."""
    if arr is None:
        return 0.0
    return arr.nbytes / 1024 / 1024


def measure_state_memory(state) -> Dict[str, float]:
    """Measure memory usage of all arrays in a SinkhornStepState."""
    memory_map = {}
    memory_map["costs"] = measure_array_memory(state.costs)
    memory_map["X"] = measure_array_memory(state.X)
    memory_map["linear_convergence"] = measure_array_memory(state.linear_convergence)
    memory_map["displacement_sqnorms"] = measure_array_memory(state.displacement_sqnorms)
    memory_map["a"] = measure_array_memory(state.a)
    memory_map["rng"] = measure_array_memory(state.rng)

    if state.sinkhorn_errors is not None:
        memory_map["sinkhorn_errors"] = measure_array_memory(state.sinkhorn_errors)
    if state.objective_vals is not None:
        memory_map["objective_vals"] = measure_array_memory(state.objective_vals)
    if state.cosin_similarity is not None:
        memory_map["cosin_similarity"] = measure_array_memory(state.cosin_similarity)
    if state.X_history is not None:
        memory_map["X_history"] = measure_array_memory(state.X_history)

    memory_map["total"] = sum(memory_map.values())
    return memory_map


def check_numerical_equivalence(
    result1: jax.Array,
    result2: jax.Array,
    atol: float = 1e-5,
    rtol: float = 1e-6,
    name: str = "arrays"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if two arrays are numerically equivalent within tolerances.

    Returns:
        Tuple of (is_equivalent, metrics_dict)
    """
    if result1.shape != result2.shape:
        return False, {
            "error": "shape_mismatch",
            "shape1": result1.shape,
            "shape2": result2.shape
        }

    abs_diff = jnp.abs(result1 - result2)
    max_abs_diff = float(jnp.max(abs_diff))
    mean_abs_diff = float(jnp.mean(abs_diff))

    rel_diff = abs_diff / (jnp.abs(result1) + 1e-10)
    max_rel_diff = float(jnp.max(rel_diff))
    mean_rel_diff = float(jnp.mean(rel_diff))

    is_close = jnp.allclose(result1, result2, atol=atol, rtol=rtol)

    metrics = {
        "name": name,
        "is_equivalent": bool(is_close),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "tolerance_abs": atol,
        "tolerance_rel": rtol
    }

    return bool(is_close), metrics


def compare_solver_states(
    state1,
    state2,
    atol: float = 1e-5,
    rtol: float = 1e-6
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two solver states for numerical equivalence.

    Returns:
        Tuple of (all_equivalent, detailed_metrics)
    """
    results = {}
    all_equivalent = True

    fields_to_check = ["costs", "X", "linear_convergence", "displacement_sqnorms"]

    for field in fields_to_check:
        arr1 = getattr(state1, field)
        arr2 = getattr(state2, field)
        is_eq, metrics = check_numerical_equivalence(arr1, arr2, atol, rtol, field)
        results[field] = metrics
        all_equivalent = all_equivalent and is_eq

    optional_fields = ["sinkhorn_errors", "objective_vals", "cosin_similarity", "X_history"]
    for field in optional_fields:
        arr1 = getattr(state1, field)
        arr2 = getattr(state2, field)
        if arr1 is not None and arr2 is not None:
            is_eq, metrics = check_numerical_equivalence(arr1, arr2, atol, rtol, field)
            results[field] = metrics
            all_equivalent = all_equivalent and is_eq

    results["all_equivalent"] = all_equivalent
    return all_equivalent, results


def print_comparison_report(results: Dict[str, Any]):
    """Pretty print comparison results."""
    print("\n" + "="*60)
    print("Numerical Equivalence Report")
    print("="*60)

    all_eq = results.pop("all_equivalent", True)

    for field, metrics in results.items():
        if isinstance(metrics, dict) and "is_equivalent" in metrics:
            status = "✓" if metrics["is_equivalent"] else "✗"
            print(f"\n{status} {field}:")
            print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
            print(f"  Mean abs diff: {metrics['mean_abs_diff']:.2e}")
            print(f"  Max rel diff: {metrics['max_rel_diff']:.2e}")
            print(f"  Mean rel diff: {metrics['mean_rel_diff']:.2e}")

    print("\n" + "="*60)
    print(f"Overall: {'PASS' if all_eq else 'FAIL'}")
    print("="*60 + "\n")
