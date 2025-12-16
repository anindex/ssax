import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from jax import random

# OTT Imports
from ott.solvers.linear import sinkhorn, sinkhorn_lr

# Internal SSAX Imports
import ssax.objectives.synthetic as synthetic_objectives
from ssax.objectives.base import ObjectiveFn
import ssax.ss.initializer as initializers
import ssax.ss.solver as solvers

# Matplotlib Styling
plt.rcParams.update({
    "font.family": "serif",
    'font.size': 18,
    "figure.figsize": (18, 6),
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.alpha": 0.3
})
plt.style.use('ggplot')


def setup_environment(cfg: DictConfig):
    """Configures XLA/GPU settings based on config."""
    if cfg.get("gpu", False):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        print("Running on GPU (XLA Preallocate: False)")
    else:
        # Force CPU if GPU not requested
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        print("Running on CPU (Simulated Devices: 4)")


def get_solver_and_task(cfg: DictConfig, polytope_type: str, ent_epsilon: float) -> Tuple[ObjectiveFn, solvers.SinkhornStep]:
    """Instantiates the task and SinkhornStep solver with benchmark overrides."""
    
    # 1. Task
    task_cls = getattr(synthetic_objectives, cfg.experiment.task.name)
    task: ObjectiveFn = task_cls.create(**cfg.experiment.task.kwargs)

    # 2. Linear OT Solver
    linear_solver_name = cfg.experiment.linear_solver.name
    linear_solver_kwargs = cfg.experiment.linear_solver.kwargs
    
    if linear_solver_name == 'Sinkhorn':
        linear_ot_solver = sinkhorn.Sinkhorn(**linear_solver_kwargs)
    elif linear_solver_name == 'LRSinkhorn':
        linear_ot_solver = sinkhorn_lr.LRSinkhorn(**linear_solver_kwargs)
    else:
        raise ValueError(f"Unknown linear solver: {linear_solver_name}")

    # 3. Sinkhorn Step Solver
    optimizer_cls = getattr(solvers, cfg.experiment.optimizer.name)
    
    # Create kwargs copy to avoid mutating config
    opt_kwargs = dict(cfg.experiment.optimizer.kwargs)
    
    # Override benchmark-specific parameters
    opt_kwargs['polytope_type'] = polytope_type
    opt_kwargs['ent_epsilon'] = ent_epsilon
    
    solver = optimizer_cls.create(
        objective_fn=task,
        linear_ot_solver=linear_ot_solver,
        epsilon=1.0,  # Unscheduled fixed epsilon for step radius scaling
        store_cosine_similarity=True,
        store_history=False, # Disable history to save memory during benchmark
        **opt_kwargs
    )
    
    return task, solver


def run_benchmark(cfg: DictConfig, task: ObjectiveFn, solver: solvers.SinkhornStep, seed: int) -> Tuple[np.ndarray, float]:
    """Runs a single benchmark iteration and returns cosine similarity history."""
    
    rng = random.PRNGKey(seed)
    rng, init_rng, warm_rng = random.split(rng, 3)

    # 1. Initialize Particles
    init_cfg = cfg.experiment.initializer
    
    # Handle bounds injection
    init_kwargs = dict(init_cfg.kwargs)
    if task.bounds is not None:
         init_kwargs['bounds'] = np.array(task.bounds).tolist()

    init_cls = getattr(initializers, init_cfg.name)
    initializer = init_cls.create(rng=init_rng, **init_kwargs)
    X_init = initializer(cfg.experiment.num_points)

    # 2. Warm Start
    # Compile the step function
    init_state = solver.warm_start(X_init, rng=warm_rng)
    
    # 3. Execution
    tic = time.perf_counter()
    res = solver.iterations(init_state)
    res.costs.block_until_ready()
    toc = time.perf_counter()
    
    # 4. Extract Data
    valid_iters = int(jnp.sum(res.costs != -1.0))
    
    # Extract cosine similarity [iter, N] up to convergence
    cs_history = res.cosin_similarity[:valid_iters]
    
    # Convert to numpy for aggregation
    return np.array(cs_history), (toc - tic)


@hydra.main(version_base=None, config_path="../configs", config_name="config-cosin-sim")
def main(cfg: DictConfig):
    setup_environment(cfg)
    
    print("="*60)
    print(f"BENCHMARK: {cfg.experiment.name}")
    print(f"Task: {cfg.experiment.task.name} | Points: {cfg.experiment.num_points}")
    print("="*60)

    # Benchmark Configuration
    ent_eps_list = [0.5, 0.1, 0.05, 0.01, 0.005]
    polytopes = ['simplex', 'orthoplex', 'cube']
    
    # Safe retrieval of base seed (try experiment.seed -> seed -> default 0)
    base_seed = cfg.experiment.get("seed", cfg.get("seed", 0))
    
    # Initialize Storage
    # Structure: results[polytope][eps_idx] = List[run_arrays]
    results = {p: [[] for _ in ent_eps_list] for p in polytopes}
    
    # --- Main Loops ---
    for p_idx, polytope in enumerate(polytopes):
        print(f"Processing Polytope: {polytope.upper()}")
        
        for e_idx, ent_eps in enumerate(ent_eps_list):
            print(f"  > Entropic Epsilon: {ent_eps}", end="... ", flush=True)
            
            # Setup Solver
            task, solver = get_solver_and_task(cfg, polytope, ent_eps)
            run_times = []
            
            # Run Seeds
            for seed_idx in range(cfg.num_seeds):
                # Deterministic seed generation
                current_seed = base_seed + seed_idx
                
                # Execute
                cs_data, duration = run_benchmark(cfg, task, solver, current_seed)
                run_times.append(duration)
                
                # Compute Mean Cosine Similarity across particles for this run
                # cs_data shape: [iters, N] -> [iters]
                cs_mean_trace = np.nanmean(cs_data, axis=-1)
                results[polytope][e_idx].append(cs_mean_trace)
            
            avg_time = np.mean(run_times)
            print(f"Done. Avg Time ({cfg.num_seeds} seeds): {avg_time:.4f}s")

    # --- Plotting ---
    if cfg.plotting:
        print("\nGenerating Plots...")
        
        # Determine paths
        logs_dir = Path(cfg.logs_dir) if cfg.logs_dir else Path.cwd() / "logs" / cfg.experiment.name / time.strftime("%Y%m%d-%H%M%S")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(24, 7))
        
        for idx, polytope in enumerate(polytopes):
            ax = axes[idx]
            ax.set_title(polytope.capitalize(), fontsize=22, pad=15)
            
            for e_idx, ent_eps in enumerate(ent_eps_list):
                seed_runs = results[polytope][e_idx]
                if not seed_runs:
                    continue
                
                # Align runs of different lengths (convergence times)
                max_len = max(len(r) for r in seed_runs)
                
                # Prepare matrix for stats
                padded_data = np.full((len(seed_runs), max_len), np.nan)
                
                for s_i, run in enumerate(seed_runs):
                    run_len = len(run)
                    padded_data[s_i, :run_len] = run
                    # Forward-fill the last value to simulate "steady state" after convergence
                    if run_len < max_len:
                        padded_data[s_i, run_len:] = run[-1]

                # Compute Stats
                mean_curve = np.nanmean(padded_data, axis=0)
                std_curve = np.nanstd(padded_data, axis=0)
                iters = np.arange(max_len)
                
                # Plot
                label = rf"$\lambda={ent_eps}$"
                ax.plot(iters, mean_curve, label=label)
                ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
            
            ax.set_xlabel("Iterations", fontsize=20)
            if idx == 0:
                ax.set_ylabel("Cosine Similarity", fontsize=20)
            
            ax.set_ylim(-0.2, 1.05)
            ax.tick_params(axis='both', which='major', labelsize=16)

        # Unified Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(ent_eps_list), 
                   bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=20)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15) # Make room for legend
        
        # Save
        plot_path = logs_dir / f"benchmark_cosin_{cfg.experiment.name}.pdf"
        plt.savefig(str(plot_path), bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {plot_path}")

    # --- Saving Raw Data ---
    if cfg.save_result:
        res_name = logs_dir / f"{cfg.experiment.name}_{cfg.experiment.num_points}_benchmark.npz"
        # Flatten dictionary for saving
        save_dict = {}
        for p in polytopes:
            for e_idx, eps in enumerate(ent_eps_list):
                key = f"{p}_eps_{eps}"
                # Save as object array of arrays (ragged)
                save_dict[key] = np.array(results[p][e_idx], dtype=object)
        
        np.savez(str(res_name), **save_dict)
        print(f"Raw results saved to: {res_name}")

    print("Done.")

if __name__ == "__main__":
    main()
