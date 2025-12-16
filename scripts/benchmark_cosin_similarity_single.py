import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

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

# Visualization settings
PLOT_TYPE = 'box'  # Options: 'box', 'violin', 'mean-std'

plt.rcParams.update({
    "font.family": "serif",
    'font.size': 16,
    "figure.figsize": (12, 8),
    "axes.grid": True,
    "grid.alpha": 0.3
})


def setup_environment(cfg: DictConfig):
    if cfg.get("gpu", False):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        print("Running on GPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        print("Running on CPU (Simulated)")


def set_violin_properties(parts):
    """Styles matplotlib violin plots."""
    for item in parts.items():
        if item[0] == "bodies":
            for vc in item[1]:
                vc.set_facecolor("gray")
                vc.set_edgecolor("black")
                vc.set_alpha(0.5)
        else:
            item[1].set_color("gray")
            item[1].set_linewidth(1.5)
    
    if 'cmeans' in parts:
        parts['cmeans'].set_color('green')
        parts['cmeans'].set_linewidth(2.0)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2.0)


def set_boxplot_properties(bp):
    """Styles matplotlib box plots."""
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2.0)
    for box in bp['boxes']:
        box.set_color('black')
    for whisker in bp['whiskers']:
        whisker.set_color('gray')
        whisker.set_linestyle('--')
    for cap in bp['caps']:
        cap.set_color('gray')


def run_single_seed(cfg: DictConfig, task: ObjectiveFn, solver: solvers.SinkhornStep, seed: int) -> Tuple[np.ndarray, float]:
    """Runs one optimization run and returns cosine similarity history."""
    
    rng = random.PRNGKey(seed)
    rng, init_rng, warm_rng = random.split(rng, 3)

    # 1. Initialize Particles
    init_cfg = cfg.experiment.initializer
    init_kwargs = dict(init_cfg.kwargs)
    if task.bounds is not None:
        init_kwargs['bounds'] = np.array(task.bounds).tolist()

    init_cls = getattr(initializers, init_cfg.name)
    initializer = init_cls.create(rng=init_rng, **init_kwargs)
    X_init = initializer(cfg.experiment.num_points)

    # 2. Warm Start
    init_state = solver.warm_start(X_init, rng=warm_rng)

    # 3. Execution
    tic = time.perf_counter()
    res = solver.iterations(init_state)
    res.costs.block_until_ready()
    toc = time.perf_counter()

    # 4. Extract Data
    # Identify valid iterations
    valid_iters = int(jnp.sum(res.costs != -1.0))
    
    # [valid_iters, num_points]
    cs_history = res.cosin_similarity[:valid_iters]
    
    # Return as numpy array for easier aggregation later
    return np.array(cs_history), (toc - tic)


@hydra.main(version_base=None, config_path="../configs", config_name="config-cosin-sim")
def main(cfg: DictConfig):
    setup_environment(cfg)
    
    print("="*60)
    print(f"BENCHMARK SINGLE: {cfg.experiment.name}")
    print(f"Task: {cfg.experiment.task.name}")
    print("="*60)

    # 1. Setup Task & Solver
    task_cls = getattr(synthetic_objectives, cfg.experiment.task.name)
    task: ObjectiveFn = task_cls.create(**cfg.experiment.task.kwargs)

    linear_solver_name = cfg.experiment.linear_solver.name
    linear_solver_kwargs = cfg.experiment.linear_solver.kwargs
    if linear_solver_name == 'Sinkhorn':
        linear_ot_solver = sinkhorn.Sinkhorn(**linear_solver_kwargs)
    else:
        linear_ot_solver = sinkhorn_lr.LRSinkhorn(**linear_solver_kwargs)

    optimizer_cls = getattr(solvers, cfg.experiment.optimizer.name)
    global_optimizer = optimizer_cls.create(
        objective_fn=task,
        linear_ot_solver=linear_ot_solver,
        epsilon=1.0,  # Unscheduled for benchmark
        store_cosine_similarity=True,
        **cfg.experiment.optimizer.kwargs
    )

    # 2. Run Seeds and Aggregate
    base_seed = cfg.experiment.get("seed", cfg.get("seed", 0))
    max_iters = cfg.experiment.optimizer.kwargs.max_iterations
    
    # Storage: list of arrays. bucket[i] contains all cosine sim values for iteration i across all seeds.
    aggregated_data = [[] for _ in range(max_iters)]
    
    durations = []

    for i in range(cfg.num_seeds):
        current_seed = base_seed + i
        print(f"Running Seed {i+1}/{cfg.num_seeds}...", end=" ", flush=True)
        
        # Run
        cs_hist, duration = run_single_seed(cfg, task, global_optimizer, current_seed)
        durations.append(duration)
        
        # Aggregate
        run_iters = cs_hist.shape[0]
        for t in range(run_iters):
            vals = cs_hist[t]
            valid_vals = vals[~np.isnan(vals)]
            aggregated_data[t].append(valid_vals)
            
        print(f"Done ({duration:.2f}s, {run_iters} iters)")

    print("-" * 60)
    print(f"Avg Duration: {np.mean(durations):.4f}s")

    # Flatten the buckets: List[List[arrays]] -> List[1D array]
    # plot_data[i] = array of all particle similarities at iter i
    plot_data = []
    valid_plot_iters = 0
    for bucket in aggregated_data:
        if len(bucket) > 0:
            plot_data.append(np.concatenate(bucket))
            valid_plot_iters += 1
        else:
            # If no data for this iteration (all runs finished), stop
            break

    # 3. Plotting
    if cfg.plotting:
        print(f"Generating {PLOT_TYPE} plot...")
        logs_dir = Path(cfg.logs_dir) if cfg.logs_dir else Path.cwd() / "logs" / cfg.experiment.name / time.strftime("%Y%m%d-%H%M%S")
        logs_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        
        # X-axis indices
        iters = np.arange(valid_plot_iters)
        
        if PLOT_TYPE == 'violin':
            parts = ax.violinplot(plot_data, positions=iters, showmeans=True, showmedians=True)
            set_violin_properties(parts)
            
        elif PLOT_TYPE == 'box':
            bp = ax.boxplot(plot_data, positions=iters, showmeans=True, showfliers=False, patch_artist=False)
            set_boxplot_properties(bp)
            
        elif PLOT_TYPE == 'mean-std':
            means = np.array([np.mean(d) for d in plot_data])
            stds = np.array([np.std(d) for d in plot_data])
            ax.plot(iters, means, color='blue', linestyle='-', linewidth=2, label='Mean')
            ax.fill_between(iters, means - stds, means + stds, alpha=0.2, color='blue', label='Std Dev')
            ax.legend()

        # Formatting
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cosine Similarity (Gradient vs Update)")
        ax.set_title(f"Convergence Quality: {cfg.experiment.name}")
        
        # Optimize X-ticks if too many iterations
        if valid_plot_iters > 20:
            step = valid_plot_iters // 10
            ax.set_xticks(np.arange(0, valid_plot_iters, step))
        
        ax.set_ylim(-1.1, 1.1)
        fig.tight_layout()
        
        plot_path = logs_dir / f"cosin_sim_single_{cfg.experiment.name}.pdf"
        fig.savefig(str(plot_path), format="pdf", bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {plot_path}")

    # 4. Save Results
    if cfg.save_result:
        res_name = logs_dir / f"{cfg.experiment.name}_{cfg.experiment.num_points}_single.npz"
        np.savez(str(res_name), plot_data=np.array(plot_data, dtype=object))
        print(f"Results saved to: {res_name}")

    print("Done.")

if __name__ == "__main__":
    main()
