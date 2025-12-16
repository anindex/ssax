import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from omegaconf import DictConfig
from jax import random

# OTT Imports
from ott.solvers.linear import sinkhorn, sinkhorn_lr

# Internal SSAX Imports
import ssax.objectives.synthetic as synthetic_objectives
from ssax.objectives.base import ObjectiveFn
from ssax.objectives.visualization import plot_contour
import ssax.ss.initializer as initializers
import ssax.ss.solver as solvers
import ssax.ss.epsilon_scheduler as schedulers

# Configure Matplotlib animation
rc('animation', html='html5')

# Setup logger to print to stdout
log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
log.addHandler(handler)
log.setLevel(logging.INFO)

def setup_linear_solver(cfg: DictConfig):
    """Factory for the inner OT solver (Sinkhorn or LRSinkhorn)."""
    solver_name = cfg.experiment.linear_solver.name
    solver_kwargs = cfg.experiment.linear_solver.kwargs
    
    if solver_name == 'Sinkhorn':
        return sinkhorn.Sinkhorn(**solver_kwargs)
    elif solver_name == 'LRSinkhorn':
        return sinkhorn_lr.LRSinkhorn(**solver_kwargs)
    else:
        raise ValueError(f"Unknown linear solver: {solver_name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("="*50)
    print(f"Starting Experiment: {cfg.experiment.name}")
    print("="*50)

    # 1. Setup Environment
    cwd = Path.cwd()
    print(f"Working Directory: {cwd}")
    
    seed = cfg.experiment.seed
    rng = random.PRNGKey(seed)
    rng, init_rng, warm_rng = random.split(rng, 3)

    # 2. Load Task (Objective Function)
    task_cls = getattr(synthetic_objectives, cfg.experiment.task.name)
    task: ObjectiveFn = task_cls.create(**cfg.experiment.task.kwargs)
    print(f"Task Loaded: {cfg.experiment.task.name} (Dim: {task.dim})")

    # 3. Initialize Particles
    init_cfg = cfg.experiment.initializer
    # Inject bounds if available
    if task.bounds is not None:
        bounds_list = np.array(task.bounds).tolist() 
        init_cfg.kwargs.bounds = bounds_list

    init_cls = getattr(initializers, init_cfg.name)
    initializer = init_cls.create(rng=init_rng, **init_cfg.kwargs)
    X_init = initializer(cfg.experiment.num_points)
    print(f"Initialized {cfg.experiment.num_points} particles using {init_cfg.name}.")

    # 4. Setup Solvers & Scheduler
    linear_ot_solver = setup_linear_solver(cfg)
    
    scheduler_cls = getattr(schedulers, cfg.experiment.epsilon_scheduler.name)
    epsilon_scheduler = scheduler_cls(**cfg.experiment.epsilon_scheduler.kwargs)

    optimizer_cls = getattr(solvers, cfg.experiment.optimizer.name)
    global_optimizer = optimizer_cls.create(
        objective_fn=task,
        linear_ot_solver=linear_ot_solver,
        epsilon=epsilon_scheduler,
        store_outer_evals=True,
        store_history=True,
        **cfg.experiment.optimizer.kwargs
    )

    # 5. Warm Start (JIT Compilation)
    print("\n[1/3] Compiling JAX functions (Warm Start)...")
    tic = time.perf_counter()  # Changed to perf_counter
    init_state = global_optimizer.warm_start(X_init, rng=warm_rng)
    init_state.costs.block_until_ready() 
    toc = time.perf_counter()
    print(f"      Compilation finished in {toc - tic:.4f} s")

    # 6. Main Optimization Loop
    print("\n[2/3] Running Optimization Loop...")
    tic = time.perf_counter()
    res = global_optimizer.iterations(init_state)
    res.costs.block_until_ready()
    toc = time.perf_counter()

    # Convergence check
    valid_iters = jnp.sum(res.costs != -1.0)
    converged_at = int(valid_iters)
    print(f"      Optimization finished in {toc - tic:.2f}s.")
    print(f"      Total Iterations: {converged_at}")

    # 7. Visualization & Saving
    print("\n[3/3] Generating Outputs...")
    if cfg.plotting or cfg.save_video:
        X_history = np.array(res.X_history)  # [iters, N, dim]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_contour(task, ax=ax, title=f"{cfg.experiment.name} (Iter 0)")
        
        scatter = ax.scatter(
            X_init[:, 0], X_init[:, 1], 
            c='white', s=20, edgecolors='black', alpha=0.9, label='Particles', zorder=5
        )
        text = ax.text(0.05, 0.95, f'Iter: 0', transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(loc='upper right')
        fig.tight_layout()

        if cfg.save_video:
            print("      Rendering GIF animation...")
            def update(frame):
                idx = min(frame, converged_at - 1)
                data = X_history[idx]
                scatter.set_offsets(data[:, :2])
                text.set_text(f'Iter: {idx}')
                ax.set_title(f"{cfg.experiment.name} (Iter {idx})")
                return scatter, text

            anim = animation.FuncAnimation(
                fig, update, frames=range(0, converged_at, 1), 
                interval=100, blit=False 
            )
            
            anim_path = cwd / f"{cfg.experiment.name}_animation.gif"
            try:
                anim.save(str(anim_path), writer='imagemagick', fps=15)
                print(f"      -> Saved GIF to: {anim_path}")
            except Exception as e:
                print(f"      -> Warning: Could not save GIF (is ImageMagick installed?): {e}")

        # B. Save Static Plots
        if cfg.plotting:
            # 1. Particles at convergence
            final_idx = max(0, converged_at - 1)
            scatter.set_offsets(X_history[final_idx, :, :2])
            text.set_text(f'Iter: {final_idx}')
            ax.set_title(f"{cfg.experiment.name} (Final)")
            
            fig_path = cwd / f"{cfg.experiment.name}_final.png"
            fig.savefig(str(fig_path), dpi=300)
            print(f"      -> Saved final plot to: {fig_path}")

            # 2. Objective Value Curve
            plt.figure(figsize=(8, 4))
            obj_vals = np.array(res.objective_vals[:converged_at]) 
            mean_obj = np.mean(obj_vals, axis=1)
            std_obj = np.std(obj_vals, axis=1)
            iters_arr = np.arange(converged_at)
            
            plt.plot(iters_arr, mean_obj, label='Mean Cost')
            plt.fill_between(iters_arr, mean_obj - std_obj, mean_obj + std_obj, alpha=0.3)
            plt.yscale('symlog')
            plt.xlabel('Iterations')
            plt.ylabel('Objective Value')
            plt.title(f"{cfg.experiment.name} Convergence")
            plt.grid(True, linestyle='--', alpha=0.5)
            
            curve_path = cwd / f"{cfg.experiment.name}_convergence.png"
            plt.savefig(str(curve_path), dpi=300)
            print(f"      -> Saved convergence curve to: {curve_path}")

    # 8. Save Raw Results
    if cfg.save_result:
        save_dict = {}
        for field in res.__dataclass_fields__:
            val = getattr(res, field)
            if isinstance(val, (jax.Array, np.ndarray)):
                save_dict[field] = np.array(val)
        
        npz_path = cwd / f"{cfg.experiment.name}_results.npz"
        np.savez(str(npz_path), **save_dict)
        print(f"      -> Saved raw results to: {npz_path}")

    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == "__main__":
    main()