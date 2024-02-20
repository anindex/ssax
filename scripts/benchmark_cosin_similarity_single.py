import os
import time

import hydra
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    'font.size': 16,
    "figure.figsize": (10, 8)
})

from omegaconf import DictConfig, OmegaConf
from hydra.conf import HydraConf
import jax
import jax.numpy as jnp
from jax import jit

from ssax.objectives.visualization import plot_objective
import ssax.objectives
from ssax.objectives import ObjectiveFn
import ssax.ss.initializer
import ssax.ss.linear_solver
import ssax.ss.solver
import ssax.ss.epsilon_scheduler

PLOT = 'box'  # 'box' or 'violin' or 'mean-std'

def set_violin_properties(r):
    for item in r.items():
        if item[0] == "bodies":
            for vc in item[1]:
                vc.set_facecolor("gray")
                vc.set_edgecolor("black")
                vc.set_alpha(0.5)
        else:
            item[1].set_color("gray")
            item[1].set_lw(2.0)
    r['cmeans'].set_color('green')
    r['cmeans'].set_linewidth(3.5)
    r['cmedians'].set_color('red')
    r['cmedians'].set_linewidth(3.5)


def set_boxplot_properties(r):
    for median in r['medians']:
        median.set_color('red')
        median.set_linewidth(3.5)
    for w in r['whiskers']:
        w.set_color('gray')
        w.set_linewidth(1.)


@hydra.main(version_base=None, config_path="../configs", config_name="config-cosin-sim")
def main(cfg: DictConfig):
    # Set data directory
    data_dir = cfg.data_dir
    if data_dir is None:
        data_dir = os.path.join(ssax.DATA_DIR, cfg.experiment.name)

    # Set log directory
    logs_dir = cfg.logs_dir
    if logs_dir is None:
        logs_dir = os.path.join(ssax.LOGS_DIR, cfg.experiment.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logs_dir, exist_ok=True)

    if cfg.gpu:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

    # Load task objective function
    task_cls = getattr(ssax.objectives, cfg.experiment.task.name)
    task: ObjectiveFn = task_cls.create(**cfg.experiment.task.kwargs)

    # Linear solver
    solver_cls = getattr(ssax.ss.linear_solver, cfg.experiment.linear_solver.name)
    linear_ot_solver = solver_cls(**cfg.experiment.linear_solver.kwargs)

    # Epislon scheduler
    # epsilon_scheduler_cls = getattr(ssax.ss.epsilon_scheduler, cfg.experiment.epsilon_scheduler.name)
    # epsilon_scheduler = epsilon_scheduler_cls(**cfg.experiment.epsilon_scheduler.kwargs)

    # global optimizer  # TODO: make a base class for global optimizers
    global_optimizer_cls = getattr(ssax.ss.solver, cfg.experiment.optimizer.name)
    global_optimizer = global_optimizer_cls.create(
        objective_fn=task,
        linear_ot_solver=linear_ot_solver,
        epsilon=1.,  # unscheduled for benchmarking
        store_cosine_similarity=True,
        **cfg.experiment.optimizer.kwargs
    )

    cosin_sim_list = None
    for i in range(cfg.num_seeds):
        seed = int(time.time())
        rng = jax.random.PRNGKey(seed)
        # Sample initializer
        init_cls = getattr(ssax.ss.initializer, cfg.experiment.initializer.name)
        if task.bounds is not None:
            cfg.experiment.initializer.kwargs.bounds = task.bounds.tolist()
        initializer = init_cls.create(rng=rng, **cfg.experiment.initializer.kwargs)

        X_init = initializer(cfg.experiment.num_points)
        init_state = global_optimizer.warm_start(X_init, rng=rng)

        tic = time.time()
        res = global_optimizer.iterations(init_state)
        toc = time.time()

        # NOTE: this is a hack getting converged iteration index
        cid = jnp.where(res.linear_convergence == -1)[0]
        if len(cid) == 0:
            converged_at = res.linear_convergence.shape[0]
        else:
            converged_at = cid.min()
        print(f"Optimization converged at {converged_at}! Time: {toc - tic:.2f}s")
        cosin_similarity = res.cosin_similarity[:converged_at]
        # remove NaN values from fix points convergences
        if cosin_sim_list is not None:
            for k, cs in enumerate(cosin_similarity):
                cosin_sim_list[k] = jnp.concatenate([cosin_sim_list[k], cs[~np.isnan(cs)]])
        else:
            cosin_sim_list = [
                cs[~jnp.isnan(cs)] for cs in cosin_similarity
            ]
        # cosin_similarity = cosin_similarity[~np.isnan(cosin_similarity)]
        # print(f'Gradient cosine similarity: {cosin_similarity.mean():.4f} +- {cosin_similarity.std():.4f}')

    # Plotting
    if cfg.plotting:
        max_iters = cfg.experiment.optimizer.kwargs.max_iterations
        fig, ax = plt.subplots(1, 1)
        if PLOT == 'violin':
            r = ax.violinplot(cosin_sim_list, showmeans=True, showmedians=True, showextrema=True)
            set_violin_properties(r)
        elif PLOT == 'box':
            r = ax.boxplot(cosin_sim_list, showmeans=True, showfliers=False)
            set_boxplot_properties(r)
        else:
            mean_cs = np.array([cs.mean() for cs in cosin_sim_list])
            std_cs = np.array([cs.std() for cs in cosin_sim_list])
            ax.plot(mean_cs, color='gray', linestyle='--')
            ax.fill_between(
                np.arange(mean_cs.shape[0]),
                mean_cs - std_cs,
                mean_cs + std_cs,
                alpha=0.2,
                color='gray',
            )
        ax.grid(visible=True, alpha=0.5)
        ax.set_xticks(np.arange(0, max_iters, step=10))
        ax.set_xticklabels(np.arange(0, max_iters // 10) * 10)
        ax.set_xlabel(rf"Iterations")
        ax.set_ylabel(rf"Cosine similarity")
        fig.tight_layout()
        fig.savefig(os.path.join(logs_dir, f"cosin_sim_{cfg.experiment.name}_{cfg.experiment.num_points}.pdf"), format="pdf", bbox_inches="tight", dpi=300)
    # Save results
    if cfg.save_result:
        res_name = os.path.join(logs_dir, f"{cfg.experiment.name}_{cfg.experiment.num_points}.npz")
        np.savez(res_name, **res)


if __name__ == "__main__":
    main()
