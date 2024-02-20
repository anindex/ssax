import os
import time

import hydra
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    'font.size': 24,
    "figure.figsize": (10, 8)
})
plt.style.use('ggplot')

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

    ent_eps = jnp.array([0.5, 0.1, 0.05, 0.01, 0.005])
    polytopes = ['simplex', 'orthoplex', 'cube']
    data = {}
    for polytope in polytopes:
        cosin_sim_list = []
        for i in range(ent_eps.shape[0]):
            current_cs_list = None
            for j in range(cfg.num_seeds):
                seed = int(time.time())
                rng = jax.random.PRNGKey(seed)

                # Epislon scheduler
                # epsilon_scheduler_cls = getattr(ssax.ss.epsilon_scheduler, cfg.experiment.epsilon_scheduler.name)
                # epsilon_scheduler = epsilon_scheduler_cls(**cfg.experiment.epsilon_scheduler.kwargs)

                # global optimizer  # TODO: make a base class for global optimizers
                global_optimizer_cls = getattr(ssax.ss.solver, cfg.experiment.optimizer.name)
                cfg.experiment.optimizer.kwargs.ent_epsilon = float(ent_eps[i])
                cfg.experiment.optimizer.kwargs.polytope_type = polytope
                global_optimizer = global_optimizer_cls.create(
                    objective_fn=task,
                    linear_ot_solver=linear_ot_solver,
                    epsilon=1.,  # unscheduled for benchmarking
                    store_cosine_similarity=True,
                    **cfg.experiment.optimizer.kwargs
                )

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
                if current_cs_list is not None:
                    for k, cs in enumerate(cosin_similarity):
                        current_cs_list[k] = jnp.concatenate([current_cs_list[k], cs[~np.isnan(cs)]])
                else:
                    current_cs_list = [cs[~jnp.isnan(cs)] for cs in cosin_similarity]
            cosin_sim_list.append(current_cs_list)
        data[polytope] = cosin_sim_list

    # Plotting
    if cfg.plotting:
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(35, 7))
        idx = 0
        for polytope, polytope_cs_list in data.items():
            for i, cs_list in enumerate(polytope_cs_list):
                mean_cs = np.array([cs.mean() for cs in cs_list])
                std_cs = np.array([cs.std() for cs in cs_list])
                ax[idx].plot(mean_cs, label=rf"λ={ent_eps[i]:.3f}")
                ax[idx].fill_between(
                    np.arange(mean_cs.shape[0]),
                    mean_cs - std_cs,
                    mean_cs + std_cs,
                    alpha=0.2
                )
            ax[idx].set_title(polytope)
            max_iters = cfg.experiment.optimizer.kwargs.max_iterations
            ax[idx].set_xticks(np.arange(0, max_iters, step=10))
            ax[idx].set_xticklabels(np.arange(0, max_iters // 10) * 10)
            idx += 1
        lines, labels = plt.gca().get_legend_handles_labels()
        axbox = ax[1].get_position()
        fig.legend(lines, labels, loc='lower center', ncol=5, bbox_to_anchor=[axbox.x0+0.5*axbox.width, axbox.y0-0.05])
        plt.ylim([-1., 1.])
        plt.grid(visible=True, alpha=0.5)
        fig.text(0.5, 0., rf"Iterations", ha='center')
        fig.text(0., 0.5, rf"Cosine similarity", va='center', rotation='vertical')
        fig.tight_layout()
        fig.savefig(os.path.join(logs_dir, f"cosin_sim_{cfg.experiment.name}_{cfg.experiment.num_points}.pdf"), format="pdf", bbox_inches="tight", dpi=300)
        # plt.show()
    # Save results
    if cfg.save_result:
        res_name = os.path.join(logs_dir, f"{cfg.experiment.name}_{cfg.experiment.num_points}.npz")
        np.savez(res_name, **res)


if __name__ == "__main__":
    main()
