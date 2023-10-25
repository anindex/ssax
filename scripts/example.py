import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='html5')
import time
import os

from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.geometry.epsilon_scheduler import Epsilon

from ssax.ss.initializer import UniformInitializer
from ssax.ss.solver import SinkhornStep
from ssax.ss.epsilon_scheduler import LinearEpsilon
from ssax.objectives.synthetic import Ackley
from ssax.objectives.visualization import plot_objective


if __name__ == '__main__':

    rng = jax.random.PRNGKey(0)
    num_points = 100000

    # cost function
    objective_fn = Ackley.create(
        dim=2
    )

    # initialize points
    initializer = UniformInitializer.create(
        rng=rng,
        bounds=objective_fn.bounds.tolist()
    )
    X_init = initializer(num_points)

    # linear solver
    linear_solver = Sinkhorn(
        threshold=1e-3,
        inner_iterations=1,
        min_iterations=1,
        max_iterations=100,
        initializer='default'
    )
    # linear_solver = LRSinkhorn(
    #     rank=2,
    #     threshold=1e-3,
    #     inner_iterations=1,
    #     min_iterations=1,
    #     max_iterations=100,
    #     initializer='default'
    # )

    # epsilon scheduler
    epsilon = LinearEpsilon(
        target=0.3,
        init=1.,
        decay=0.01,
    )

    # Sinkhorn Step solver
    sinkhorn_step = SinkhornStep.create(
        objective_fn=objective_fn,
        linear_ot_solver=linear_solver,
        epsilon=epsilon,
        polytope_type='orthoplex',
        step_radius=0.15,
        probe_radius=0.2,
        num_probe=5,
        min_iterations=5,
        max_iterations=100,
        store_history=True,
        threshold=1e-3
    )

    # warm start for compiling the step function
    tic = time.time()
    init_state = sinkhorn_step.warm_start(X_init, rng=rng)
    toc = time.time()
    print(f"Warm up JIT compile time: {toc - tic:.2f} s")

    # optimization
    tic = time.time()
    res = sinkhorn_step.iterations(init_state)
    toc = time.time()
    print(f"Optimization time: {toc - tic:.2f} s")

    # NOTE: this is a hack getting converged iteration index
    cid = jnp.where(res.linear_convergence == -1)[0]
    if len(cid) == 0:
        converged_at = res.linear_convergence.shape[0]
    else:
        converged_at = cid.min()

    # Plotting
    if objective_fn.bounds is None:
        bounds = [[-10.0, 10.0], [-10.0, 10.0]]
    else:
        bounds = objective_fn.bounds
    fig, ax = plt.subplots()
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect('equal')
    ax.axis('off')
    plot_objective(objective_fn=objective_fn, ax=ax)
    text = ax.text(bounds[0][1], bounds[1][1] + 0.5, f'Iters {0}', style='italic')
    scatter = ax.scatter(X_init[:, 0], X_init[:, 1], s=3)
    fig.tight_layout()

    print("Rendering GIF...")
    X_hist = res.X_history
    def plot_particle(iter):
        """ Plots the objective function in 2D """
        X = X_hist[iter]
        scatter.set_offsets(X[:, :2])
        text.set_text(f'Iters {iter}')
        return scatter, text
    anim = animation.FuncAnimation(fig, plot_particle, frames=converged_at, interval=100, blit=True)
    anim_name = os.path.join('.', f"ackley_{num_points}.gif")
    anim.save(anim_name, writer='imagemagick', fps=20)
