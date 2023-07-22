import jax
import jax.numpy as jnp
from jax import jit

import numpy as np
import matplotlib.pyplot as plt
import time

from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.geometry.epsilon_scheduler import Epsilon

from ssax.ss.initializer import SSGaussianInitializer, SSUniformInitializer
from ssax.ss.costs import GenericCost
from ssax.ss.solver import SinkhornStep
from ssax.ss.epsilon_scheduler import LinearEpsilon

from ssax.objectives.visualization import plot_objective
from ssax.objectives.synthetic import Ackley, Beale


if __name__ == '__main__':
    # plt.figure()
    # plot_objective(Ackley())
    # rng = jax.random.PRNGKey(0)
    # num_points = 100
    # # initializer = SSGaussianInitializer(
    # #     jnp.array([0.0, 0.0]),
    # #     jnp.eye(2),
    # #     rng=rng
    # # )
    # initializer = SSUniformInitializer(
    #     jnp.array([[-5.0, 5.0], [-5.0, 5.0]]),
    #     rng=rng
    # )
    # X = initializer(num_points)
    # plt.scatter(X[:, 0], X[:, 1], c='r')
    # plt.show()

    rng = jax.random.PRNGKey(0)
    num_points = 100000
    # sinkhorn solver
    solver = Sinkhorn(
        threshold=1e-3,
        inner_iterations=1,
        min_iterations=1,
        max_iterations=100,
        initializer='default'
    )

    # cost function
    objective_fn = Ackley()
    
    # initialize points
    initializer = SSUniformInitializer(
        jnp.array([[-5.0, 5.0], [-5.0, 5.0]]),
        rng=rng
    )
    X = initializer(num_points)

    # Sinkhorn Step solver
    epsilon = LinearEpsilon(
        target=0.3,
        init=1.,
        decay=0.01,
    )
    sinkhorn_step = SinkhornStep(
        objective_fn=objective_fn,
        linear_ot_solver=solver,
        epsilon=epsilon,
        polytope_type='orthoplex',
        step_radius=0.15,
        probe_radius=0.2,
        num_probe=5,
        min_iterations=5,
        max_iterations=100,
        threshold=1e-3,
        rng=rng
    )

    # run Sinkhorn Step
    state = sinkhorn_step.init_state(X)
    plt.figure()
    ax = plt.gca()
    for i in range(100):
        plt.clf()
        tic = time.time()
        state = sinkhorn_step.step(state, i)
        toc = time.time()
        print(f'Iteration {i}, time: {toc - tic}')
        plot_objective(objective_fn, ax=ax)
        X = state.X
        plt.scatter(X[:, 0], X[:, 1], c='r', s=3)
        ax.set_aspect('equal')
        plt.draw()
        plt.pause(1e-4)

    # tic = time.time()
    # state = sinkhorn_step.iterations(X)
    # toc = time.time()
    # print(f'Time: {toc - tic}')
    # plt.figure()
    # plot_objective(objective_fn)
    # X = state.X
    # plt.scatter(X[:, 0], X[:, 1], c='r', s=3)
    # plt.show()
