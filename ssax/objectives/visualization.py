import numpy as np
import matplotlib.pyplot as plt



def plot_objective(objective_fn, ax=None):
    """ Plots the objective function in 2D """
    if ax is None:
        ax = plt.gca()
    bounds = objective_fn.bounds
    if bounds is None:
        bounds = [[-5, 5], [-5, 5]]
    x = np.linspace(*bounds[0], 5, 100)
    y = np.linspace(*bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(np.stack([X, Y], axis=-1))
    ax.contourf(X, Y, Z, 100)
    ax.set_aspect('equal')
    # plt.colorbar()
