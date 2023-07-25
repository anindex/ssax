import numpy as np
import matplotlib.pyplot as plt



def plot_objective(objective_fn, ax=None, resolution=100):
    """ Plots the objective function in 2D """
    if ax is None:
        ax = plt.gca()
    bounds = objective_fn.bounds
    if bounds is None:
        bounds = [[-5, 5], [-5, 5]]
    x = np.linspace(*bounds[0], resolution)
    y = np.linspace(*bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(np.stack([X, Y], axis=-1))
    ax.contourf(X, Y, Z, resolution)
    ax.set_aspect('equal')
    # plt.colorbar()
