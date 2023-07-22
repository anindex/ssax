import numpy as np
import matplotlib.pyplot as plt



def plot_objective(objective_fn, ax=None):
    """ Plots the objective function in 2D """
    if ax is None:
        ax = plt.gca()
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(np.stack([X, Y], axis=-1))
    plt.contourf(X, Y, Z, 100)
    # plt.colorbar()
