import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Any, Union

from .base import ObjectiveFn

__all__ = ["plot_contour", "plot_surface"]


def _prepare_grid(
    objective_fn: ObjectiveFn, 
    bounds: Optional[Union[List, np.ndarray]], 
    resolution: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper to generate meshgrid and evaluate Z values."""
    
    # 1. Determine Bounds (Use objective bounds or defaults)
    if bounds is None:
        if objective_fn.bounds is not None:
            # Convert JAX array to Numpy for plotting logic
            b = np.array(objective_fn.bounds)
            # Take bounds for the first two dimensions
            xlim = b[0]
            ylim = b[1]
        else:
            xlim = (-5, 5)
            ylim = (-5, 5)
    else:
        xlim = bounds[0]
        ylim = bounds[1]

    # 2. Create Meshgrid
    x = jnp.linspace(xlim[0], xlim[1], resolution)
    y = jnp.linspace(ylim[0], ylim[1], resolution)
    X, Y = jnp.meshgrid(x, y)
    
    # 3. Stack inputs
    # Shape: [resolution, resolution, 2]
    grid_points = jnp.stack([X, Y], axis=-1)

    # 4. Handle Dimensions > 2 (Slicing)
    # If the function requires more dimensions, we pad them with zeros 
    # to visualize the slice through the origin of the other dimensions.
    if objective_fn.dim > 2:
        # Create zeros for remaining dimensions
        # Shape: [resolution, resolution, dim - 2]
        padding = jnp.zeros((*grid_points.shape[:-1], objective_fn.dim - 2))
        grid_points = jnp.concatenate([grid_points, padding], axis=-1)

    # 5. Evaluate
    Z = objective_fn(grid_points)

    return np.array(X), np.array(Y), np.array(Z)


def plot_contour(
    objective_fn: ObjectiveFn,
    ax: Optional[plt.Axes] = None,
    bounds: Optional[Union[List, np.ndarray]] = None,
    resolution: int = 100,
    plot_optima: bool = True,
    log_scale: bool = False,
    cmap: str = 'viridis',
    colorbar: bool = True,
    title: Optional[str] = None
) -> plt.Axes:
    """Plots a 2D contour map of the objective function.

    Args:
        objective_fn: The objective function to plot.
        ax: Matplotlib axes. If None, creates a new figure.
        bounds: Manual bounds [[xmin, xmax], [ymin, ymax]]. 
                If None, uses objective_fn.bounds.
        resolution: Grid density.
        plot_optima: If True, marks the global optimizers with red stars.
        log_scale: If True, uses logarithmic color scaling.
        cmap: Colormap name.
        colorbar: Whether to show the colorbar.
        title: Custom title.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    X, Y, Z = _prepare_grid(objective_fn, bounds, resolution)

    # Configuration
    levels = 50
    norm = LogNorm() if log_scale else None
    
    # Plot Contour
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    
    if colorbar:
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Plot Optima
    if plot_optima and objective_fn.optimizers is not None:
        opts = np.array(objective_fn.optimizers)
        # Scatter the first 2 dimensions of the optimizers
        ax.scatter(opts[:, 0], opts[:, 1], c='red', marker='*', s=150, 
                   edgecolors='white', linewidths=1.0, label='Optima', zorder=10)

    # Styling
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    if title:
        ax.set_title(title)
    elif hasattr(objective_fn, '__class__'):
        ax.set_title(objective_fn.__class__.__name__)

    return ax


def plot_surface(
    objective_fn: ObjectiveFn,
    ax: Optional[plt.Axes] = None,
    bounds: Optional[Union[List, np.ndarray]] = None,
    resolution: int = 100,
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> plt.Axes:
    """Plots a 3D surface of the objective function.

    Args:
        objective_fn: The objective function.
        ax: Matplotlib 3D axes. If None, creates a new figure.
        bounds: [[xmin, xmax], [ymin, ymax]].
        resolution: Grid resolution.
        cmap: Colormap.
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    X, Y, Z = _prepare_grid(objective_fn, bounds, resolution)

    # Plot Surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9)
    
    # Styling
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x)$')
    
    if title:
        ax.set_title(title)
    elif hasattr(objective_fn, '__class__'):
        ax.set_title(f"{objective_fn.__class__.__name__} Surface")

    return ax