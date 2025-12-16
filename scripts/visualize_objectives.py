import math
import argparse
import matplotlib.pyplot as plt

# Import all synthetic objectives
from ssax.objectives.synthetic import (
    Ackley, Beale, Branin, Bukin, DropWave, Cosine8, EggHolder, 
    HolderTable, SixHumpCamel, ThreeHumpCamel, Rosenbrock, 
    DixonPrice, Michalewicz, Griewank, Powell, Rastrigin, 
    StyblinskiTang, Levy
)
from ssax.objectives.visualization import plot_contour, plot_surface

# List of all function classes to visualize
ALL_OBJECTIVES = [
    Ackley, Beale, Branin, Bukin, DropWave, EggHolder, 
    HolderTable, SixHumpCamel, ThreeHumpCamel, Rosenbrock, 
    DixonPrice, Michalewicz, Griewank, Rastrigin, 
    StyblinskiTang, Levy, Cosine8, Powell
]

def plot_gallery(resolution=100):
    """Plots a grid of 2D contours for all objectives."""
    n_funcs = len(ALL_OBJECTIVES)
    cols = 4
    rows = math.ceil(n_funcs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flatten()
    
    print(f"Generating gallery for {n_funcs} functions...")

    for i, obj_cls in enumerate(ALL_OBJECTIVES):
        ax = axes[i]
        
        # Instantiate with default settings
        # Note: High-dim functions (like Powell/Cosine8) will be sliced to 2D automatically
        try:
            if obj_cls.__name__ == 'Powell':
                obj = obj_cls.create(dim=4) # Powell requires dim multiple of 4
            elif obj_cls.__name__ == 'Cosine8':
                obj = obj_cls.create(dim=8) # Fixed dim 8
            else:
                obj = obj_cls.create(dim=2)
            
            plot_contour(obj, ax=ax, resolution=resolution, colorbar=False, plot_optima=True)
            ax.set_title(obj_cls.__name__, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
        except Exception as e:
            print(f"Skipping {obj_cls.__name__}: {e}")
            ax.text(0.5, 0.5, "Error", ha='center', va='center')

    # Turn off empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_single(name: str, resolution=100):
    """Plots 2D Contour and 3D Surface for a specific function."""
    # Find the class by name (case insensitive)
    obj_cls = next((o for o in ALL_OBJECTIVES if o.__name__.lower() == name.lower()), None)
    
    if obj_cls is None:
        print(f"Error: Function '{name}' not found.")
        print("Available functions:", ", ".join([o.__name__ for o in ALL_OBJECTIVES]))
        return

    print(f"Visualizing {obj_cls.__name__}...")
    
    # Setup Figure
    fig = plt.figure(figsize=(12, 5))
    
    # Instantiate
    if obj_cls.__name__ == 'Powell':
        obj = obj_cls.create(dim=4)
    elif obj_cls.__name__ == 'Cosine8':
        obj = obj_cls.create(dim=8)
    else:
        obj = obj_cls.create(dim=2)

    # Plot 1: 2D Contour
    ax1 = fig.add_subplot(1, 2, 1)
    plot_contour(obj, ax=ax1, resolution=resolution, title=f"{obj_cls.__name__} (Contour)")

    # Plot 2: 3D Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_surface(obj, ax=ax2, resolution=resolution, title=f"{obj_cls.__name__} (Surface)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SSAX Objective Functions")
    parser.add_argument('--name', type=str, default=None, 
                        help='Name of specific function to plot (e.g., "Ackley"). If not provided, plots gallery.')
    parser.add_argument('--res', type=int, default=100, 
                        help='Grid resolution for plotting (default: 100)')

    args = parser.parse_args()

    if args.name:
        plot_single(args.name, resolution=args.res)
    else:
        plot_gallery(resolution=args.res)
