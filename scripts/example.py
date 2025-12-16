import argparse
import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation, rc

# SSAX Imports
from ssax.objectives.synthetic import Ackley, Rastrigin, Rosenbrock, StyblinskiTang
from ssax.objectives.visualization import plot_contour
from ssax.ss.initializer import UniformInitializer
from ssax.ss.solver import SinkhornStep
from ssax.ss.epsilon_scheduler import LinearEpsilon

# OTT Imports
from ott.solvers.linear.sinkhorn import Sinkhorn

# Configure Animation
rc('animation', html='html5')

def main(args):
    print("="*60)
    print(f"SSAX Example Script: {args.objective}")
    print("="*60)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, warm_rng = jax.random.split(rng, 3)

    # 1. Define Objective Function
    # Select function based on argument
    if args.objective.lower() == 'ackley':
        objective_fn = Ackley.create(dim=2)
    elif args.objective.lower() == 'rastrigin':
        objective_fn = Rastrigin.create(dim=2)
    elif args.objective.lower() == 'rosenbrock':
        objective_fn = Rosenbrock.create(dim=2)
    elif args.objective.lower() == 'styblinskitang':
        objective_fn = StyblinskiTang.create(dim=2)
    else:
        raise ValueError(f"Unknown objective: {args.objective}")

    print(f"Objective: {objective_fn.__class__.__name__} (Dim: {objective_fn.dim})")

    # 2. Initialize Particles
    initializer = UniformInitializer.create(
        rng=init_rng,
        bounds=objective_fn.bounds.tolist()
    )
    X_init = initializer(args.num_points)
    print(f"Particles: {args.num_points}")

    # 3. Setup Solver Components
    # Linear OT Solver (Inner Loop)
    linear_solver = Sinkhorn(
        threshold=1e-3,
        inner_iterations=1,
        min_iterations=1,
        max_iterations=100
    )

    # Epsilon Scheduler (Annealing)
    epsilon = LinearEpsilon(
        target=args.epsilon_target,
        init=args.epsilon_init,
        decay=args.epsilon_decay,
    )

    # Sinkhorn Step Solver (Outer Loop)
    sinkhorn_step = SinkhornStep.create(
        objective_fn=objective_fn,
        linear_ot_solver=linear_solver,
        epsilon=epsilon,
        polytope_type='orthoplex',
        step_radius=0.15,
        probe_radius=0.2,
        num_probe=5,
        min_iterations=5,
        max_iterations=args.max_iters,
        store_history=True,
        threshold=1e-3
    )

    # 4. Warm Start (JIT Compilation)
    print("\n[1/3] Compiling JAX functions...")
    tic = time.perf_counter()
    init_state = sinkhorn_step.warm_start(X_init, rng=warm_rng)
    # Block to measure true compilation time
    init_state.costs.block_until_ready()
    toc = time.perf_counter()
    print(f"      Done in {toc - tic:.4f} s")

    # 5. Run Optimization
    print("\n[2/3] Running Optimization...")
    tic = time.perf_counter()
    res = sinkhorn_step.iterations(init_state)
    res.costs.block_until_ready()
    toc = time.perf_counter()

    # Determine convergence index
    valid_iters = jnp.sum(res.costs != -1.0)
    converged_at = int(valid_iters)
    
    print(f"      Finished in {toc - tic:.4f} s")
    print(f"      Converged at iteration: {converged_at}")

    # 6. Visualization
    print("\n[3/3] Generating Animation...")
    
    # Extract History [iters, N, dim]
    X_history = res.X_history
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Use the new visualization util for the background
    plot_contour(objective_fn, ax=ax, plot_optima=True, title=f"{objective_fn.__class__.__name__}")
    
    # Initialize Scatter Plot
    # Adjust marker size based on number of points for clarity
    s_size = 20 if args.num_points < 1000 else 2
    scatter = ax.scatter([], [], c='white', s=s_size, edgecolors='black', alpha=0.7, linewidth=0.5)
    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def init():
        scatter.set_offsets(X_init[:, :2])
        text.set_text('Iter: 0')
        return scatter, text

    def update(frame):
        # Ensure we don't index past the converged iteration
        idx = min(frame, converged_at - 1)
        data = X_history[idx]
        scatter.set_offsets(data[:, :2])
        text.set_text(f'Iter: {idx}')
        return scatter, text

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, converged_at, 2), # Skip every other frame for speed
        interval=50, blit=True
    )
    
    filename = f"{args.objective.lower()}_{args.num_points}.gif"
    try:
        anim.save(filename, writer='imagemagick', fps=20)
        print(f"      -> Saved to {filename}")
    except Exception as e:
        print(f"      -> Error saving GIF: {e}")
        # Fallback to showing plot if saving fails
        plt.show()

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', type=str, default='Ackley', help='Objective function name')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of particles')
    parser.add_argument('--max_iters', type=int, default=100, help='Max optimization iterations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Hyperparameters
    parser.add_argument('--epsilon_init', type=float, default=1.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.01)
    parser.add_argument('--epsilon_target', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
