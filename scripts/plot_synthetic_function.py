import matplotlib.pyplot as plt

from ssax.objectives.synthetic import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    DropWave,
    Cosine8,
    EggHolder,
    HolderTable,
    SixHumpCamel,
    ThreeHumpCamel,
    Rosenbrock,
    DixonPrice,
    Michalewicz,
    Griewank,
    Powell,
    Rastrigin,
    StyblinskiTang
)
from ssax.objectives.visualization import plot_objective


if __name__ == '__main__':
    plt.figure()
    # plot_objective(Ackley())
    plot_objective(StyblinskiTang())
    plt.show()
