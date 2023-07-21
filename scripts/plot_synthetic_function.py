import matplotlib.pyplot as plt

from ssax.objectives.synthetic import Ackley, Beale
from ssax.objectives.visualization import plot_objective


if __name__ == '__main__':
    plt.figure()
    # plot_objective(Ackley())
    plot_objective(Beale())
    plt.show()
