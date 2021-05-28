import sys
sys.path.append('../')
import numpy as np
import os
import solver
import matplotlib.pyplot as plt
from Matrices import *

def main():
    """
    main control sequence of the code
    """

    # define the parameters needed

    # number of gridpoints
    Nx = 100

    # size of the grid
    Boxsize = 1.0

    # strength of interaction
    U = 1.0

    # dist between the two nuclei
    core_dist = 0.5

    # softenting parameter
    alpha = .1

    # asymetric potential parameter
    del_v = 0.0

    # building the interacting hamiltonian
    T, V, W = energy_mats(Nx, Boxsize, U, core_dist, alpha=alpha)

    #calculate the realspace eigenvectors of the system made with energy_mats
    vals, vecs = solver.solve(T+V, W)
    np.savetxt('eigenvalues.txt',vals)
    np.savetxt('eigenvectors.txt',vecs)

if __name__ == '__main__':
    main()