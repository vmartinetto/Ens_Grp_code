import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv
Lx=1
Nx=1000
# Define the grid
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint = False)
# This reads "I want Nx points equally distributed"

# Define dx
dx =x[2] - x[1] #int(L/Nx) #this gives zero division error

# Define normalizing prefactor wavefunction
Norm = 1/np.sqrt(dx)
# 1) Define the size
s = (x.size,x.size)

# 2) Define the x.size X x.size matrix
h = np.zeros(s)

# Element minor diagonal
vec = np.ones(x.size - 1)/(-2*dx**2)

h = np.diag(1/(dx**2)) + np.diag(vec, k = 1) + np.diag(vec, k = -1)
#########################################################################

