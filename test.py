import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm
import scipy as sp

L = 1 # Size of the box

a = 0.1 # Coulomb softening parameter

Nx = 1000 # Number of gridpoints

x1 = np.linspace(0, L, Nx, endpoint = True) # Grid for particle 1

dx1 = x1[2] - x1[1] # Spacing for grid 1

x2 = np.linspace(0, L, Nx, endpoint = True) # Grid for particle 2

dx2 = x2[2] - x2[1] # Spacing for grid 2

vec1 = np.ones(x1.size - 1)/(-2*dx1**2) 

vec2 = np.ones(x2.size - 1)/(-2*dx2**2) 

Kin1 = np.diag(np.ones(x1.size)*1/(dx1**2)) + np.diag(vec1, k = 1) + np.diag(vec1, k = -1) # Kinetic energy of particle 1

Kin2 = np.diag(np.ones(x2.size)*1/(dx2**2)) + np.diag(vec2, k = 1) + np.diag(vec2, k = -1) # Kinetic energy of particle 2


Vsc = np.empty((x2.size,x2.size))


for i, XX1 in enumerate(x1):
	for j, XX2 in enumerate(x2):
		Vsc[i,j] = 1/(np.sqrt((XX1-XX2)**2 + a**2))

Ham = Kin1 + Kin2 + Vsc ????


print(Vsc)
