import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import norm
import scipy as sp
from scipy.integrate import simps
from numpy.linalg import inv

################################Density Function####################################

def density(vecs,occs):
    """
    Takes the KS eigenvectors and constructs the ground state density of the 
    interacting system.

    INPUT:
        vecs: np.array, shape=(Nx,nvecs)
            Nx is equal to the number of grid points of the calculation. Nvecs
            can be any size but only the first len(occ) vectors will be used.
        occs: list
            a list of the occupation numbers of the KS eigenfunctions that construct
            the ground state density of the interacting system
    OUTPUT:
        density: np.array, shape=(Nx)
            A vector of of Nx gridpoints describing the ground state density of the 
            interacting density.
    """
    Nx = len(vecs[:,0])
    density = np.zeros(Nx)
    for i,occ in enumerate(occs):
        if occ == 0:
            continue
        density += occ*np.square(vecs[:,i])
    return density


np.set_printoptions(precision=4)

dat = np.loadtxt('evecs_ION_1001_9-2030.dat')

occs = [[2],[1,1],[0,2],[0,1,1],[0,0,2],[0,0,1,1]]
states = len(occs)

dens = np.empty((len(dat[:,0]),states))

Nx = len(dat[:,0])
x = np.linspace(0,6,Nx)

#density calculation
for i,occ in enumerate(occs):
    dens[:,i] = density(dat,occ)

print(simps(dens[:,0]))

#normilization
for i in range(states):
    dens[:,i] = dens[:,i]*(2/simps(dens[:,i],x))

print(simps(dens[:,0],x))
print(dens[:,0].dot(dens[:,0]))

#overlap calculation
overlap = np.empty((states,states))
for i in range(states):
    for j in range(states):
        overlap[i,j] = dens[:,i].dot(dens[:,j])

print(overlap)
