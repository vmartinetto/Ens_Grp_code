import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv

Lx = 1
Nx = 10
# Define the grid
x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
# This reads "I want Nx points equally distributed"

# Define dx
dx = x[2] - x[1]  # int(L/Nx) #this gives zero division error

# Define softening parameter
# a=0.01#in agreement with parameter in PYTB14, see eq (67)
a = 0.1
# Define normalizing prefactor wavefunction
Norm = 1 / np.sqrt(dx)
# 1) Define the size
s = (Nx, Nx)

# 2) Define the x.size X x.size matrix
vext = 0 * np.square(x)

# Element minor diagonal
# vec = np.ones(x.size - 1) / (-2 * dx ** 2)

# kinetic energy operator matrix as dense matrices
# Tmat1 = np.diag(vext + 1 / (dx ** 2)) + np.diag(vec, k=1) + np.diag(vec, k=-1)
# Tmat2 = np.kron(Tmat1, np.identity(Nx)) + np.kron(np.identity(Nx), Tmat1)

# kinetic energy operator matrix as sparse matrices
diag = np.ones(Nx)/dx**2
diags = np.array([vext+diag, diag/-2, diag/-2])
Tmat1 = spa.dia_matrix((diags,[0,-1,1]),shape=s)
Tmat2 = spa.kron(Tmat1,spa.identity(Nx)) + spa.kron(spa.identity(Nx),Tmat1)


# interaction energy operator matrix as dense matrix
#rootdist = []
#for i in range(Nx):
#    for j in range(Nx):
#        rootdist.append(abs(i - j))
#Wmat = np.zeros((Nx ** 2, Nx ** 2))
#for i in range(Nx ** 2):
#    Wmat[i, i] = 1 / math.sqrt(dx ** 2 * (rootdist[i]) ** 2 + a ** 2)

# interaction energy operator matrix as sparse matrix
rootdist = np.empty(Nx**2)
k = 0
for i in range(Nx):
    for j in range(Nx):
        rootdist[k] = abs(i-j)
        k += 1
Wmat = spa.dia_matrix((1/np.sqrt(dx**2*rootdist**2+a**2),0),shape=(Nx**2,Nx**2))

# Hamiltonian operator matrix
ham = Tmat2 + Wmat

# Diagonalization of dense matrix
vals_c, vecs_c = eigh(ham.toarray())

# Diagonalizations of sparse matrix
vals, vecs = eigsh(ham,which='SA')

# Check results
e0 = vals[0]
print('Groundstate energy:  ', e0)
print('kinetic energy 0:  ', vecs[:, 0].dot(Tmat2.dot(vecs[:, 0])))
e1 = vals[1]
print('first excited singlet state energy: ', e1)
print('kinetic energy 1:  ', vecs[:, 1].dot(Tmat2.dot(vecs[:, 1])))
e2 = vals[2]
print('second excited singlet state energy: ', e2)
print('kinetic energy 2:  ', vecs[:, 2].dot(Tmat2.dot(vecs[:, 2])))

#########################################################################