import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
from scipy.sparse import dia_matrix,block_diag
import math
from scipy.linalg import norm
import scipy as sp
import scipy.special
from numpy.linalg import inv
Lx=1
Nx=100
# Define the grid
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint = False)
# This reads "I want Nx points equally distributed"

# Define dx
dx =x[2] - x[1] #int(L/Nx) #this gives zero division error

#Define softening parameter
#a=0.01#in agreement with parameter in PYTB14, see eq (67)
a=0.1
# Define normalizing prefactor wavefunction
Norm = 1/np.sqrt(dx)
# 1) Define the size
s = (x.size,x.size)

# 2) Define the x.size X x.size matrix
vext=0*np.square(x)

# Element minor diagonal
vec = np.ones(x.size - 1)/(-2*dx**2)


# kinetic energy operator matrix as sparse matrices
diag = np.ones(Nx)/dx**2
diags = np.array([vext+diag, diag/-2, diag/-2])
Tmat1 = spa.dia_matrix((diags,[0,-1,1]),shape=s)
Tmat2 = spa.kron(Tmat1,spa.identity(Nx)) + spa.kron(spa.identity(Nx),Tmat1)
#extension T mat
addnr=int(scipy.special.binom(2*Nx, 2)-Nx**2)
adddiag =np.ones(addnr)/dx**2
adddiags = np.array([2*adddiag, adddiag/-2, adddiag/-2])
Tmatadd = spa.dia_matrix((adddiags,[0,-2,2]),shape=(addnr,addnr))
#T fusion
extT=block_diag((Tmat2, Tmatadd))


# interaction energy operator matrix as sparse matrix
rootdist = np.empty(Nx**2)
k = 0
for i in range(Nx):
    for j in range(Nx):
        rootdist[k] = abs(i-j)
        k += 1
Wmat = spa.dia_matrix((1/np.sqrt(dx**2*rootdist**2+a**2),0),shape=(Nx**2,Nx**2))
#extension W mat
warray = list(rootdist)
try:
    while True:
        warray.remove(0)
except ValueError:
    pass
warray=np.repeat(warray,2)

Wmatadd = spa.dia_matrix((1/np.sqrt(dx**2*warray**2+a**2),0),shape=(addnr,addnr))
#W fusion
extW=block_diag((Wmat, Wmatadd))
# Hamiltonian operator matrix
#ham = Tmat2 + Wmat
ham=extT+extW

# Diagonalization of dense matrix
#vals,vecs = np.linalg.eigh(ham)

# Diagonalizations of sparse matrix
vals, vecs = eigsh(ham,which='SA')

#Check results
e0 = vals[0]
print('Groundstate energy:  ',e0)
print('kinetic energy 0:  ',vecs[:,0].dot(extT.dot(vecs[:,0])))
#exporting
e1= vals[1]
print('first excited state energy: ', e1)
print('kinetic energy 1:  ',vecs[:,1].dot(extT.dot(vecs[:,1])))
e2= vals[2]
print('second excited state energy: ', e2)
print('kinetic energy 2:  ',vecs[:,2].dot(extT.dot(vecs[:,2])))
e3= vals[3]
print('third excited state energy: ', e3)
print('kinetic energy 3:  ',vecs[:,3].dot(extT.dot(vecs[:,3])))
e4= vals[4]
print('fourth excited state energy: ', e4)
print('kinetic energy 4:  ',vecs[:,4].dot(extT.dot(vecs[:,4])))



