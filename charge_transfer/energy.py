import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv

def Kin_op_sparse(Nx,dx,vext):

    # main diagonal of finite difference matrix

    diag = np.ones(Nx)/dx**2

    # full finite difference matrix

    diags = np.array([vext+diag, diag/-2, diag/-2])

    # reforming the Matrix as a sparse object and expanding

    Tmat1 = spa.dia_matrix((diags,[0,-1,1]),shape=(Nx,Nx))
    Tmat2 = spa.kron(Tmat1,spa.identity(Nx)) + spa.kron(spa.identity(Nx),Tmat1)

    return Tmat2

def Int_op_sparse(Nx,dx):

    # calculate the rootdist

    rootdist = np.empty(Nx**2)
    k = 0
    for i in range(Nx):
        for j in range(Nx):
            rootdist[k] = abs(i-j)
            k += 1

    # calculate the soft-coulomb interaction and make it a sparse matrix

    Wmat = spa.dia_matrix((1/np.sqrt(dx**2*rootdist**2+a**2),0),shape=(Nx**2,Nx**2))

    return Wmat

# Import the data
dat = np.loadtxt('vecs-1001-9-2030.dat')

# Define my states
ground = dat[:,0]
first = dat[:,1]
second = dat[:,2]
third = dat[:,3]

# set variables
Nx = int(np.sqrt(len(dat[:,0])))
print(Nx)
x = np.linspace(0,6,Nx)
vext = np.zeros(Nx)
dx = np.abs(x[1]-x[0])
a = .1
HtEV = 27.2114

# compute external potnetial
for i in range(Nx):
    if (dx*i > 1) and (dx*i < 2):
        vext[i] = 20
    if (dx*i > 4) and (dx*i < 5):
        vext[i] = 30

# calculate the sparse Kinetic Energy Operator
print('T')
KIN_OP = Kin_op_sparse(Nx,dx,0)
Tmat2 = Kin_op_sparse(Nx,dx,vext)

# calculate the sparse Interaction Energy Operator
print('W')
Wmat = Int_op_sparse(Nx,dx)

# Construct the Hamiltonian Operator from the previous two
print('H')
ham = Tmat2 + Wmat

# calculate energies
print('En')
print('Energy0: ' ,ground.dot(ham.dot(ground)))
print('Energy0: ', HtEV*ground.dot(ham.dot(ground)))
print('T0: ',HtEV*ground.dot(KIN_OP.dot(ground)))
print('Energy1: ', HtEV*first.dot(ham.dot(first)))
print('T1: ',HtEV*first.dot(KIN_OP.dot(first)))
print('Energy2: ', HtEV*second.dot(ham.dot(second)))
print('T2: ',HtEV*second.dot(KIN_OP.dot(second)))
print('Energy3: ', HtEV*third.dot(ham.dot(third)))
print('T3: ',HtEV*third.dot(KIN_OP.dot(third)))
