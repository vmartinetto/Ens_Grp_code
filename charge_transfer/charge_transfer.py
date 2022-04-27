import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv

##################Kinetic Energy Operator Functions###############


def Kin_op_dense(Nx,dx,vext):

    #  off-diagonal vector of finite difference matrix 

    vec = np.ones(Nx - 1)/(-2*dx**2)


    # kinetic energy operator matrix as dense matrix

    Tmat1 = np.diag(vext+1/(dx**2)) + np.diag(vec, k = 1) + np.diag(vec, k = -1)
    Tmat2=np.kron(Tmat1,np.identity(Nx))+np.kron(np.identity(Nx),Tmat1)

    return Tmat2

def Kin_op_sparse(Nx,dx,vext):

    # main diagonal of finite difference matrix

    diag = np.ones(Nx)/dx**2

    # full finite difference matrix

    diags = np.array([vext+diag, diag/-2, diag/-2])

    # reforming the Matrix as a sparse object and expanding

    Tmat1 = spa.dia_matrix((diags,[0,-1,1]),shape=s)
    Tmat2 = spa.kron(Tmat1,spa.identity(Nx)) + spa.kron(spa.identity(Nx),Tmat1)

    return Tmat2


#################Interaction Energy Operators#################


def Int_op_dense(Nx,dx):

    # cacculate rootdist
    rootdist = []
    for i in range(Nx):
        for j in range(Nx):
            rootdist.append(abs(i - j))

    # define Wmat as a matrix that is [Nx^2,Nx^2]        

    Wmat = np.zeros((Nx ** 2, Nx ** 2))

    #populate Wmat

    for i in range(Nx ** 2):
        Wmat[i, i] = 1 / math.sqrt(dx ** 2 * (rootdist[i]) ** 2 + a ** 2)

    return Wmat

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


#################Density Function###############################

def density_calc(Nx,psi):

    # reshape data and calculate density
    psimat = np.matrix(np.reshape(psi,(Nx,Nx)))
    corr1RDM = 2*np.dot(psimat.getH(),psimat)
    density     = np.diag(corr1RDM)

    return density


##################Defining Variables##############################


# Length of box and number of gridpoints

Lx=4
Nx=1001

# Define the grid

x = np.linspace(0, 4, Nx)

# Define the spacing between each grid point

dx =x[2] - x[1] 

#Define softening parameter

a=0.1

# Define normalizing prefactor wavefunction

Norm = 1/np.sqrt(dx)

# Define the size of the of a matrix [Nx,Nx]

s = (x.size,x.size)

# Define the external potetial

vext = np.zeros(len(x))
for i in range(Nx):
    if (dx*i >= 1) and (dx*i <= 2):
        vext[i] = 20


###################Operator Calculation########################


# calculate the sparse Kinetic Energy Operator

Tmat2 = Kin_op_sparse(Nx,dx,vext)

# calculate the sparse Interaction Energy Operator

Wmat = Int_op_sparse(Nx,dx)

# Construct the Hamiltonian Operator from the previous two

ham = Tmat2 + Wmat


##################Digonilization#############################


# Diagonalization of dense matrix

#vals,vecs = np.linalg.eigh(ham)

# Diagonalizations of sparse matrix
print('starting diagonilization, this may take a while:')
print()
vals, vecs = eigsh(ham,which='SA')
print('diagonilization done, saving densities and vectors:')
print()

###################Density Calculation#######################

density0 = density_calc(Nx,vecs[:,0]) 
density1 = density_calc(Nx,vecs[:,1])

###################Density Saving###############################
np.savetxt('denspy-1001-9.dat', density0, fmt='%.9e', delimiter=' ')
np.savetxt('denspy-1001-1-9.dat', density1, fmt='%.9e', delimiter=' ')
np.savetxt('vecs-1001-9.dat', vecs, fmt='%.9e', delimiter=' ')
print('done')
