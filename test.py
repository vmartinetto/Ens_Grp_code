import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv
Lx=1
Nx=50
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

# kinetic energy operator matrix
Tmat1 = np.diag(vext+1/(dx**2)) + np.diag(vec, k = 1) + np.diag(vec, k = -1)
Tmat2=np.kron(Tmat1,np.identity(Nx))+np.kron(np.identity(Nx),Tmat1)

# interaction energy operator matrix
rootdist=[]
for i in range(Nx):
    for j in range(Nx):
        rootdist.append(abs(i-j))
Wmat=np.zeros((Nx**2,Nx**2))
for i in range(Nx**2):
    Wmat[i,i]=1/math.sqrt(dx**2*(rootdist[i])**2+a**2)
    
# Hamiltonian operator matrix
ham=Tmat2 + Wmat

#Diagonalization
vals,vecs = np.linalg.eigh(ham)

#Check results
e0 = vals[0]
print('Groundstate energy:  ',e0)
print('kinetic energy 0:  ',vecs[:,0].dot(Tmat2.dot(vecs[:,0])))
e1= vals[1]
print('first excited singlet state energy: ', e1)
print('kinetic energy 1:  ',vecs[:,1].dot(Tmat2.dot(vecs[:,1])))
e2= vals[2]
print('second excited singlet state energy: ', e2)
print('kinetic energy 2:  ',vecs[:,2].dot(Tmat2.dot(vecs[:,2])))
e3= vals[3]
print('third excited singlet state energy: ', e3)
print('kinetic energy 3:  ',vecs[:,3].dot(Tmat2.dot(vecs[:,3])))
e4= vals[4]
print('fourth excited singlet state energy: ', e4)
print('kinetic energy 4:  ',vecs[:,4].dot(Tmat2.dot(vecs[:,4])))

#Density
psi0 = np.matrix(vecs[:,0])
psi0mat = np.matrix(np.reshape(psi0,(Nx,Nx)))
corr1RDM = 2*np.dot(psi0mat.getH(),psi0mat)
density     = np.diag(corr1RDM)
psi1 = np.matrix(vecs[:,1])
psi1mat = np.matrix(np.reshape(psi1,(Nx,Nx)))
corr1RDM1 = 2*np.dot(psi1mat.getH(),psi1mat)
density1     = np.diag(corr1RDM1)
psi2 = np.matrix(vecs[:,2])
psi2mat = np.matrix(np.reshape(psi2,(Nx,Nx)))
corr1RDM2 = 2*np.dot(psi2mat.getH(),psi2mat)
density2     = np.diag(corr1RDM2)
psi3 = np.matrix(vecs[:,3])
psi3mat = np.matrix(np.reshape(psi3,(Nx,Nx)))
corr1RDM3 = 2*np.dot(psi3mat.getH(),psi3mat)
density3     = np.diag(corr1RDM3)
psi4 = np.matrix(vecs[:,4])
psi4mat = np.matrix(np.reshape(psi4,(Nx,Nx)))
corr1RDM4 = 2*np.dot(psi4mat.getH(),psi4mat)
density4     = np.diag(corr1RDM4)
#
plt.plot(density, label='GS dens ')
plt.plot(density1, label='1st ES dens ')
plt.plot(density2, label='2nd ES dens ')
plt.plot(density3, label='3rd ES dens ')
plt.plot(density4, label='4th ES dens ')
plt.legend()
plt.show()
plt.close()

#########################################################################

