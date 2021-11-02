import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv
from scipy import interpolate
Lx=1
Nx=300
# Define the grid
x = np.linspace(-Lx*0.5, Lx*0.5, Nx, endpoint = False)
# This reads "I want Nx points equally distributed"
# Define dx
dx =x[2] - x[1] #int(L/Nx) #this gives zero division error
#Define softening parameter
a=0.1
#Import density for N=300 and x = np.linspace(-Lx*0.5, Lx*0.5, Nx, endpoint = True)
density1=np.genfromtxt('denspy1-300-9.dat')
densityGS=np.genfromtxt('denspy-300-9.dat')
gw=0.125
density= (1-gw)*densityGS + gw*density1
#Interpolating density
#ximp = np.linspace(-Lx*0.5, Lx*0.5, 1000, endpoint = False)
#densweights = interpolate.splrep(ximp, densityGS, s=0)
#densityGS = interpolate.splev(x, densweights, der=0)
#
#sqrt_dens0 = np.sqrt(densityGS)
# KS potential from analytical inversion
#vext=0*np.square(ximp)
#dximp =ximp[2] - ximp[1]
#vec = np.ones(ximp.size - 1)/(-2*dximp**2)
#diag = np.ones(1000)/dximp**2
#Tmat = np.diag(vext+1/(dximp**2)) + np.diag(vec, k = 1) + np.diag(vec, k = -1)
#v_KS =(np.divide(np.dot(-Tmat, sqrt_dens0),sqrt_dens0))
#####################################################################
#KS Hamiltonian
#diags = np.array([v_KS+diag, diag/-2, diag/-2])
s = (x.size,x.size)
#HS1 = spa.dia_matrix((diags,[0,-1,1]),shape=s)
#HS2 = spa.kron(HS1,spa.identity(Nx)) + spa.kron(spa.identity(Nx),HS1)
# Diagonalizations of sparse matrix
#vals, vecs = eigsh(HS1,which='SA')
#phi0 = np.matrix(vecs[:,0])
#KSdens= 2*np.square(vecs[:,0])
#checking the correctness of the KS potential wrt the imported density
#plt.plot(x,density, label='GS dens ')
#plt.plot(x,KSdens, label='KS dens ')
#plt.legend()
#plt.show()
#plt.close()
#Hartree potential
#vh = np.convolve(density, 1/np.sqrt(a**2+x**2), mode='same')

def interaction(var):
    return 1/np.sqrt(a**2+var**2)
SC2intmat = np.empty((Nx,Nx))
for i in range(Nx):
    for j in range(Nx):
        SC2intmat[i,j]=interaction(x[i]-x[j])
vh = SC2intmat.dot(density)
diag = np.ones(Nx)/dx**2
diags = np.array([vh+diag, diag/-2, diag/-2])
HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
valsg, vecsg = eigsh(HSguess,which='SA')
#phi0g = np.array(vecsg[:,0])
#gdens = 2*np.square(phi0g)
#phi1g = np.array(vecsg[:,1])
gdens = (2-gw)*np.square(vecsg[:,0])+gw*np.square(vecsg[:,1])
#gdens = np.sum(vecsg[:,0:2]**2,axis=1)
#print(vecsg.shape)
#print(np.sum(gdens*dx))

#plt.plot(x,density-gdens,label='target-inv')
#plt.plot(x, gdens, label='inv dens')
#plt.legend()
#plt.show()
#plt.close

number=1
vinv=np.copy(vh)
while number > 0.000005:
    vinv = gdens*vinv/density
    diags = np.array([vinv+diag, diag/-2, diag/-2])
    HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
    valsg, vecsg = eigsh(HSguess,which='SA')
    #phi0g = np.array(vecsg[:,0])
    #gdens = 2*np.square(phi0g)
    gdens = (2-gw)*np.square(vecsg[:,0])+gw*np.square(vecsg[:,1])
    number=(np.abs(np.ones(Nx)-gdens/density)).max()
    print(number)
E1ks = valsg[0]+valsg[1]
E0ks = 2*valsg[0]
print(valsg[1]-valsg[0])


 
#plt.plot(x,density-gdens, label='GS dens-guess dens ')
plt.plot(x,vinv -vinv[0]-(vh-vh[0]),label='vinv')
#plt.plot(ximp, v_KS-v_KS[0], label='vKS-GS')
plt.legend()
plt.show()
plt.close

#plt.plot(x,gdens,label='init guess 1st')
#plt.plot(x,density,label='ref 1st')
#plt.legend()
#plt.show()
#plt.close()

plt.plot(x,density-gdens,label='target-inv')
#plt.plot(x, gdens, label='inv dens')
plt.legend()
plt.show()
plt.close


#save vH
#np.savetxt('vhpy.dat', zip(np.transpose(x),np.transpose(vh)), fmt='%.4e', delimiter=' ')


