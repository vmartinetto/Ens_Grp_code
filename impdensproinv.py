import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv
from scipy import interpolate

#########################Define Soft-Coulomb matrix#################################


def interaction(var):
    return 1/np.sqrt(a**2+var**2)

def SC2mat(Nx):
    SC2intmat = np.empty((Nx,Nx))
    for i in range(Nx):
        for j in range(Nx):
            SC2intmat[i,j]=interaction(x[i]-x[j])
    return SC2intmat


#########################Inversion Defenition#######################################


def Lee_Bar_Inv_Ens(vinv,gdens,density,w,conv):
    number = np.Inf
    while  number > conv:
        vinv = gdens*vinv/density
        diags = np.array([vinv+diag, diag/-2, diag/-2])
        HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
        valsg, vecsg = eigsh(HSguess,which='SA')
        gdens = (2-w)*np.square(vecsg[:,0])+w*np.square(vecsg[:,1])
        number=(np.abs(np.ones(Nx)-gdens/density)).max()
        print(number)
    return vinv,valsg,vecsg


#########################Define Variables###########################################


# define the size, number of points, and the 1D grid

Lx=1
Nx=1000
x = np.linspace(-Lx*0.5, Lx*0.5, Nx, endpoint = False)

# Define dx

dx =x[2] - x[1] #int(L/Nx) #this gives zero division error

#Define softening parameter

a=0.1

# define the size of a matrix [Nx,Nx]

s = (Nx,Nx)


########################Import Data#################################################


#Import density for N=300 and x = np.linspace(-Lx*0.5, Lx*0.5, Nx, endpoint = True)
density1=np.genfromtxt('denspy-1000-1-9.dat')
densityGS=np.genfromtxt('denspy-1000-9.dat')


########################Calculate Interacting ensemble density######################


gw=0.125
density= (1-gw)*densityGS + gw*density1

########################Interpolation?##############################################


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

######################Initial Guess###############################################


# 2 body soft-coulomb matrix

SC2intmat = SC2mat(Nx)

# Hartree potnetial

vh = SC2intmat.dot(density)

# Inintial KS eigenfunctions 

diag = np.ones(Nx)/dx**2
diags = np.array([vh+diag, diag/-2, diag/-2])
HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
valsg, vecsg = eigsh(HSguess,which='SA')

# Initial KS  ensemble density 

gdens = (2-gw)*np.square(vecsg[:,0])+gw*np.square(vecsg[:,1])

# the potential updated is a copy of the hartree potential

number=1
vinv=np.copy(vh)

###########################Inversion###################################


vinv, valsg, vecsg = Lee_Bar_Inv_Ens(vinv,gdens,density,gw,.0000005)


###########################Checking energy difference#################


E1ks = valsg[0]+valsg[1]
E0ks = 2*valsg[0]
print('Eksw1-Eksw0: ',valsg[1]-valsg[0])


###########################Plotting###################################

 
#plt.plot(x,density-gdens, label='GS dens-guess dens ')
#plt.plot(x,vinv -vinv[0]-(vh-vh[0]),label='vinv')
#plt.plot(ximp, v_KS-v_KS[0], label='vKS-GS')
#plt.legend()
#plt.show()
#plt.close

#plt.plot(x,gdens,label='init guess 1st')
#plt.plot(x,density,label='ref 1st')
#plt.legend()
#plt.show()
#plt.close()

#plt.plot(x,density-gdens,label='target-inv')
#plt.plot(x, gdens, label='inv dens')
#plt.legend()
#plt.show()
#plt.close


########################Saving info###################################


#save ensemble KS eigenfunctions, Eigenvalues, and Potential

np.savetxt('vks_bi_w=.125_1000_9.dat', vinv, fmt='%.9e', delimiter=' ')
np.savetxt('evecs_bi_w=.125_1000_9.dat', vecsg[:,0:2], fmt='%.9e', delimiter=' ')
np.savetxt('evals_bi_w=.125_1000_9.dat', valsg[0:2], fmt='%.9e', delimiter=' ')

