import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import norm
import scipy as sp
from scipy.integrate import simps
from scipy.special import hyp2f1
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

#########################Density Construction#######################################

def ground(vecs,occs):
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
        ground: np.array, shape=(Nx)
            A vector of of Nx gridpoints describing the ground state density of the 
            interacting density.
    """
    Nx = len(vecs[:,0])
    ground = np.zeros(Nx)
    for i,occ in enumerate(occs):
        ground += occ*np.square(vecs[:,i])
    return ground

#########################Exchange and Correlation###################################

def LDA_exchange(n,dx):
    Ex = -3/4*(3/np.pi)**(1/3)*simps(n**(4/3),dx=dx)
    Vx = -(3/np.pi)**(1/3)*n**(1/3)
    return Ex, Vx

def LDA_correlation(n,dx):
    #function for the ground state correlation of 1D systems employing the LDA

    #parameters
    a1 = -np.pi**2/360
    a2 = 3/4 - np.log(2*np.pi)/2
    a3 = 2.408779

    #correlation energy per particle
    ec = a1*hyp2f1(1,1.5,a3,(a1*(1-a3))/(a2*n))

    #correlation energy
    Ec = simps(n*ec,dx=dx)

    return Ec, ec

#########################Inversion Defenition#######################################

def Lee_Bar_Inv(vinv,gdens,density,conv):
    Nx = len(density)
    number = np.Inf
    print('before loop')
    while  number > conv:
        print('in loop')
        vinv = gdens*vinv/density # prev implemented
        #vinv = vinv*(gdens/density)
        plt.plot(vinv)
        plt.show()
        plt.close()
        diags = np.array([vinv+diag, diag/-2, diag/-2])
        HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=(Nx,Nx))
        print('diag')
        valsg, vecsg = eigsh(HSguess,which='SA')
        print('ground')
        gdens = ground(vecsg,[2])
        plt.plot(gdens-density,label='diff')
        plt.plot(density,label='density')
        plt.plot(gdens,label='gdens')
        plt.legend()
        plt.show()
        plt.close()
        number=(np.abs(np.ones(Nx)-gdens/density)).max()
        print(np.abs(gdens-density).max())
        print(number)
    return vinv,valsg,vecsg

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

def Piers_Neck_War(x,r,vexact,density,accel,conv):
    number = np.Inf
    vxc = -((3/np.pi)*density)**(1/3)
    diag = np.ones(Nx)/dx**2
    diags = np.array([vexact+vxc+diag, diag/-2, diag/-2])
    HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
    valsg, vecsg = eigsh(HSguess,which='SA')
    gdens = ground(vecsg,[2])
    gdens = 2/(simps(gdens,x))*gdens
    while number > conv:
        vxc = vxc + accel[0]*r**accel[1]*(gdens-density) 
        
        #plotting
        #plt.plot(vexact+vxc)
        #plt.show()
        #plt.close()

        diags = np.array([vexact+vxc+diag, diag/-2, diag/-2])
        HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
        valsg, vecsg = eigsh(HSguess,which='SA')
        gdens = ground(vecsg,[2])
        gdens = 2/(simps(gdens,x))*gdens

        #more plotting
        #plt.plot(gdens-density,label='diff')
        #plt.plot(density,label='density')
        #plt.plot(gdens,label='gdens')
        #plt.legend()
        #plt.show()
        #plt.close()

        #number=(np.abs(np.ones(Nx)-gdens/density)).max() 
        number = 4*np.pi*simps(r**2*np.abs(gdens-density),x)
        print('conv criterion: ', number)
    return vxc, valsg, vecsg

def Piers_Neck_War_Ion(x,r,fr,Ie,vexact,density,accel,conv):
    number = np.Inf
    vxc = -((3/np.pi)*density)**(1/3)
    diag = np.ones(Nx)/dx**2
    diags = np.array([vexact+vxc+diag, diag/-2, diag/-2])
    HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
    valsg, vecsg = eigsh(HSguess,which='SA')
    gdens = ground(vecsg,[2])
    gdens = 2/(simps(gdens,x))*gdens
    while number > conv:
        vxc = vxc + accel[0]*r**accel[1]*(gdens-density) + (-valsg[0]-Ie)*fr
        
        #plotting
        #plt.plot(vexact+vxc)
        #plt.show()
        #plt.close()

        diags = np.array([vexact+vxc+diag, diag/-2, diag/-2])
        HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
        valsg, vecsg = eigsh(HSguess,which='SA')
        gdens = ground(vecsg,[2])
        gdens = 2/(simps(gdens,x))*gdens

        #more plotting
        #plt.plot(gdens-density,label='diff')
        #plt.plot(density,label='density')
        #plt.plot(gdens,label='gdens')
        #plt.legend()
        #plt.show()
        #plt.close()

        #number=(np.abs(np.ones(Nx)-gdens/density)).max() 
        number = 4*np.pi*simps(r**2*np.abs(gdens-density),x)
        print('conv criterion: ', number)
    return vxc, valsg, vecsg

#########################Define Variables###########################################


# define the size, number of points, and the 1D grid

Lx=6
Nx=1001
x = np.linspace(0, Lx, Nx)

# Define dx

dx =x[2] - x[1] #int(L/Nx) #this gives zero division error

#Define softening parameter

a=0.1

# define the size of a matrix [Nx,Nx]

s = (Nx,Nx)

# compute external potnetial
vext = np.zeros(Nx)
for i in range(Nx):
    if (dx*i > 1) and (dx*i < 2):
        vext[i] = 20
    if (dx*i > 4) and (dx*i < 5):
        vext[i] = 30


########################Import Data#################################################


#Import density for N=1001 and x = np.linspace(0, 6, Nx)
density=np.genfromtxt('denspy-1001-9-2030.dat')


########################Calculate Interacting ensemble density######################


#gw=0.125
#density= (1-gw)*densityGS + gw*density1

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

print('matrix formation')
SC2intmat = SC2mat(Nx)

# Hartree potnetial

vH = SC2intmat.dot(density)
Ex, vX = LDA_exchange(density,dx)
Ec, vC = LDA_correlation(density,dx)

#normalize
density = (2/simps(density,x))*density

# Inintial KS eigenfunctions 

#vHm = .5
#vextm = 4
#vXm = 1
#vCm = 1

#diag = np.ones(Nx)/dx**2
#diags = np.array([(vextm*vext)+(vHm*vH)+(vXm*vX)+(vCm*vC)+diag, diag/-2, diag/-2])
#HSguess= spa.dia_matrix((diags,[0,-1,1]),shape=s)
#valsg, vecsg = eigsh(HSguess,which='SA')

# Initial KS density 

#gdens = ground(vecsg,[2])
#plt.plot(gdens)
#plt.show()
#plt.close()

# the potential updated is a copy of the hartree potential

#vinv=np.copy(vh)
#vinv = np.copy(vext)
#vinv = (vextm*vext)+(vHm*vH)+(vXm*vX)+(vCm*vC)
#plt.plot(vinv)
#plt.show()
#plt.close()

###########################Inversion###################################

print('beginiing inversion')
#vinv, valsg, vecsg = Lee_Bar_Inv(vinv,gdens,density,.0005)

accel = [1,2.5,1,3]

L = 1
r = L-np.abs(np.linspace(-L,L,Nx))
f1  =np.heaviside(1,r)*r**accel[2]
f2 = np.heaviside(r,1)/(r**accel[3]+1) 
f3 = f1+f2

Ie = -4.703283906868827

vinv, valsg, vecsg = Piers_Neck_War_Ion(x,r,f3,Ie,vH+vext,density,accel,.005)
plt.plot(vinv)
plt.show()
plt.close()


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

np.savetxt('vxc_ION_1001_9-2030.dat', vinv, fmt='%.9e', delimiter=' ')
np.savetxt('vks_ION_1001_9-2030.dat', vinv+vH+vext, fmt='%.9e', delimiter = ' ')
np.savetxt('evecs_ION_1001_9-2030.dat', vecsg, fmt='%.9e', delimiter=' ')
np.savetxt('evals_ION_1001_9-2030.dat', valsg, fmt='%.9e', delimiter=' ')
