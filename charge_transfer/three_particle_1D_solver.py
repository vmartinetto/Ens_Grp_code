import numpy as np
from scipy.integrate import simps
import scipy.sparse as spa
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import math
from scipy.sparse.linalg import eigsh

###################################Kronecker Delta#################################

def kron(a,b):
    '''
    A simple Kronecker delta implementation
    Input
        a,b: Real, Float, Int
            Any real, float, or integer value
    Output
        1 or 0:
            returns 1 if the value are the same, returns 0 if they do not
    '''
    if a==b:
        return 1
    else:
        return 0

#############################Kinectic Element Calculator############################ 

'''

Formation of the dense Kinetic Energy Matrix using the function Kin

mat = np.empty((Nx**3,Nx**3))


for p in range(Nx):
    for q in range(Nx):
        for r in range(Nx):
            for i in range(Nx):
                for j in range(Nx):
                    for k in range(Nx):
                        n = (p*Nx**2)+(q*Nx)+r
                        m = (i*Nx**2)+(j*Nx)+k
                        mat[n,m] = Kin(p,q,r,i,j,k)
                        if ((n*Nx**3)+m)%1000000==0:
                            print((n*Nx**3)+m)

mat = mat/dx**2
'''

def Kin(p,q,r,i,j,k):
    '''
    Calculates the value of an element of the 1D three particle Kinetic
    Energy matrix given the six values p,q,r,i,j,k.

    INPUT
        p,q,r,i,j,k: Int
            The indices of the left and right eigenfunctions given three
            particles confined to a one dimensional box. 
    OUTPUT
        Knm: real, float
            The value of the kinetic energy matrix at indixes n and m given 
            the single particle basis functions are delta functions.
    '''
    qjrk = kron(q,j)*kron(r,k)*(kron(p,i-1)-2*kron(p,i)+kron(p,i+1))
    pirk = kron(p,i)*kron(r,k)*(kron(q,j-1)-2*kron(q,j)+kron(q,j+1))
    qjpi = kron(q,j)*kron(p,i)*(kron(r,k-1)-2*kron(r,k)+kron(r,k+1))
    Knm = -(1/2)*(qjrk+pirk+qjpi)
    return Knm

###########################Soft-Coulomb Interaction Calculator#####################

def Int(Nx,dx,a):
    '''
    Calculates the vector that lies along the main diagonal of the interaction potetnial
    matrix given delta function basis fnuctions in a 1D box. This is a soft-coulomb interaction
    not a full coulomb. A full coulomb would require special processing to solve in 1D.

    INPUT
        Nx: Int
            The number of wanted gridpoints in the 1D box.
        dx: float
            The grid spacing in the 1D box. dx*(Nx-1) = Length of box
        a: float
            The softening parameter of the soft-coulomb interaction.
    OUTPUT
        vint: np.array, vector, len=Nx**3
            The soft-coulomb interaction that lies along the main diagonal of the 
            interaction matrix given delta function basis functions in a 1D box.
    '''
    vint = np.empty(Nx**3)
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                m = (i*Nx**2)+(j*Nx)+k
                vint[m]= (1 / math.sqrt(dx ** 2 * (i-j) ** 2 + a ** 2) 
                        + 1 / math.sqrt(dx ** 2 * (i-k) ** 2 + a ** 2) 
                        + 1 / math.sqrt(dx ** 2 * (j-k) ** 2 + a ** 2)
                        )
    return vint

#############################3-particle Sparse Operators##########################

def Int_sparse(Nx,dx,a):
    '''
    Mostly a wrapper around Int. makes the sparse matrix from Int and returns it.

    INPUT
        Nx: Int
            The number of wanted gridpoints in the 1D box.
        dx: float
            The grid spacing in the 1D box. dx*(Nx-1) = Length of box
        a: float
            The softening parameter of the soft-coulomb interaction.
    OUTPUT
        W: scipy.sparse.dia_matrix, shape=(Nx**3,Nx**3)
            A scipy sparse matrix object with vint from Int along the main diagonal.
    '''
    vint = Int(Nx,dx,a)
    W = spa.dia_matrix((vint,0),shape=(Nx**3,Nx**3))
    return W

def Sparse_Kin_3par(Nx,dx,vext):
    '''
    Constructs the kinetic energy matrix for three interacting particles in a 1D
    box given a delta function basis set. Using a three point centeral finite differnce for
    the second derivative.

    INPUT
        Nx: Int
            The number of wanted gridpoints in the 1D box.
        dx: float
            The grid spacing in the 1D box. dx*(Nx-1) = Length of box
        vext: np,array, vector, len=Nx
            A vector containg the external potential within the 1D box. It 
            is repeated Nx**2 times over the main diagonal of K.
    OUTPUT
        K: scipy.sparse.dia_matrix, shape=(Nx**3,Nx**3) 
            A scipy sparse matrix object with the bands of the kinetic matrix as well
            as the external potnetial.

    '''
    # make the diagonals of the sparse matrix
    main = np.ones(Nx**3)*(3/dx**2)
    off1 = np.ones(Nx**3-1)*(-.5/dx**2)
    offNx = np.ones(Nx**3-Nx)*(-.5/dx**2)
    offNx2 = np.ones(Nx**3)*(-.5/dx**2)

    #add zeroes where necessary
    for i in range(Nx**2-1):
        off1[(Nx*(i+1))-1] = 0
    for i in range(Nx-1):
        offNx[(Nx**2*(i+1))-Nx:(Nx**2*(i+1)+Nx)-Nx] = 0

    #pad vectors
    offu1 =  np.append([0],off1)
    offd1 =  np.append(off1,[0])
    offuNx = np.append(np.zeros(Nx), offNx)
    offdNx = np.append(offNx, np.zeros(Nx))

    #construct the diagonal matrix
    diags = np.array([main+np.tile(vext,Nx**2), offd1, offu1, offdNx, offuNx, offNx2, offNx2])
    print(diags)
    K = (spa.dia_matrix((diags, [0, -1, 1, -Nx, Nx, -Nx**2, Nx**2]), 
        shape= (Nx**3,Nx**3))
            )

    return K

###################################Matrix Visulization###########################

def Mat_view(mat):
    plt.spy(mat)
    plt.title('Mat View')
    plt.show()
    plt.close()
    return _

def Sparse_mat_view(mat):

    plt.spy(mat)
    plt.title('Sparse Mat View')
    plt.show()
    plt.close()
    return _

########################################MAIN####################################

if __name__ == '__main__':
    Nx = 23
    L = 3
    x = np.linspace(0,L,Nx)
    dx = np.abs(x[1]-x[0])

    vext = np.zeros(len(x))

'''
    for i in range(Nx):
        if (dx*i > .25) and (dx*i < .75):
            vext[i] = 20
'''

    K = Sparse_Kin_3par(Nx,dx,vext)
    V = Int_sparse(Nx,dx,.01)
    ham = K+V
    vals, vecs = eigsh(ham, which='SA')
    np.savetxt('3part_Nx'+str(Nx)+'_L'+str(L)+'_sc.01_sparse.dat', vecs, fmt='%.9e', delimiter=' ')

