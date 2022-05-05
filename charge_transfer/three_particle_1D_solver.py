import numpy as np
from scipy.integrate import simps
import scipy.sparse as spa
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import math
from scipy.sparse.linalg import eigsh


np.set_printoptions(linewidth=150)


def kron(a,b):
    if a==b:
        return 1
    else:
        return 0

def Kin(p,q,r,i,j,k):
    
    qjrk = kron(q,j)*kron(r,k)*(kron(p,i-1)-2*kron(p,i)+kron(p,i+1))
    pirk = kron(p,i)*kron(r,k)*(kron(q,j-1)-2*kron(q,j)+kron(q,j+1))
    qjpi = kron(q,j)*kron(p,i)*(kron(r,k-1)-2*kron(r,k)+kron(r,k+1))
    return -(1/2)*(qjrk+pirk+qjpi)

def Int(Nx,dx,a):
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

def Int_sparse(Nx,dx,a):
    vint = Int(Nx,dx,a)
    W = spa.dia_matrix((vint,0),shape=(Nx**3,Nx**3))
    return W

def Sparse_Kin_3par(Nx,dx,vext):

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

Nx = 21
L = 2
x = np.linspace(0,L,Nx)
dx = np.abs(x[1]-x[0])
'''
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

vext = np.zeros(len(x))

'''
for i in range(Nx):
    if (dx*i > .25) and (dx*i < .75):
        vext[i] = 20
'''
vint = Int(Nx,dx,.01)

K = Sparse_Kin_3par(Nx,dx,vext)
V = Int_sparse(Nx,dx,.01)
ham = K+V
ham = ham.todense()
vals, vecs = eigsh(ham, which='SA')
'''
plt.spy(mat!=K.todense())
plt.title('dense sparse kinetic not equal')
plt.show()
plt.close()

np.fill_diagonal(mat,np.diagonal(mat)+np.tile(vext,Nx**2)+vint)
'''
np.savetxt('3part_Nx'+str(Nx)+'_L'+str(L)+'_sc.01_sparse.dat', vecs, fmt='%.9e', delimiter=' ')

'''
plt.spy(ham)
plt.show()
plt.close()

plt.spy(mat)
plt.show()
plt.close()

plt.spy(mat!=ham)
plt.title('dense sparse full not equal')
plt.show()
plt.close()
'''
