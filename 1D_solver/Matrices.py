import numpy as np
import math

def gradient_mat(Nsites, boxsize):
    """
    this functions takes parameters Nsites and boxsize and computes the matrix representation for the first finite
    difference acting on a real space wavefunction in 1D.

    :param Nsites: integer. The number of sites in the box.
    :param boxsize: float. the total length of the box in 1D
    :return Gmat: the two point central finite difference matrix for the first derivative in 1D
    """
    dx = boxsize/Nsites
    Gmat = np.zeros((Nsites,Nsites))
    for i in range(Nsites-1):
        Gmat[i, i+1] = 1/(2*dx)
        Gmat[i+1, i] = -1/(2*dx)
    return Gmat

def laplacian_mat(Nsites,boxsize):
    """
    this functions takes parameters Nsites and boxsize and computes the matrix representation for the second finite
    difference acting on a real space wavefunction in 1D.

    :param Nsites: integer. The number of sites in the box.
    :param boxsize: float. the total length of the box in 1D
    :return Lmat:  the three point central finite difference matrix for the second derivative in 1D
    """
    dx = boxsize / Nsites
    Lmat = np.zeros((Nsites, Nsites))
    for i in range(Nsites - 1):
        Lmat[i, i + 1] = 1/dx**2
        Lmat[i + 1, i] = -1/dx**2
        Lmat[i,i] = -2/dx**2
    Lmat[Nsites-1,Nsites-1] = -2/dx**2
    return Lmat

def energy_mats(Nsites, boxsize, u0, dist, del_v=0, alpha=1):
    """
    :param Nsites:  integer. The number of sites in the box.
    :param boxsize: float. the total length of the box in 1D
    :param u0: float. interaction strength
    :param dist: the distance between the nuclei
    :param del_v: the difference in the potential wells, default is none at zero
    :param alpha: the sofenting parameter of the soft-coulomb interaction.
    :return Tmat: the matrix that represents the kinetic energy operator for a realspace wavefunction
    :return V_ext_mat: the matrix that represents the given external potential operator for a realspace wavefunction
    :return V_SC_mat: the Tensor that represents the SC operator for the realspace wavefunction
    """
    #calculate the contant U which is dependent on stepsize and u0
    dx = boxsize/Nsites
    U = u0/(2*dx)

    # kinetic energy is 1/2 * lap * phi so the matrix for the energy is .5*lap
    Tmat = .5*laplacian_mat(Nsites, boxsize)

    # the soft coulomb tensor is .5*U over the main diagonals and .5*U times the soft coulomb potential,
    # 1/sqrt(dx**2*(j-i)**2+alpha**2), along the off diagonals
    V_SC_mat = np.zeros((Nsites, Nsites, Nsites, Nsites))
    for i in range(Nsites-1):
        V_SC_mat[i, i, i, i]= 0.5*U
        for j in range(Nsites-1):
            V_SC_mat[i, j, i, j] = V_SC_mat[j, i, j, i] = 0.5*U/math.sqrt(dx**2*(j-i)**2+alpha**2)
    V_SC_mat[Nsites-1, Nsites-1, Nsites-1, Nsites-1] = 0.5*U

    # generate a symertric grid of points around 0 ranging from aproxximately -boxsise/2 to boxsize/2
    xx = np.linspace(0,1,Nsites)
    # xx = dx*(range(Nsites)-float(Nsites-1)/2*np.ones(Nsites))

    # define the external potetial due to the two nuclei of the system and their respective charges z1 and z2
    if 0.0 <= del_v <= 2.0:
        z1 = (del_v-2.0)/2
        z2 = -(del_v+2.0)/2
    else:
        print('potential difference not in the correct range! Setting both charges to 1')
        z1 = -1.0
        z2 = -1.0
    V_ext_mat = np.zeros((Nsites, Nsites))
    #for ii in range(Nsites):
    #    V_ext_mat[ii, ii] = z1/math.sqrt((xx[ii] - dist/2)**2+alpha**2) + z2/math.sqrt((xx[ii] + dist/2)**2+alpha**2) - (del_v*del_v/2.0 - 2)/math.sqrt(dist**2 + alpha**2)

    return Tmat, V_ext_mat, V_SC_mat