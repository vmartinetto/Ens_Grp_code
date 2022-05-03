import numpy as np

vecs = np.loadtxt('vecs-1001-9-2030.dat')
vec2 = np.matrix(vecs[:,3].reshape(1001,1001))
corr1RDM = 2*np.dot(vec2.getH(),vec2)
density = np.diag(corr1RDM)
np.savetxt('denspy-1001-3-9-2030.dat', density, fmt='%.9e', delimiter=' ')
