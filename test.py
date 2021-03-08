import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm
import scipy as sp
from numpy.linalg import inv

# Define box size
L = 10
# Define number of GridPoints
Nx = 100
# Define the grid
x = np.linspace(-L/2, L/2, Nx, endpoint = False) 
# This reads "I want Nx points equally distributed"..
