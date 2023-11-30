#!/usr/bin/env python
"""
Created on Tue Apr  7 10:36:49 2020

@author: Christian Wiskott
"""
#%%
import numpy as np
from numpy import linalg as lin
import datetime
from numba import jit, prange # Included in anaconda installation, uncomment if available
import copy

# ---------------------------------------------------------------------------------------------#

@jit(nopython=True, fastmath=True, parallel=True) # Speeds up the code considerably, uncomment if available
def SubCalc1(U, dim): # Fastest with numba, uses all CPU-cores
        """ LU-Calculation using only one matrix U. Utilizes parallelization of code to minimize runtime """
        for k in range(dim-1): # Column index
            if U[k,k] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
            for j in prange(k+1,dim): # row index, parallization apllied
                U[j,k] = U[j,k] / U[k,k] # Coeffs of the elimination
                for i in prange(k+1,dim): # Gauss elimination. parallization apllied. Skips the 
                # first element of the row, to not overwrite the U[j,k] coeffs.
                    U[j,i] = U[j,i] - U[j,k] * U[k,i]
        return U
'''
#@jit(nopython=True, fastmath=True) # Speeds up the code considerably, uncomment if available
def SubCalc(U, dim): # Fastest without numba
    """ LU-Calculation using only one matrix U plus slicing instead of three explicit for loops """
    for k in range(dim-1): # Column index
        if U[k,k] == 0:
           print("Division by 0. Please provide a non-singular matrix")
           break
        for j in range(k+1,dim): # Row index. Skips the first element of the row, to not overwrite the U[j,k] coeffs.
           U[j,k] = U[j,k] / U[k,k] # Coeffs of the elimination
           U[j,k+1:dim] = U[j,k+1:dim] - U[j,k] * U[k,k+1:dim] # Slicing instead of the third for loop
           # Makes code around 10 times faster
    return U
'''
def LU(dim):
    """ LU-Decomp for non-singular matrix A """
    
    A = np.random.uniform(90,100,(dim,dim)) # Values taken from random uniform distribution
    U = copy.deepcopy(A) # Copy of A. To not overwrite A
    
    start = datetime.datetime.now() # Starts the timer for the main calculation
    U = SubCalc1(U,dim) # Selects the chosen method of computation
    runtime = (datetime.datetime.now() - start).total_seconds() * 1000; # in milliseconds
    
    L = np.tril(U); np.fill_diagonal(L,1)# Lower triangular part of M. Still neccessary to set
    # main diagonal entries to 1.
    U = np.triu(U) # Upper triangular part of M
    
    residual = lin.norm(L @ U - A,1) / lin.norm(A,1) # Relative factorization error
    
    return runtime, residual

# ------------------- Calculation -----------------------------------------------------------#
'''

data = open('LU.dat', 'a')
data.write("#Dimension runtime residual\n")

#data1 = open('LU2.dat', 'a')
#data1.write("#Dimension runtime residual\n")

for n in range(100, 1505, 100):
    a = LU(n)
    data.write(" ".join([str(n), str((a)[0]), str((a)[1]), "\n"]))
    #data1.write(" ".join([str(n), str((b)[0]), str((b)[1]), "\n"]))

for n in range(2000, 5000, 500):
    a = LU(n)
    data.write(" ".join([str(n), str((a)[0]), str((a)[1]), "\n"]))
    #data1.write(" ".join([str(n), str((b)[0]), str((b)[1]), "\n"]))


for n in range(5000,11000,1000):
    a = LU(n)
    data.write(" ".join([str(n), str((a)[0]), str((a)[1]), "\n"]))
    #data1.write(" ".join([str(n), str((b)[0]), str((b)[1]), "\n"]))

data.close()
#data1.close()
'''

# ------------------- Running Time Analysis --------------------------------------#
'''
from scipy.optimize import curve_fit

def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

import matplotlib.pyplot as plt
import seaborn as sns

input = "LU.dat"
input1 = "LU2.dat"

dim = np.genfromtxt(input,usecols=(0))
runtime = np.genfromtxt(input,usecols=(1))/1000
res = np.genfromtxt(input,usecols=(2))

runtime1 = np.genfromtxt(input1,usecols=(1))/1000
res1 = np.genfromtxt(input1,usecols=(2))

popt, pcov = curve_fit(func, dim, runtime)
popt1, pcov1 = curve_fit(func, dim, runtime1)

x = np.arange(min(dim),max(dim)+0.1, 0.1)

sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('poster')

plt.subplot(121)
plt.title("Run-Time Analysis LU-Factorization")
plt.xlabel('Matrix Dimension'); plt.ylabel('Elapsed Time of Computation [s]')
plt.scatter(dim, runtime, s=10)
plt.scatter(dim, runtime1, s=10)

plt.plot(x, func(x, *popt), "r", label="with parallelization", lw=1.5)
plt.plot(x, func(x, *popt1), "g", label="without parallelization", lw=1.5)

plt.legend()

plt.subplot(122)
plt.title("Relative Factorization Error")
plt.yscale('log')
plt.xlabel('Matrix Dimension'); plt.ylabel('Error')
plt.scatter(dim, res, s=10, label='with parallelization')
plt.scatter(dim, res1, s=10, label='without parallelization')

plt.legend()
plt.show()
'''

''' #Alternative Methods
@jit(nopython=True, fastmath=True) # Speeds up the code considerably
def SubCalc1(L,U): # With two matrices
    """ LU-Calculation """
    for k in range(dim): # Column index
        if U[k,k] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
        for i in range(k+1,dim): # row index
            L[i,k] = U[i,k] / U[k,k]   
        for j in range(k+1,dim): # Gauß elimination
                for i in range(k+1,dim):   
                    U[i,j] = U[i,j] - L[i,k] * U[k,j]
    
    return L,U

@jit(nopython=True, fastmath=True) # Speeds up the code considerably, uncomment if not available
def SubCalc3(U, dim): # Slowest of the three, despite only using one explicit for loop
    """ LU-Calculation """
    for k in range(dim-1): # Column index
        if U[k,k] == 0:
            #print("Division by 0. Please provide a non-singular matrix")
            break
        U[k+1:dim,k] = U[k+1:dim,k] / U[k,k] # Coeffs of the elimination
        U[k+1:dim,k+1:dim] = U[k+1:dim,k+1:dim] - np.outer(U[k+1:dim,k], U[k,k+1:dim])
    
    return U
'''
# %%
