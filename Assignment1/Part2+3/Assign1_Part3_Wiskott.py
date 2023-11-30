#!/usr/bin/env python
"""
Created on Tue Apr  7 10:36:49 2020

@author: Christian Wiskott
"""

import numpy as np
from numpy import linalg as lin
import datetime
#from numba import jit, prange # Included in anaconda installation, uncomment if available
import copy

# -------------------------------------------------------------------------------------------
# uncomment if numba is available
'''
@jit(nopython=True, fastmath=True, parallel=True) # Speeds up the code considerably, uncomment if not available
def SubCalc(U, P, dim):
    """ LU-Calculation """
    for k in range(dim-1): # Column index
        if U[k,k] == 0:
            print("Division by 0. Please provide a non-singular matrix")
            break
        index = np.argmax(np.abs(U[k:dim,k])) + k # Index biggest element in k-th column below main diagonal
        if index != k: # If U[k,k] biggest element, no swap is needed 
            temp = np.copy(U[index,:])
            U[index,:] = U[k,:] # Swaps rows U[k,:] with U[index,:]
            U[k,:] = temp
            temp = np.copy(P[index,:])
            P[index,:] = P[k,:] # Swaps rows P[k,:] with P[index,:]
            P[k,:] = temp
        for j in prange(k+1,dim): # row index
           U[j,k] = U[j,k] / U[k,k] # Coeffs of the elimination
           for i in prange(k+1,dim): # Gauss elimination
                    U[j,i] = U[j,i] - U[j,k] * U[k,i]  
    return U, P
'''

#@jit(nopython=True, fastmath=True) # Speeds up the code considerably, uncomment if available
def SubCalc1(U, P, dim):
        """ LU-Calculation """
        for k in range(dim-1): # Column index
            if U[k,k] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
            index = np.argmax(np.abs(U[k:dim,k])) + k # Index biggest element in k-th column below main diagonal
            if index != k: # If U[k,k] biggest element, no swap is needed 
                U[[k,index]] = U[[index,k]] # Swaps rows U[k,:] with U[index,:]
                P[[k,index]] = P[[index,k]] # Swaps rows P[k,:] with P[index,:]
            for j in range(k+1,dim): # row index
                U[j,k] = U[j,k] / U[k,k] # Coeffs of the elimination
                U[j,k+1:dim] = U[j,k+1:dim] - U[j,k] * U[k,k+1:dim] # Slicing instead of the third for loop
           # Makes code around 10 times faster 
        return U, P

def LUp(dim):
    """ LU-Decomp for non-singular matrix A with partial pivoting. 
        Results are such that PA = LU. Uses only one matrix for 
        L and U in the calculation to save memory """
    
    A = np.random.uniform(90,100,(dim,dim)) # Random values from uniform distribution 
    P = np.eye(dim,dim) # Permutation matrix
    U = copy.deepcopy(A) # deepcopy to not overwrite A
    
    start = datetime.datetime.now() # Starts the timer for the main calculation
    U, P = SubCalc1(U, P, dim) # Chooses the calculation method
    runtime = (datetime.datetime.now() - start).total_seconds() * 1000 # in milliseconds
    
    L = np.tril(U); np.fill_diagonal(L,1) # Lower triangular part of U. Still neccessary to set
    # main diagonal entries to 1.
    U = np.triu(U) # Upper triangular part of U
    
    residual = lin.norm(P @ A - L @ U,1) / lin.norm(A,1) # Relative factorization error
    
    return runtime, residual

# ------------------------- Calculation ---------------------------------------------------- #
'''
data = open('LUp.dat', 'a')
data.write("#Dimension runtime residual\n")

for n in range(100,2000,100):
    a = LUp(n)
    data.write(" ".join([str(n), str((a)[0]), str((a)[1]), "\n"]))


for n in range(2000,5000,500):
    a = LUp(n)
    data.write(" ".join([str(n), str((a)[0]), str((a)[1]), "\n"]))
"""
for n in range(5000,11000,2000):
    a = LUp(n)
    data.write(" ".join([str(n), str((a)[0]), str((a)[1]), "\n"]))
"""
data.close()

'''
# ------------------- Running Time Analysis ------------------------------------------------#
'''
from scipy.optimize import curve_fit

def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

import matplotlib.pyplot as plt
import seaborn as sns

input = "LU.dat"
input1 = "LUp.dat"

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
#plt.yscale('log')
plt.xlabel('Matrix Dimension'); plt.ylabel('Elapsed Time of Computation [s]')
plt.scatter(dim, runtime, s=10)
plt.scatter(dim, runtime1, s=10)

plt.plot(x, func(x, *popt), "r", label="without pivoting", lw=1.5)
plt.plot(x, func(x, *popt1), "g", label="with pivoting", lw=1.5)

plt.legend()

plt.subplot(122)
plt.title("Residual Factorization Error")
plt.yscale('log')
plt.xlabel('Matrix Dimension'); plt.ylabel('Error')
plt.scatter(dim, res, s=10, label='without pivoting')
plt.scatter(dim, res1, s=10, label='with pivoting')

plt.legend()
plt.show()
'''