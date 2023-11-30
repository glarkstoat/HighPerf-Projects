# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:52:36 2020
@author: Christian Wiskott
"""

import numpy as np
from numpy import linalg as lin
import datetime
from numba import jit, prange # Included in anaconda installation, uncomment if available
import scipy.linalg.blas as blas

@jit(nopython=True, fastmath=True, parallel=True) # Speeds up the code considerably
def Factorize(A):
    """ LU-Decomp for m x n - matrix """
    m = A.shape[0]; n = A.shape[1]
    
    for i in range(min(m-1,n)): # Column index
        if A[i,i] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
        for j in prange(i+1,m): # row index
            A[j,i] = A[j,i] / A[i,i]
            if i < n:
                A[j,i+1:n] = A[j,i+1:n] - A[j,i] * A[i,i+1:n]
    return A

@jit(nopython=True, fastmath=True, parallel=True) # Speeds up the code considerably, uncomment if available
def SubCalc(A,b): 
    """ Blocked LU without pivoting """
    n = len(A)
    for i in range(0,n-1,b): # Column index
        A[i:n,i:i+b] = Factorize(A[i:n,i:i+b])    
        L22 = np.tril(A[i:i+b,i:i+b]); np.fill_diagonal(L22, 1)
        A[i:i+b,i+b:n] = np.dot(lin.inv(L22),A[i:i+b,i+b:n]) # form U23
        A[i+b:n,i+b:n] = A[i+b:n,i+b:n] - np.dot(A[i+b:n,i:i+b], A[i:i+b,i+b:n]) # form Â33
    return A

def LU_Block(dim,b): # dimension of nxn-matrix and block size
    """ LU-Decomp for non-singular matrix A """
    
    A = np.random.uniform(90,100,(dim,dim)) # Values taken from random uniform distribution
    B = np.copy(A)
    
    start = datetime.datetime.now() # Starts the timer for the main calculation
    B = SubCalc(B,b)
    runtime = (datetime.datetime.now() - start).total_seconds() * 1000; # in milliseconds

    L = np.tril(B); np.fill_diagonal(L,1)# Lower triangular part of M. Still neccessary to set
    # main diagonal entries to 1.
    U = np.triu(B) # Upper triangular part of M
    
    residual = lin.norm(L @ U - A,1) / lin.norm(A,1) # Relative factorization error
            
    return runtime, residual

# ------------------------- Calculation ---------------------------------------------------- #
"""
data = open('LUBlock.dat', 'a')
data.write("#Block_Size Dimension runtime residual\n")
b = 500

for n in range(500,2000,100):
    a = LU_Block(n,b)
    data.write(" ".join([str(b), str(n), str((a)[0]), str((a)[1]), "\n"]))

for n in range(2000,5000,500):
    a = LU_Block(n,b)
    data.write(" ".join([str(b),str(n), str((a)[0]), str((a)[1]), "\n"]))

for n in range(5000,12000,2000):
    a = LU_Block(n,b)
    data.write(" ".join([str(b),str(n), str((a)[0]), str((a)[1]), "\n"]))

data.close()
"""
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