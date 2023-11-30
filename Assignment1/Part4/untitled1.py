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
'''
def Factorize(A):
    """ LU-Decomp for m x n - matrix """
    m = A.shape[0]; n = A.shape[1]
    
    for i in range(min(m-1,n)): # Column index
        if A[i,i] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
        for j in range(i+1,m): # row index
            A[j,i] = A[j,i] / A[i,i]
            if i < n:
                A[j,i+1:n] = A[j,i+1:n] - A[j,i] * A[i,i+1:n]
    return A
'''
#@jit(nopython=True, fastmath=True, parallel=True) # Speeds up the code considerably, uncomment if available
def SubCalc(A,b): 
    """ Blocked LU without pivoting """
    n = len(A)
    for i in range(0,n-1,b): # Column index
        A[i:n,i:i+b] = Factorize(A[i:n,i:i+b]) # scipy.linalg.lu() stattdessen???
        L22 = np.tril(A[i:i+b,i:i+b]); np.fill_diagonal(L22, 1)
        A[i:i+b,i+b:n] = blas.dtrsm(1,L22,A[i:i+b,i+b:n],lower=1) # 
        A[i+b:n,i+b:n] = A[i+b:n,i+b:n] - np.dot(A[i+b:n,i:i+b], A[i:i+b,i+b:n]) # form Â33
    return A

def LU_Block1(dim,b): # dimension of nxn-matrix and block size
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
'''
data = open('LUBlock.dat', 'a')
#data.write("\n#Block_Size Dimension runtime residual\n")
b = 10
"""
for n in range(100,2000,100):
    a = LU_Block(n,b)
    data.write(" ".join([str(b), str(n), str((a)[0]), str((a)[1]), "\n"]))

for n in range(2000,5200,500):
    a = LU_Block(n,b)
    data.write(" ".join([str(b),str(n), str((a)[0]), str((a)[1]), "\n"]))
"""
for n in range(5000,12000,2000):
    a = LU_Block(n,b)
    data.write(" ".join([str(b),str(n), str((a)[0]), str((a)[1]), "\n"]))

data.close()
'''
# ------------------- Running Time Analysis ------------------------------------------------#
'''
from scipy.optimize import curve_fit

def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func1(x, a, b):
    return a * x + b 

import matplotlib.pyplot as plt
import seaborn as sns

input = "LUBlock.dat"
input1 = "LU.dat"
input2 = "LUp.dat"

dim = np.genfromtxt(input,usecols=(1))[:29]
runtime = np.genfromtxt(input,usecols=(2))/1000 # in seconds
res = np.genfromtxt(input,usecols=(3))
n = 29
block = np.genfromtxt(input,usecols=(0))[144:212]

dim1 = np.genfromtxt(input1,usecols=(0))
runtime1 = np.genfromtxt(input1,usecols=(1))/1000
res1 = np.genfromtxt(input1,usecols=(2))

runtime2 = np.genfromtxt(input2,usecols=(1))/1000
res2 = np.genfromtxt(input2,usecols=(2))

popt11, pcov = curve_fit(func, dim1, runtime1)
popt12, pcov1 = curve_fit(func, dim1, runtime2)

x = np.arange(min(dim1),max(dim1)+0.1, 0.1)

popt1, pcov = curve_fit(func, dim, runtime[n:2*n])# b=25
popt2, pcov = curve_fit(func, dim, runtime[2*n:3*n]) # b=50
popt3, pcov = curve_fit(func, dim, runtime[3*n:4*n]) # b=100
popt4, pcov = curve_fit(func, dim[1:], runtime[116:144]) # b=200
popt5, pcov = curve_fit(func, block, runtime[144:212]) # b=25
popt6, pcov = curve_fit(func, dim, runtime[212:]) # b=10

x = np.arange(min(dim),max(dim)+0.1, 0.1)
x1 = np.arange(200,11000+0.1, 0.1)
x2 = np.arange(115,300+0.1, 0.1) # Optimal blockzise

sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('poster')

plt.subplot(131)
plt.title("Run-Time Analysis Block LU-Factorization")
plt.xlabel('Matrix Dimension'); plt.ylabel('Elapsed Time of Computation [s]')
plt.scatter(dim, runtime[n:2*n],c="g", s=5) # b=25
plt.scatter(dim, runtime[2*n:3*n], c="y",s=5) # b=50
plt.scatter(dim, runtime[3*n:4*n],c="r", s=5) # b=100
plt.scatter(dim[1:], runtime[116:144],c="black", s=5) # b=200
plt.scatter(dim1, runtime1, s=5, c="b") # LU
plt.scatter(dim1, runtime2, s=5, c="r") # LUP
plt.scatter(dim, runtime[212:241],c="c", s=5) # b=10

plt.plot(x, func(x, *popt12), "r", label="LUp", lw=1.5)
plt.plot(x, func(x, *popt11), "b", label="LU", lw=1.5)
plt.plot(x1, func(x1, *popt6), "c", label="b=10", lw=1.5)
plt.plot(x, func(x, *popt1), "g", label="b=25", lw=1.5)
plt.plot(x, func(x, *popt2), "y", label="b=50", lw=1.5)
plt.plot(x, func(x, *popt3), "m", label="b=100", lw=1.5)
plt.plot(x1, func(x1, *popt4), "black", label="b=200", lw=1.5)

plt.legend(prop={'size': 15})


plt.subplot(132)
plt.title("Residual Factorization Error")
plt.yscale('log')
plt.xlabel('Matrix Dimension'); plt.ylabel('Error')
plt.scatter(dim[1:], res[116:144],c="black", s=5, label='b=200')
plt.scatter(dim, res[3*n:4*n],c="m", s=5, label='b=100')
plt.scatter(dim, res[2*n:3*n],c="y", s=5, label='b=50')
plt.scatter(dim, res[n:2*n], c="g", s=5, label='b=25')
plt.scatter(dim, res[212:241],c="b", s=5, label='b=10') # b=10
plt.scatter(dim1, res1, s=5, c="c", label='LU')
plt.scatter(dim1, res2, s=5, c="r", label='LUp')

plt.legend(prop={'size': 15},ncol=3)


plt.subplot(133)
plt.scatter(block, runtime[144:212], s=5)
plt.plot(x2, func(x2, *popt5), "g", label="Minimum", lw=1.5)
plt.xlabel('Block Size'); plt.ylabel('Elapsed Time of Computation [s]')
plt.title("Optimal Block Size for N=10,000")

# ------------
fig = plt.figure()


input1 = "LU2_server.txt"
input2 = "LUp_server.txt"
input3 = "LUBlock_server.txt"

dim1 = np.genfromtxt(input1,usecols=(0))
runtime1 = np.genfromtxt(input1,usecols=(1))/1000
res1 = np.genfromtxt(input1,usecols=(2))

runtime2 = np.genfromtxt(input2,usecols=(1))/1000
res2 = np.genfromtxt(input2,usecols=(2))

runtime3 = np.genfromtxt(input3,usecols=(1))/1000
res3 = np.genfromtxt(input3,usecols=(2))

popt11, pcov = curve_fit(func, dim1, runtime1)
popt12, pcov1 = curve_fit(func, dim1, runtime2)
popt14, pcov1 = curve_fit(func, dim1, runtime3[109:126])
popt15, pcov1 = curve_fit(func, dim1, runtime3[126:]) # b=10

x = np.arange(min(dim1),max(dim1)+0.1, 0.1)

sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('poster')

plt.subplot(131)
plt.title("Run-Time LU-Factorization Server")
plt.xlabel('Matrix Dimension'); plt.ylabel('Elapsed Time of Computation [s]')
plt.scatter(dim1, runtime1, s=5,c="r")
plt.scatter(dim1, runtime2, s=5, c="g")
plt.scatter(dim1, runtime3[126:], c="c", s=5) # b=10
plt.scatter(dim1, runtime3[109:126], c="black", s=5) # b=60


plt.plot(x, func(x, *popt11), "r", label="without pivoting", lw=1.5)
plt.plot(x, func(x, *popt12), "g", label="with pivoting", lw=1.5)
plt.plot(x, func(x, *popt15), "c", label="Blocksize=10", lw=1.5)
plt.plot(x, func(x, *popt14), "black", label="Blocksize=62", lw=1.5)

plt.legend(prop={'size': 15})

plt.subplot(132)
plt.title("Comparison Residual Factorization Error")
plt.yscale('log')
plt.xlabel('Matrix Dimension'); plt.ylabel('Error')
plt.scatter(dim1, res3[109:126],c="black", s=5, label='Blocksize=62')
plt.scatter(dim1, res3[126:],c="c", s=5, label='Blocksize=10')
plt.scatter(dim1, res1,c="r", s=5, label='without pivoting')
plt.scatter(dim1, res2, c="g", s=5, label='with pivoting')

plt.legend(prop={'size': 15})

block = np.genfromtxt(input3,usecols=(3))[81:109]
x3 = np.arange(30,105+0.1, 0.1) # Optimal blockzise
popt13, pcov1 = curve_fit(func, block, runtime3[81:109])

plt.subplot(133)
plt.scatter(block, runtime3[81:109], s=5)
plt.plot(x3, func(x3, *popt13), "g", lw=1.5)
plt.xlabel('Block Size'); plt.ylabel('Elapsed Time of Computation [s]')
plt.title("Optimal Block Size for N=1500")




plt.show()

'''