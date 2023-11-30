#!/usr/bin/env python
"""
Created on Tue Apr  7 10:36:49 2020

@author: Christian Wiskott
"""
#%% 
import numpy as np
from numpy import linalg as lin
import datetime
from numba import jit # Included in anaconda installation, uncomment if not available

# ------------ Backward Substitution -----------------------------------

@jit(nopython=True, fastmath=True) # Speeds up the code considerably, uncomment if not available
def SubCalcB(U,b,x): 
        """ Backward Substitution """
        for i in range(len(U)-1,-1,-1): # Row index
            if U[i,i] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
            temp=0
            for j in range(i+1,len(U)): # Column index
                temp += U[i,j] * x[j]
            x[i] = (b[i] - temp) / U[i,i]

        return x # Solution of Ux=b

def BackSub(dim): 
    """ Calculates solution x for system U*x = b with upper triangular matrix U of a given dimension """
    
    A = np.random.uniform(90,100,(dim,dim)) # Random entries from uniform sample
    x_exact = np.ones(dim) # Analytic solution of lin. system
    U = np.triu(A) # Upper triangular part of A
    x = np.zeros(len(U)) # Building the computed solution vector
    b = np.dot(U,x_exact) # Ux = b
    
    start = datetime.datetime.now() # Starts the timer
    x = SubCalcB(U,b,x) # Calculation of x
    runtime = (datetime.datetime.now() - start).total_seconds()*1000; # in milliseconds 

    fwd_error = lin.norm(x-x_exact,1)/lin.norm(x_exact,1) # Relative forward error
    residual = lin.norm(np.dot(U,x)-b,1)/lin.norm(b,1) # Relative residual norm
    
    return x, runtime, fwd_error, residual

# ------------ Forward Substitution -----------------------------------

@jit(nopython=True, fastmath=True) # Speeds up the code considerably, uncomment if not available
def SubCalcF(L,b,x):
        """ Forward Substitution """
        for i in range(len(L)): # Row index
            if L[i,i] == 0:
                print("Division by 0. Please provide a non-singular matrix")
                break
            temp=0
            for j in range(i): # Column index
                temp += L[i,j] * x[j]
            x[i] = (b[i] - temp) / L[i,i]
        
        return x # Solution of Lx=b

def FwdSub(dim): 
    """ Calculates solution x for system L*x = b with lower triangular matrix L of a given dimension """

    A = np.random.uniform(90,100,(dim,dim)) # Random entries from uniform sample
    x_exact = np.ones(dim) # Analytic solution of lin. system
    L = np.tril(A) # Lower triangular part of A
    x = np.zeros(len(L)) # Building the computed solution vector
    b = np.dot(L,x_exact) # Lx = b
    
    start = datetime.datetime.now() # Starts the timer
    x = SubCalcF(L,b,x) # Calculation of x
    runtime = (datetime.datetime.now() - start).total_seconds()*1000; # in milliseconds 
    
    fwd_error = lin.norm(x-x_exact,1)/lin.norm(x_exact,1) # Relative forward error
    residual = lin.norm(np.dot(L,x)-b,1)/lin.norm(b,1) # Relative residual norm
    
    return x, runtime, fwd_error, residual

# -------------------- Computation --------------------------------


data = open('analysis_bwd.dat', 'w')
data1 = open('analysis_fwd.dat', 'w')
data.write("#Dimension runtime fwd_error residual\n")
data1.write("#Dimension runtime fwd_error residual\n")

for n in range(100,1100,100):
    a = BackSub(n); b = FwdSub(n)
    data.write(" ".join([str(n), str((a)[1]), str((a)[2]), str((a)[3]), "\n"]))
    data1.write(" ".join([str(n), str((b)[1]), str((b)[2]), str((b)[3]), "\n"]))
"""
for n in range(10000,20500,500):
    a = BackSub(n); b = FwdSub(n)
    data.write(" ".join([str(n), str((a)[1]), str((a)[2]), str((a)[3]), "\n"]))
    data1.write(" ".join([str(n), str((b)[1]), str((b)[2]), str((b)[3]), "\n"]))
"""
data.close()
data1.close()


# ------------------- Running Time Analysis -----------------------

import matplotlib.pyplot as plt
import seaborn as sns

input = "analysis_bwd.dat"

dim = np.genfromtxt(input,usecols=(0))
fwd_err = np.genfromtxt(input,usecols=(2))
res = np.genfromtxt(input,usecols=(3))

sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('poster')


plt.subplot(221)
plt.title("Backward Substitution")
plt.yscale('log')
plt.xlabel('Matrix Dimension');# plt.ylabel('Elapsed Time of Computation [s]')
plt.scatter(dim, fwd_err, s=10, label="Relative Forward Error")
plt.legend()

plt.subplot(222)
plt.title("Backward Substitution")
plt.yscale('log')
plt.xlabel('Matrix Dimension'); #plt.ylabel('')
plt.scatter(dim, res, s=10, label="Relative Residual Error")
plt.legend()

input1 = "analysis_fwd.dat"

dim = np.genfromtxt(input1,usecols=(0))
fwd_err = np.genfromtxt(input1,usecols=(2))
res = np.genfromtxt(input1,usecols=(3))

plt.subplot(223)
plt.yscale('log')
plt.title("Forward Substitution")
plt.xlabel('Matrix Dimension');# plt.ylabel('Elapsed Time of Computation [s]')
plt.scatter(dim, fwd_err, s=10, label="Relative Forward Error")
plt.legend()

plt.subplot(224)
plt.yscale('log')
plt.title("Forward Substitution")
plt.xlabel('Matrix Dimension'); #plt.ylabel('')
plt.scatter(dim, res, s=10, label="Relative Residual Error")

plt.legend()
plt.show()

# %%
