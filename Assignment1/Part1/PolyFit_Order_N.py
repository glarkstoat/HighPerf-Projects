# -*- coding: utf-8 -*-
"""
Calculates a fit for a given data set and plots the result.
Plus deeper statistical analysis. The only parameter that should be altered is 
the order in PolyFit(). 
Everything else is automatically altered to fit the speciefied order.
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

input = "analysis_bwd.dat"
input_body = np.genfromtxt(input)

x_n = np.genfromtxt(input,usecols=(0))
y_n = np.genfromtxt(input,usecols=(1))

input1 = "analysis_fwd.dat"
input_body1 = np.genfromtxt(input1)

x_n1 = np.genfromtxt(input1,usecols=(0))
y_n1 = np.genfromtxt(input1,usecols=(1))

#x_n=np.array([765.11, 812.8, 705.32, 616.5]); y_n=np.array([218.94, 192.51, 186.97, 150.98])

def PolyFit(Array1,Array2, order):
    """ Calculates a polynomial fit of a specified order for a given data set. 
    Solution via solving the normal equations """
    m = len(Array1)
    A = np.ones(m) # Creates the foundation of the needed matrix A.

    for i in range(1,order+1): # Biulds the neccessary columns relative to 
                                # the given order ontop of the foundation.
        A = np.column_stack((A, Array1**i)) # Every loop adds another column 
                                            # to the matrix A
    
    B = np.dot(A.T, A); b= np.dot(A.T, Array2) # Following procedure to 
    # solve the normal equations
    BI = np.linalg.inv(B)
    alpha = np.dot(BI, b)

    return alpha, order # Optimized values for polynomial

alpha, order = PolyFit(x_n, y_n, 2) # Order is needed for future use.
alpha1, order1 = PolyFit(x_n1, y_n1, 2)

# Creates a model polynomial function 
def PolyGenerator(x,*alpha):
    f=0
    for i in range(order+1): # Every loop adds the next needed part of the 
                            # polynomial in respect to the given order.
        f += alpha[i]*x**(i) # for expample: order 3: a+bx+cx^2+dx^3
    return f # Only use if x is a single value. Doesn't work for arrays, 
            # except if used in a loop (where every x is element of array)! 
            # For arrays change x to x[i]. 
def ExpFit(x):
    return np.exp(x)

# The values x that will be used for the plot of the model function f(x) 
# calculated by PolyGenerator.Constructed so that the plot looks nice and 
# even. Otherwise it will get bumpy!
x = np.arange(np.min(x_n)*-1, np.max(x_n)+0.1, 0.01)
x1 = np.arange(np.min(x_n1)*-1, np.max(x_n1)+0.1, 0.01)

# Plot
sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('notebook')

plt.subplot(121)
plt.plot(x, PolyGenerator(x, *alpha),"r", label="Fit line")
""" The * iterates trough the values of alpha """
plt.scatter(x_n, y_n, label="Data Points", s=15)
plt.title("Running Time Analysis of Backward Substitution")
plt.xlabel('Matrix Dimension'); plt.ylabel('Computation Time [ms]')
plt.legend()

plt.subplot(122)
plt.plot(x1, PolyGenerator(x1, *alpha1),"r", label="Fit line")
""" The * iterates trough the values of alpha """
plt.scatter(x_n1, y_n1, label="Data Points", s=15)
plt.title("Running Time Analysis of Forward Substitution")
plt.xlabel('Matrix Dimension'); plt.ylabel('Computation Time [ms]')
plt.legend()

plt.show()

#print(PolyGenerator(1,*alpha)) # Returns value of y(x) at position x

""" 
# Omitting biggest outlier
dif = np.absolute(y_n-PolyGenerator(x_n, *alpha))
maximum = np.max(dif)
for i in range(len(y_n)-1):
    if(np.absolute(y_n[i]-PolyGenerator(x_n, *alpha)[i])==maximum):
        y_n = np.delete(y_n,i)
        x_n = np.delete(x_n,i)

alpha1 = PolyFit(x_n, y_n, 2) # Parameters for excluded outliner 

plt.plot(x, PolyGenerator(x, *alpha1), "g", label="Fit-Outliner")
"""
# %%
