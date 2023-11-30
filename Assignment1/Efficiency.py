# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:54:22 2020

@author: f
"""

import numpy as np
from numpy import linalg as lin
import datetime
from numba import jit, prange # Included in anaconda installation, uncomment if available

peak = 8.8e+9 # Flops / s

def flops_LU(n):
    return (2/3) * n**3

def flops_sub(n):
    return n**2

def efficiency(flops,t):
    return flops / t / (peak)

import matplotlib.pyplot as plt
import seaborn as sns

input1b = "bwd_server.dat"
input1f = "fwd_server.dat"

fig = plt.figure()
plt.subplot(121)

sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('poster')

dim1b = np.genfromtxt(input1b,usecols=(0)); dim1f = np.genfromtxt(input1f,usecols=(0)) # Back & Fwd - Sub
runtime1b = np.genfromtxt(input1b,usecols=(1))/1000 ;runtime1f = np.genfromtxt(input1f,usecols=(1))/1000 # seconds

plt.title("Efficiency Plot of Back- & Forward Substitution", fontweight='bold')
plt.xlabel('Matrix Dimension'); plt.ylabel('Efficiency %')

plt.plot(dim1b, efficiency(flops_sub(dim1b),runtime1b)*100, label="Backward", lw=1.5, c='brown', ls='-')
plt.plot(dim1f, efficiency(flops_sub(dim1f),runtime1f)*100, label="Forward", lw=1.5, c='g', ls='-')
plt.legend()

# -------------------------

input2 = "LU_server.txt"
input3 = 'LUp_server.txt'
input4 = "LUBlock_server.txt"

#fig = plt.figure()
plt.subplot(122)
sns.set(color_codes=True) # Nice layout.
sns.set_style('darkgrid')
sns.set_context('poster')

dim2 = np.genfromtxt(input2,usecols=(0)); dim3 = np.genfromtxt(input3,usecols=(0)) # LU & LUp
dim4 = np.genfromtxt(input4,usecols=(0))
runtime2 = np.genfromtxt(input2,usecols=(1))/1000 ;runtime3 = np.genfromtxt(input3,usecols=(1))/1000 # seconds
runtime4 = np.genfromtxt(input4,usecols=(1))/1000

plt.title("Efficiency Plot of Blocked/Unblocked LU", fontweight='bold')
plt.xlabel('Matrix Dimension'); plt.ylabel('Efficiency %')

plt.plot(dim2, efficiency(flops_LU(dim2),runtime2)*100, label="LU", lw=1.5, c='y', ls='-.')
plt.plot(dim3, efficiency(flops_LU(dim3),runtime3)*100, label="LU with pivoting", lw=1.5, c='r', ls='--')
plt.plot(dim4, efficiency(flops_LU(dim4),runtime4)*100, label="Blocked LU", lw=1.5, c='orange', ls='-')

plt.legend()
