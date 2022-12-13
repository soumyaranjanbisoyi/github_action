#!/usr/bin/env python
# coding: utf-8

pip install numba


# ### Required Imports

import numpy as np  # to generate random values
from numba import jit, cuda # jit-just in time(compilation), cuda - for parallel computing
from timeit import default_timer as timer # to measure exec time


# ### normal function to run on cpu



def func(a):                                
    for i in range(10000000):
        a[i]+= 1


# ###  function optimized to run on gpu 



@jit(target_backend='cuda')  # pointing to cuda to use the GPU, @jit - just in time compilation                       
def func2(a):
    for i in range(10000000):
        a[i]+= 1


# ### running the script



if __name__=="__main__":
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
      
    start = timer()
    func(a)
    print("without GPU:", timer()-start)    
      
    start = timer()
    func2(a)
    print("with GPU:", timer()-start)

