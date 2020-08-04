
# coding: utf-8


import numpy as np
from numba import njit, float64, int64


## A function for estimating the null distribution with Naive Python code
### input arguments
#### mu1, mu2: the parameters of means
#### n: the size of each sample
#### iteration: the numbers of iterations for estimating the null distributions

def est_null(mu1, mu2, n, iteration = 10000):
    np.random.seed(42)
    
    x = np.random.normal(loc = mu1, size = n)
    y = np.random.normal(loc = mu2, size = n)
    
    null_dist = np.zeros(iteration)
    xy = np.r_[x,y]
    
    for i in range(iteration):
        temp = np.random.permutation(xy)
        temp_x, temp_y = temp[:n] , temp[n:]
        null_dist[i] = np.mean(temp_x) - np.mean(temp_y)
    
    return null_dist


## A function for estimating the null distribution with just-in-time eager compilation 
### input arguments
#### x, y: the one dimensional vectors of two samples
#### xy: an integration of x and y

@njit('float64[:](float64[:],float64[:],float64[:],int64)')
def njit_est_null(x, y, xy, iteration = 10000):
    n = x.shape[0]
    null_dist = np.zeros(iteration)
    
    np.random.seed(42)
    for i in range(iteration):
        temp = np.random.permutation(xy)
        temp_x, temp_y = temp[:n], temp[n:]
        null_dist[i] = np.mean(temp_x) - np.mean(temp_y)
        
    return null_dist

## A function for estimating the null distribution with just-in-time lazy compilation 


@njit
def njit_est_null_lazy(x, y, xy, iteration = 10000):
    n = x.shape[0]
    null_dist = np.zeros(iteration)
    
    np.random.seed(42)
    for i in range(iteration):
        temp = np.random.permutation(xy)
        temp_x, temp_y = temp[:n], temp[n:]
        null_dist[i] = np.mean(temp_x) - np.mean(temp_y)
        
    return null_dist


## Compute p-value given the computed null distribution
### input arguments
#### null_dist: the computed null distributions by function that "est_null" or "njit_est_null" or "njit_est_null_lazy"

def compute_pvalue(mu1, mu2, n, null_dist, iteration = 10000):
    
    pvalue_vec = np.zeros(iteration)
    
    for i in range(iteration):
        x = np.random.normal(loc = mu1, size = n)
        y = np.random.normal(loc = mu2, size = n)
    
        diff_value = np.mean(x) - np.mean(y)
        pvalue_vec[i] = 1.0 - np.mean(np.abs(diff_value) > np.abs(null_dist))
        
    return pvalue_vec




