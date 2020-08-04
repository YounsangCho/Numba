
# coding: utf-8

import numpy as np
from numba import njit, float64, int64, types

## Naive Python code (Multivariate Normal)
### input arguments
#### X: Data
#### mu: the parameter of mean, cov: the parameter of covariance matrix
#### max_iter: the number of iterations for EM
#### tau: Initialization for the probability of belonging to the kth class
#### q : Initialization for the probability of belonging to the kth class given parameters and data
#### tol: tolerance of convergence

def multi_ll(X, mu, cov):
    n = X.shape[0]
    p = X.shape[1]
    res = np.zeros(n)
    
    for i in range(n):
        exp_inter = np.dot(np.dot((X[i, :] - mu).T, np.linalg.inv(cov)), 
                           (X[i, :] - mu)) / 2.
        res[i] = (2*np.pi)**(-p/2) * np.linalg.det(cov)**(-0.5)*np.exp(-exp_inter)
    
    return res


def GMM_EM_multi(X, mu, cov, max_iter, tau, q, tol = 1e-08):
    n = X.shape[0]
    p = X.shape[1]
    K = mu.shape[0]
    
    for iteration in range(max_iter):
        for k in range(K):
            ll = multi_ll(X, mu[k, :], cov[:, :, k])
            q[:, k] = tau[k] * ll
            
        for i in range(n):
            q[i, :] /= np.sum(q[i, :])
        
        mu_before = mu
        cov_before = cov
        tau_before = tau
        
        for k in range(K):
            q_k = np.sum(q[:, k])
            mu[k, :] = np.sum(q[:, k].reshape(n,1)*X, axis=0) / q_k
            cov[:, :, k] = np.dot((q[:, k].reshape(n,1) * (X - mu[k, :])).T, 
                                 (X - mu[k, :])) / q_k
            tau[k] = q_k / n
        
        mu_diff = np.max(np.abs(mu - mu_before))
        cov_diff = np.max(np.abs(cov - cov_before))
        tau_diff = np.max(np.abs(tau - tau_before))
        diff = np.max(np.array([np.abs(mu_diff), np.abs(cov_diff), np.abs(tau_diff)]))
        
        if ( (iteration > 1) & (diff < tol)):
            break
        
    return mu, cov, tau, iteration


# ## Numba Python code (Multivariate Normal)


@njit('float64[:](float64[:,:],float64[:],float64[:,:])')
def multi_ll_njit(X, mu, cov):
    n = X.shape[0]
    p = X.shape[1]
    res = np.zeros(n)
    
    for i in range(n):
        exp_inter = np.dot(np.dot((X[i, :] - mu).T, np.linalg.inv(cov)), 
                           (X[i, :] - mu)) / 2.
        res[i] = (2*np.pi)**(-p/2) * np.linalg.det(cov)**(-0.5)*np.exp(-exp_inter)
    
    return res



@njit('float64(float64[:])')
def nb_sum(X):
    res = 0.0
    for i in range(X.shape[0]):
        res += X[i]
    
    return res


r_sig = types.Tuple([float64[:,:],float64[:,:,:],float64[:],int64])
sig = r_sig(float64[:,:],float64[:,:],float64[:,:,:],int64,float64[:],float64[:,:],float64)


@njit(sig)
def GMM_EM_multi_njit(X, mu, cov, max_iter, tau, q, tol = 1e-08):
    n = X.shape[0]
    p = X.shape[1]
    K = mu.shape[0]
    
    for iteration in range(max_iter):
        for k in range(K):
            ll = multi_ll_njit(X, mu[k, :], cov[:, :, k])
            q[:, k] = tau[k] * ll
            
        for i in range(n):
            q[i, :] /=  nb_sum(q[i, :])
        
        mu_before = mu
        cov_before = cov
        tau_before = tau
        
        for k in range(K):
            q_k = nb_sum(q[:, k])
            q = np.ascontiguousarray(q[:, k]).reshape(n,1)
            mu[k, :] = np.sum(q * X, axis = 0) / q_k
            cov[:, :, k] = np.dot((q*(X - mu[k, :])).T, (X - mu[k, :])) / q_k
            tau[k] = q_k / n
        
        mu_diff = np.max(np.abs(mu - mu_before))
        cov_diff = np.max(np.abs(cov - cov_before))
        tau_diff = np.max(np.abs(tau - tau_before))
        
        diff = np.max(np.array([np.abs(mu_diff), np.abs(cov_diff), np.abs(tau_diff)]))
        
        if ( (iteration > 1 ) & (diff < tol)):
            break
        
    return mu, cov, tau, iteration


