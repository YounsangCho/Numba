
# coding: utf-8



import numpy as np
from numba import njit, float64, int64, types


## Naive Python code (Univariate Normal)
### input arguments
#### X: Data
#### mu: the parameter of mean, sigma: the parameter of standard error
#### max_iter: the number of iterations for EM
#### tau: Initialization for the probability of belonging to the kth class
#### q : Initialization for the probability of belonging to the kth class given parameters and data
#### tol: tolerance of convergence

def normal_ll(X, mu, sigma):
    return np.exp(-(X-mu)**2 / (2 * sigma)) / np.sqrt(2*np.pi*sigma)



def GMM_EM(X, mu, sigma, max_iter, tau, q, tol = 1e-15):
    K = len(mu)
    n = len(X)
    
    for iteration in range(max_iter):
        for k in range(K):
            ll = normal_ll(X, mu[k], sigma[k])
            q[:, k ] = tau[k] * ll
        
        for i in range(n):
            q[i, :] /= np.sum(q[i, :])
        
        mu_before = mu
        sigma_before = sigma
        tau_before = tau
        
        for k in range(K):
            q_k = np.sum(q[:, k])
            mu[k] = np.sum(q[:, k] * X) / q_k
            sigma[k] = np.sum(q[:, k] * (X - mu[k])**2) / q_k
            tau[k]  = q_k / n
            
        mu_diff = np.max(np.abs(mu - mu_before))
        sigma_diff = np.max(np.abs(sigma-sigma_before))
        tau_diff = np.max(np.abs(tau - tau_before))
        
        diff = np.max(np.array([np.abs(mu_diff), np.abs(sigma_diff), np.abs(tau_diff)]))
        
        if ( (iteration > 1) & (diff < tol)): break
        
    return mu, sigma, tau, iteration


# ## Numba Python code (Univariate Normal)


@njit('float64[:](float64[:],float64,float64)')
def normal_ll_njit(X, mu, sigma):
    return np.exp(-(X-mu)**2 / (2 * sigma)) / np.sqrt(2*np.pi*sigma)



@njit('float64(float64[:])')
def nb_sum(x):
    res = 0.0
    for i in range(x.shape[0]):
        res += x[i]
    return res



r_sig = types.Tuple([float64[:],float64[:],float64[:],int64])
sig = r_sig(float64[:],float64[:],float64[:],int64,float64[:],float64[:,:],float64)


@njit(sig)
def GMM_EM_njit(X, mu, sigma, max_iter, tau, q, tol = 1e-15):
    K = len(mu)
    n = len(X)
    
    for iteration in range(max_iter):
        for k in range(K):
            ll = normal_ll_njit(X, mu[k], sigma[k])
            q[:, k ] = tau[k] * ll
        
        for i in range(n):
            q[i, :] /= nb_sum(q[i, :])
        
        mu_before = mu
        sigma_before = sigma
        tau_before = tau
        
        for k in range(K):
            q_k = nb_sum(q[:, k])
            mu[k] = nb_sum(q[:, k] * X) / q_k
            sigma[k] = nb_sum(q[:, k] * (X - mu[k])**2) / q_k
            tau[k]  = q_k / n
            
        mu_diff = np.max(np.abs(mu - mu_before))
        sigma_diff = np.max(np.abs(sigma-sigma_before))
        tau_diff = np.max(np.abs(tau - tau_before))
        
        diff = np.max(np.array([np.abs(mu_diff), np.abs(sigma_diff), np.abs(tau_diff)]))
        
        if ( (iteration > 1) & (diff < tol)): break
        
    return mu, sigma, tau, iteration



