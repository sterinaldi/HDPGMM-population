import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from numpy.random import uniform
from utils import integrand

'''
from https://codereview.stackexchange.com/questions/107094/create-symmetrical-matrix-from-list-of-values
'''
def make_sym_matrix(n,vals):
  m = np.zeros([n,n], dtype=np.double)
  xs,ys = np.triu_indices(n,k=1)
  m[xs,ys] = vals
  m[ys,xs] = vals
  m[ np.diag_indices(n) ] = 0 - np.sum(m, 0)
  return m

def log_posterior(mu, cov, events, sigma_min, sigma_max, m_min, m_max):
    events_sum = np.sum([logsumexp([np.log(component['weight']) + log_norm(mu, component['mean'], cov) for component in ev.values()]) for ev in events])
    return events_sum

def propose_point(old_point, dm, ds, lower_bound, upper_bound, s_min, s_max, dim):
    new_point = np.zeros(len(old_point))
    # mean
    for i, o in enumerate(old_point[:dim]):
        t = o + dm*uniform(-1,1)
        if lower_bound[i] < t < upper_bound[i]:
            new_point[i] = t
        else:
            new_point[i] = o
    # sigma
    for i, o in enumerate(old_point[dim:2*dim+1]):
        t = o + ds*uniform(-1,1)
        if s_min < t < s_max:
            new_point[i+dim] = t
        else:
            new_point[i+dim] = o
    # rho
    for i, o in enumerate(old_point[2*dim+1:]):
        t = o + 0.2*uniform(-1,1)
        if -1 < t < 1:
            new_point[2*dim+1 + i] = t
        else:
            new_point[2*dim+1 + i] = o
    return new_point

def sample_point(events, lower_bound, upper_bound, s_min, s_max, n_dim, starting_point, burnin = 4000, dm = 1):
    old_point = starting_point
    for _ in range(burnin):
        new_point = propose_point(old_point, dm, sigma_max/4., lower_bound, upper_bound, s_min, s_max, dim)
        log_new = integrand(new_point, events, 0, n_dim)
        log_old = integrand(old_point, events, 0, n_dim)
        if log_new - log_old > np.log(uniform(0,1)):
            old_point = new_point
            
    mu       = old_point[:n_dim]
    sigma    = old_point[n_dim:2*dim+1]
    rho      = old_point[2*n_dim+1:]
    norm_cov = np.identity(n_dim)*0.5*sigma*sigma
    
    for i in range(n_dim):
        for j in range(n_dim):
            if i < j:
                norm_cov[i,j] = rho[dim*i + (j-i)]*sigma[i]*sigma[j]
    cov = norm_cov + norm_cov.T
    return mu, cov

def MH_single_event(p, upper_bound, lower_bound, len, burnin = 1000, thinning = 100):
    old_point = [(m_max + m_min)/2 for m_min, m_max in zip(lower_bound, upper_bound)]
    delta = (upper_bound - lower_bound)/15
    chain = []
    for _ in range(burnin + thinning*len):
        new_point = [o + dm*uniform(-1,1) for o, dm in zip(old_point, delta)]
        try:
            p_new = np.log(p(new_point))
        except:
            p_new = -np.inf
        p_old = np.log(p(old_point))
        if p_new - p_old > np.log(uniform(0,1)):
            old_point = new_point
        chain.append(old_point)
    return chain[burnin::thinning]
