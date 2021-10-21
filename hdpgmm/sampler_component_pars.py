import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from numpy.random import uniform

def log_norm(x, x0, sigma):
    """
    Logarithm of the Gaussian distribution evaluated at x.
    
    Arguments:
        :double x:     value
        :double x0:    gaussian mean
        :double sigma: gaussian variance
    Returns:
        :double: log probability
    """
    return -((x-x0)**2)/(2*(sigma**2)) - np.log(np.sqrt(2*np.pi)) - 0.5*np.log(sigma**2)

def log_posterior(mu, sigma, events, sigma_min, sigma_max, m_min, m_max):
    '''
    Conditional probability for mean and std of a single component conditioned on assigned events. Eq. (3.7)
    
    Arguments:
        :double mu:        component mean
        :double sigma:     component standard deviation
        :list events:      event mixtures associated to the considered cluster
        :double sigma_min: lower bound for std
        :double sigma_max: upper bound for std
        :double m_min:     lower bound for mean
        :double m_max:     upper bound for mean
    Returns:
        :double: log probability
    '''
    if not (sigma_min < sigma < sigma_max and m_min < mu < m_max):
        return -np.inf
    events_sum = np.sum([logsumexp([np.log(component['weight']) + log_norm(mu, component['mean'], sigma) for component in ev.values()]) for ev in events]) - np.log(sigma)
    return events_sum

def propose_point(old_point, dm, ds):
    """
    Draw a new proposed point for MH sampling scheme
    
    Arguments:
        :list old_point: [mean, std]
        :double dm:      maximum increment on mean
        :double ds:      maximum increment on std
    Returns:
        :list: [proposed mean, proposed std]
    """
    m = old_point[0] + uniform(-1,1)*dm
    s = old_point[1] + uniform(-1,1)*ds
    return [m,s]

def sample_point(events, m_min, m_max, s_min, s_max, burnin = 1000, dm = 3, ds = 1):
    """
    Draws mean and standard deviation for a mixture component using a Metropolis-Hastings sampling scheme.
    
    Arguments:
        :list events:      event mixtures associated to the considered cluster
        :double m_min:     lower bound for mean
        :double m_max:     upper bound for mean
        :double sigma_min: lower bound for std
        :double sigma_max: upper bound for std
        :int burnin:       burnin for MH
        :double dm:        maximum increment on mean
        :double ds:        maximum increment on std
    Returns:
        :double: mean
        :double: standard deviation
    """
    old_point = [uniform(m_min, m_max), uniform(s_min, s_max)]
    for _ in range(burnin):
        new_point = propose_point(old_point, dm, ds)
        log_new = log_posterior(new_point[0], new_point[1], events, s_min, s_max, m_min, m_max)
        log_old = log_posterior(old_point[0], old_point[1], events, s_min, s_max, m_min, m_max)
        if log_new - log_old > np.log(uniform(0,1)):
            old_point = new_point
    return old_point[0], old_point[1]
