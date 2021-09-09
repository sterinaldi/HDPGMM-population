import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from numpy.random import uniform

def log_norm(x, x0, sigma1, sigma2):
    return -((x-x0)**2)/(2*(sigma1**2)) - np.log(np.sqrt(2*np.pi)) - 0.5*np.log(sigma1**2)

def log_posterior(mu, sigma, events, sigma_min, sigma_max, m_min, m_max):
    if not (sigma_min < sigma < sigma_max and m_min < mu < m_max):
        return -np.inf
    events_sum = np.sum([logsumexp([np.log(component['weight']) + log_norm(mu, component['mean'], sigma, component['sigma']) for component in ev.values()]) for ev in events]) - np.log(sigma)
    return events_sum

def propose_point(old_point, dm, ds):
    m = old_point[0] + uniform(-1,1)*dm
    s = old_point[1] + uniform(-1,1)*ds
    return [m,s]

def sample_point(events, m_min, m_max, s_min, s_max, burnin = 1000, dm = 3, ds = 1):
    old_point = [uniform(m_min, m_max), uniform(s_min, s_max)]
    for _ in range(burnin):
        new_point = propose_point(old_point, dm, ds)
        log_new = log_posterior(new_point[0], new_point[1], events, s_min, s_max, m_min, m_max)
        log_old = log_posterior(old_point[0], old_point[1], events, s_min, s_max, m_min, m_max)
        if log_new - log_old > np.log(uniform(0,1)):
            old_point = new_point
    return old_point[0], old_point[1]
