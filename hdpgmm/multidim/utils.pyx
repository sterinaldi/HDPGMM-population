from __future__ import division
cimport numpy as np
import numpy as np
from libc.math cimport log, sqrt, M_PI, exp, HUGE_VAL, atan2, acos, sin, cos
cimport cython
from numpy.linalg import det, inv
from scipy.stats import multivariate_normal as mn

cdef double LOGSQRT2 = log(sqrt(2*M_PI))

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double _scalar_log_norm(double x, double x0, double s) nogil: return -(x-x0)*(x-x0)/(2*s*s) - LOGSQRT2 - 0.5*log(s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_norm(np.ndarray x, np.ndarray x0, np.ndarray sigma):
    try:
        logP = mn(mean = x0, cov = sigma).logpdf(x)
    except:
        logP = -np.inf
    return logP
    #cdef np.ndarray diff = x-x0
    #cdef double D = det(sigma)
    #return -np.dot(diff.T, np.dot(inv(sigma), diff)) -n*0.5*LOGSQRT2 -0.5*log(D)


#solo un alias per scipy.stats.multivariate_normal
def log_norm(np.ndarray x, np.ndarray x0, np.ndarray sigma):
    return mn(mean = x0, cov = sigma).logpdf(x)
    #return _log_norm(x, x0, sigma, n)

def scalar_log_norm(double x, double x0, double s):
    return _scalar_log_norm(x,x0,s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_prob_component(np.ndarray mu, np.ndarray mean, np.ndarray sigma, double w):
    return log(w) + _log_norm(mu, mean, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_prob_mixture(np.ndarray mu, np.ndarray sigma, dict ev):
    cdef double logP = -HUGE_VAL
    cdef dict component
    for component in ev.values():
        logP = log_add(logP,_log_prob_component(mu, component['mean'], sigma, component['weight']))
    return logP

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrand(double[:] mean, double[:,:] covariance, list events, unsigned int dim):
    cdef unsigned int i,j
    cdef double logprob = 0.0
    cdef dict ev
    for ev in events:
        logprob += _log_prob_mixture(mean, covariance, ev)
    return logprob

def integrand(double[:] mean, double[:,:] covariance, list events, unsigned int dim):
    return _integrand(mean, covariance, events, dim)
