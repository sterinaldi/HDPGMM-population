cimport numpy as np
import numpy as np
from libc.math cimport log, sqrt, M_PI, exp, HUGE_VAL
cimport cython

cdef double LOGSQRT2 = log(sqrt(2*M_PI))

"""
INTERNAL METHOD
Computes log(e^(x)+e^(y)).

Arguments:
    :double x: first addend
    :double y: second addend
Returns:
    :double: sum
"""
cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_norm(double x, double x0, double sigma) nogil:
    """
    INTERNAL METHOD
    Logarithm of the Gaussian distribution evaluated at x.
    
    Arguments:
        :double x:     value
        :double x0:    gaussian mean
        :double sigma: gaussian variance
    Returns:
        :double: log probability
    """
    return -((x-x0)**2)/(2*sigma*sigma) - LOGSQRT2 - log(sigma)

def log_norm(double x, double x0, double sigma):
    """
    Logarithm of the Gaussian distribution evaluated at x.
    
    Arguments:
        :double x:     value
        :double x0:    gaussian mean
        :double sigma: gaussian variance
    Returns:
        :double: log probability
    """
    return _log_norm(x, x0, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_prob_component(double mu, double mean, double sigma, double w) nogil:
    """
    INTERNAL METHOD
    Addend of sum in Eq. (3.7).
    
    Arguments:
        :double mu:    gaussian mean
        :double mean:  mean of the event's gaussian component
        :double sigma: gaussian std
        :double w:     relative weight of the event's gaussian component
    Returns:
        :double: log probability of the component
    """
    return log(w) + _log_norm(mu, mean, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_prob_mixture(double mu, double sigma, dict ev):
    """
    INTERNAL METHOD
    Sum in Eq. (3.7).
    
    Arguments:
        :double mu:    gaussian mean
        :double sigma: gaussian variance
        :dict ev:      mixture components for the considered event
    Returns:
        :double: log probability
    """
    cdef double logP = -HUGE_VAL
    cdef dict component
    for component in ev.values():
        logP = log_add(logP,_log_prob_component(mu, component['mean'], sigma, component['weight']))
    return logP

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrand(double mu, double sigma, list events, double logN_cnst):
    """
    INTERNAL METHOD
    Integrand of Eq. (2.39), to be passed to dblquad.
    
    Arguments:
        :double sigma:     gaussian std
        :double mu:        gaussian mean
        :list events:      events assigned to cluster
        :double logN_cnst: normalisation constant
    Returns:
        :double: integrand
    """
    cdef double logprob = 0.0
    cdef dict ev
    for ev in events:
        logprob += _log_prob_mixture(mu, sigma, ev)
    return exp(logprob - logN_cnst)

def integrand(double sigma,double mu, list events, double logN_cnst):
    """
    Integrand of Eq. (2.39), to be passed to dblquad.
    
    Arguments:
        :double sigma:     gaussian std
        :double mu:        gaussian mean
        :list events:      events assigned to cluster
        :double logN_cnst: normalisation constant
    Returns:
        :double: integrand
    """
    return _integrand(mu, sigma, events, logN_cnst)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _compute_norm_const(double mu, double sigma, list events):
    """
    INTERNAL METHOD
    Compute normalisation constant to avoid underflow while integrating.
    
    Arguments:
        :double mu:    mean of the gaussian distribution
        :double sigma: variance of the gaussian distribution
        :list events:  events assigned to cluster
    Returns:
        :double: normalisation constant
    """
    cdef double logprob = 0.0
    cdef dict ev
    for ev in events:
        logprob += _log_prob_mixture(mu, sigma, ev)
    return logprob

def compute_norm_const(double mu, double sigma, list events):
    """
    Compute normalisation constant to avoid underflow while integrating.
    
    Arguments:
        :double mu:    mean of the gaussian distribution
        :double sigma: variance of the gaussian distribution
        :list events:  events assigned to cluster
    Returns:
        :double: normalisation constant
    """
    return _compute_norm_const(mu, sigma, events)
