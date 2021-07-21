import cpnest.model
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import gamma, dirichlet, beta
import numpy as np
from numba.extending import get_cython_function_address
from numba import vectorize, njit, jit
from numpy.random import randint, shuffle
from random import sample, shuffle
import matplotlib.pyplot as plt
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_sub(x, y): return x + np.log1p(-np.exp(y-x))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

class DirichletDistribution(cpnest.model.Model):
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, probs, n = 30, prior_pars = lambda x: 0, max_a = 3):
    
        super(DirichletDistribution, self).__init__()
        self.samples    = samples
        self.labels     = pars
        self.names      = pars + ['a']
        self.bounds     = bounds + [[0, max_a]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.n          = n
        self.m          = np.linspace(self.x_min, self.x_max, self.n)
        self.dm         = self.m[1] - self.m[0]
        self.model      = model
        self.probs      = np.array(probs)
    
    def log_prior(self, x):
    
        logP = super(DirichletDistribution,self).log_prior(x)
        if np.isfinite(logP):
            logP = - x['a']
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP

    def log_likelihood(self, x):

        pars = [x[lab] for lab in self.labels]
        base = np.array([self.model(mi, *pars)*self.dm for mi in self.m])
        base = base/np.sum(base)
        a = x['a']*base
        #implemented as in scipy.stats.dirichlet.logpdf() w/o checks
        lnB = np.sum([numba_gammaln(ai) for ai in a]) - numba_gammaln(np.sum(a))
        logL = np.sum([- lnB + np.sum((xlogy(a-1, p.T)).T, 0) for p in self.probs])
        return logL

class DirichletProcess(cpnest.model.Model):
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, prior_pars = lambda x: 0, max_a = 10000, max_N = 300):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.n_samps    = len(samples)
        self.labels     = pars
        self.names      = pars + ['a', 'N']
        self.bounds     = bounds + [[0, max_a], [10,max_N]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.model      = model
        self.prec_probs = {}

    
    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            logP = -np.log(x['N']) #- x['a']
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP

    def log_likelihood(self, x):
        N  = int(x['N'])
        m  = np.linspace(self.x_min, self.x_max, N)
        dm = m[1] - m[0]
        if N in self.prec_probs.keys():
            probs = self.prec_probs[N]
        else:
            probs = []
            for samp in self.samples:
                p = np.ones(N) * -np.inf
#                for component in samp.values():
#                    logW = np.log(component['weight'])
#                    mu   = component['mean']
#                    s    = component['sigma']
#                    for i, mi in enumerate(m):
#                        p[i] = log_add(p[i], logW + log_norm(mi, mu, s))
                p = samp(m)
                p = p + np.log(dm) - logsumexp(p+np.log(dm))
                probs.append(p)
            probs = np.mean(probs, axis = 0)
            probs = probs - logsumexp(probs+np.log(dm))
#            t = [shuffle(p) for p in probs.T]
#            probs = np.array([p - logsumexp(p) for p in probs])
            self.prec_probs[N] = probs
        
        pars = [x[lab] for lab in self.labels]
        base = np.array([self.model(mi, *pars)*dm for mi in m])
        base = base/(np.sum(base))
        c_par = x['a']
        a = c_par*base
#        p = np.array([probs[randint(self.n_samps), i] for i in range(N)])
#        p = p - logsumexp(p+np.log(dm))
        #implemented as in scipy.stats.dirichlet.logpdf() w/o checks
        lnB = np.sum([numba_gammaln(ai) for ai in a]) - numba_gammaln(np.sum(a))
        logL = -lnB + my_dot(a-1, probs)#np.sum([my_dot(a-1, p) for p in probs])#/self.n_samps#np.sum((xlogy(a-1, p.T)).T, 0)
#        logL = np.sum([ai*p + (c_par - ai)*log_sub(0,p) + gammaln(c_par) - gammaln(ai) - gammaln(c_par - ai) for ai, p in zip(a, probs.T)])
#        logL = np.sum([beta(ai, c_par - ai).logpdf(p) for ai, p in zip(a, probs.T)])#- lnB + my_dot(a-1, p)#np.sum((xlogy(a-1, p.T)).T, 0)
        return logL


@njit
def numba_gammaln(x):
  return gammaln_float64(x)
  
@jit
def my_dot(a,b):
    return np.sum(a*b)
