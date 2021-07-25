import cpnest.model
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import gamma, dirichlet, beta
import numpy as np
from numba.extending import get_cython_function_address
from numba import vectorize, njit, jit, prange
from numpy.random import randint, shuffle
from random import sample, shuffle
import matplotlib.pyplot as plt
import ctypes
from multiprocessing import Pool

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@jit
def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_sub(x, y): return x + np.log1p(-np.exp(y-x))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

def my_betaln(x, logx, a1, a0):
    return gammaln(a1 + a0) - gammaln(a1) - gammaln(a0) + (a1-1)*logx + (a0-1)*log1p(-x)

#@jit
#def integrand(logq, m, s2, n, a, nu, p):
#    """
#    logq: logaritmo di ^q
#    m: media dei logaritmi
#    s2: varianza dei logaritmi
#    n: numero di estrazioni
#    a: alpha*H(xi)
#    nu: n-1
#    p: -(n+1)/2
#    """
##    d = (logq - m)
##    t2 = n*d*d/s2
#    return np.log1p(t2/nu)*p + a*logq
#
##    return (1 + t2/nu)**p*np.exp((a)*logq)
#
#def integrator(m, s, n, a):
#    logqs = np.linspace(m-3*s, m+3*s,100)
##    logq = (logqs[1]-logqs[0])
#    dlogq = np.log(logqs[1]-logqs[0])
#    I = -np.inf
#    s2 = s**2
#    p  = -(n+1)/2.
#    nu = n-1
#    for logq in logqs:
#        I = log_add(integrand(logq, m, s2, n, a, nu, p) + dlogq, I)
#
#    return I + gammaln(-p) - gammaln(nu/2.) - 0.5*np.log(nu*np.pi)#np.log(I)

@jit
def integrator(qs, ai, g):
#    hatq  = np.linspace(np.min(qs), np.max(qs),100)
    N = 30
    dq = (np.max(qs) - np.min(qs))/N
    logdq = np.log(dq)#hatq[1] - hatq[0])
    I = -np.inf
    log1mqs = np.log(1 - qs)
    logqs = np.log(qs)
    qmin = np.min(qs)
    n = len(qs)
    for i in range(N):
        I = log_add(integrand(qmin + i*dq , logqs, log1mqs, ai, n, g) + logdq, I)
    return I

@jit(nopython = True)
def integrand(q, logqs, log1mqs, a, n, g):
    return np.sum(logqs*(g*q-1)) + np.sum(log1mqs*(g*(1-q)-1)) + (a-1)*np.log(q) - n*numba_gammaln(g*q) - n*numba_gammaln(g*(1-q)) + numba_gammaln(g)


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
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, prior_pars = lambda x: 0, max_a = 10000, max_g = 200, max_N = 300, nthreads = 4):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.n_samps    = len(samples)
        self.labels     = pars
        self.names      = pars + ['a', 'N', 'g']
        self.bounds     = bounds + [[0, max_a], [10,max_N], [0, max_g]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.model      = model
        self.prior_norm = np.log(1/bounds[-1][0]**2 - 1/bounds[-1][1]**2) + np.log(np.exp(-bounds[-2][0]) - np.exp(-bounds[-2][1]))
        self.prec_probs = {}
        self.p          = Pool(nthreads)

    
    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            logP = -np.log(x['N']) - x['a']# - x['g']# - self.prior_norm
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP
    
    def log_likelihood(self, x):
        N  = int(x['N'])
        m  = np.linspace(self.x_min, self.x_max, N)
        dm = m[1] - m[0]
        ns = self.n_samps*np.ones(N)
        if N in self.prec_probs.keys():
            probs = self.prec_probs[N]
#            means = self.prec_probs[N][0]
#            std   = self.prec_probs[N][1]
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
#                p = p + np.log(dm) - logsumexp(p+np.log(dm))
                probs.append(np.exp(p)*dm)
#            probs = np.mean(probs, axis = 0)
#            probs = probs - logsumexp(probs+np.log(dm))
#            std   = np.std(probd)
#            t = [shuffle(p) for p in probs.T]
#            probs = np.array([p - logsumexp(p) for p in probs])
            probs = np.array([p/np.sum(p) for p in probs])
#            means = np.mean(probs, axis = 0)
#            means = means - logsumexp(means+np.log(dm))
#            std   = np.std(probs, axis = 0)
            self.prec_probs[N] = probs
        
#        means = np.mean(probs, axis = 0)
#        means = means - logsumexp(means+np.log(dm))
#        std   = np.std(probs, axis = 0)
        pars = [x[lab] for lab in self.labels]
        base = np.array([self.model(mi, *pars)*dm for mi in m])
        base = base/(np.sum(base))
        c_par = x['a']
        g = x['g']
        a = c_par*base
#        p = np.array([probs[randint(self.n_samps), i] for i in range(N)])
#        p = p - logsumexp(p+np.log(dm))
        #implemented as in scipy.stats.dirichlet.logpdf() w/o checks
        lnB = np.sum([numba_gammaln(ai) for ai in a]) - numba_gammaln(np.sum(a))
        integrals = np.zeros(N)
#        vals = [[mi,si,ni,ai] for mi,si,ni,ai in zip(means, std, ns, a)]
#        integrals = self.p.map(integrator, [[p,ai] for p, ai in zip(probs.T, a)])
        for i in prange(N):
            integrals[i] = integrator(probs[:,i], a[i], g)#(means[i], std[i], ns[i], a[i])
        logL = - lnB + np.sum(integrals)#[integrator(ms, ss, ns, ai) for ms, ss, ai in zip(means, std, a)])#my_dot(a-1, probs)#np.sum([my_dot(a-1, p) for p in probs])#/self.n_samps#np.sum((xlogy(a-1, p.T)).T, 0)
#        logL = np.sum([ai*p + (c_par - ai)*log_sub(0,p) + gammaln(c_par) - gammaln(ai) - gammaln(c_par - ai) for ai, p in zip(a, probs.T)])
#        logL = np.sum([beta(ai, c_par - ai).logpdf(p) for ai, p in zip(a, probs.T)])#- lnB + my_dot(a-1, p)#np.sum((xlogy(a-1, p.T)).T, 0)
        return logL
    

@njit
def numba_gammaln(x):
  return gammaln_float64(x)
  
@jit
def my_dot(a,b):
    return np.sum(a*b)
