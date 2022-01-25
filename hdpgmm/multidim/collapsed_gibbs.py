import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle

from pathlib import Path

from corner import corner
import cpnest.model
import itertools

from collections import namedtuple, Counter
from numpy import random

from scipy import stats
from scipy.stats import entropy, gamma, multivariate_t
from scipy.special import logsumexp, betaln, gammaln, erfinv
from scipy.interpolate import RegularGridInterpolator

from hdpgmm.multidim.utils import integrand, log_norm, scalar_log_norm

from time import perf_counter

import ray
from ray.util import ActorPool

from numba import jit, njit
from numba.extending import get_cython_function_address
import ctypes

from distutils.spawn import find_executable

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@njit
def numba_gammaln(x):
  return gammaln_float64(x)
  
if find_executable('latex'):
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

class Integrator(cpnest.model.Model):
    
    def __init__(self, bounds, events, logN_cnst, dim):
        
        super(Integrator, self).__init__()
        self.events    = events
        self.logN_cnst = logN_cnst
        self.dim       = dim
        self.names     = ['m{0}'.format(i+1) for i in range(self.dim)] + ['s{0}'.format(i+1) for i in range(self.dim)] + ['r{0}'.format(j) for j in range(int(self.dim*(self.dim-1)/2.))]
        self.bounds    = bounds + [[-1,1] for _ in range(int(self.dim*(self.dim-1)/2.))]
    
    def log_prior(self, x):
        logP = super(Integrator, self).log_prior(x)
        if not np.isfinite(logP):
            return -np.inf
            
        self.mean = np.array(x.values[:self.dim])
        
        corr = np.identity(self.dim)/2.
        corr[np.triu_indices(self.dim, 1)] = x.values[2*self.dim:]
        corr         = corr + corr.T
        sigma        = x.values[self.dim:2*self.dim]
        ss           = np.outer(sigma,sigma)
        self.cov_mat = ss@corr
        
        if not np.linalg.slogdet(self.cov_mat)[0] > 0:
            return -np.inf
        
        return logP
    
    def log_likelihood(self, x):
        return integrand(self.mean, self.covariance, self.events, self.logN_cnst, self.dim)

"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
"""

# natural sorting.
# list.sort(key = natural_keys)

def sort_matrix(a, axis = -1):
    '''
    Matrix sorting algorithm
    '''
    mat = np.array([[m, f] for m, f in zip(a[0], a[1])])
    keys = np.array([x for x in mat[:,axis]])
    sorted_keys = np.copy(keys)
    sorted_keys = np.sort(sorted_keys)
    indexes = [np.where(el == keys)[0][0] for el in sorted_keys]
    sorted_mat = np.array([mat[i] for i in indexes])
    return sorted_mat[:,0], sorted_mat[:,1]
    

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def my_student_t(df, t, mu, sigma, dim, s2max = np.inf):
    """
    http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    """
    vals, vecs = np.linalg.eigh(sigma)
    vals       = np.minimum(vals, s2max)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    x = 0.5 * (df + dim)
    A = numba_gammaln(x)
    B = numba_gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return float(A - B - C - D + E)

class CGSampler:
    '''
    Class to analyse a set of mass posterior samples and reconstruct the mass distribution.
    WARNING: despite being suitable to solve many different inference problems, thia algorithm was implemented to infer the black hole mass function. Both variable names and documentation are written accordingly.
    
    Arguments:
        :iterable events:               list of single-event posterior samples
        :list samp_settings:            settings for mass function chain (burnin, number of draws and thinning)
        :list samp_settings_ev:         settings for single event chain (see above)
        :float alpha0:                  initial guess for single-event concentration parameter
        :float gamma0:                  initial guess for mass function concentration parameter
        :list hyperpriors_ev:           hyperpriors for single-event NIG prior
        :float m_min:                   lower bound of mass prior
        :float m_max:                   upper bound of mass prior
        :bool verbose:                  verbosity of single-event analysis
        :str output_folder:             output folder
        :double initial_cluster_number: initial guess for the number of active clusters
        :bool process_events:           runs single-event analysis
        :int n_parallel_threads:        number of parallel actors to spawn
        :function injected_density:     python function with simulated density
        :iterable true_masses:          draws from injected_density around which are drawn simulated samples
        :iterable names:                str containing names to be given to single-event output files (e.g. ['GW150814', 'GW170817'])
        :iterable var_names:            str containing parameter names for corner plot
    
    Returns:
        :CGSampler: instance of CGSampler class
    
    Example:
        sampler = CGSampler(*args)
        sampler.run()
    '''
    def __init__(self, events,
                       samp_settings, # burnin, draws, step (list)
                       samp_settings_ev = None,
                       m_min = np.inf,
                       m_max = -np.inf,
                       alpha0 = 1,
                       gamma0 = 1,
                       prior_ev = [1,1/4.], #a, V
                       verbose = True,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       process_events = True,
                       n_parallel_threads = 8,
                       true_samples = None,
                       names = None,
                       seed = 0,
                       var_names = None,
                       n_samp_to_plot = 2000,
                       deltax = 1e-4, # Arbitrary choice!
                       restart = False,
                       n_gridpoints = 20,
                       ):
        
        # Settings
        self.burnin_mf, self.n_draws_mf, self.n_steps_mf = samp_settings
        
        if samp_settings_ev is not None:
            self.burnin_ev, self.n_draws_ev, self.n_steps_ev = samp_settings_ev
        else:
            self.burnin_ev, self.n_draws_ev, self.n_steps_ev = samp_settings
        
        self.restart            = restarts
        self.verbose            = verbose
        self.process_events     = process_events
        self.n_parallel_threads = n_parallel_threads
        self.events             = events
        
        if not seed == 0:
            self.rdstate = np.random.RandomState(seed = 1)
        else:
            self.rdstate = np.random.RandomState()
            
        self.seed = seed
        
        # Priors
        self.a_ev, self.V_ev = prior_ev
        self.sample_min      = np.array([np.min(ai) for ai in np.concatenate(self.events)])
        self.sample_max      = np.array([np.max(ai) for ai in np.concatenate(self.events)])
        self.m_min           = np.minimum(m_min, sample_min)
        self.m_max           = np.maximum(m_max, sample_max)
        self.dim             = len(self.events[-1][-1])
        
        # Sanity check for zeros in bounds
        for i in range(self.dim):
            if self.m_min[i] == 0:
                if self.sample_min > deltax:
                    self.m_min[i] = deltax
                else:
                    self.m_min[i] = self.sample_min/2.
            elif self.m_max[i] == 0:
                if self.sample_min < -deltax:
                    self.m_max[i] = -deltax
                else:
                    self.m_max[i] = self.sample_min/2.
                    
        # Probit
        self.transformed_events = [self.transform(ev) for ev in events]
        self.t_min = self.transform([self.m_min])
        self.t_max = self.transform([self.m_max])
        
        # Dirichlet Process
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        self.icn    = initial_cluster_number

        # Output
        self.output_folder  = Path(output_folder)
        self.true_samples   = true_samples
        self.output_recprob = Path(self.output_folder, 'reconstructed_events','mixtures')
        self.var_names      = var_names
        self.n_samp_to_plot = n_samp_to_plot
        self.n_gridpoints   = n_gridpoints
        
        if names is not None:
            self.names = names
        else:
            self.names = [str(i+1) for i in range(len(self.events))]
            
    def transform(self, samples):
        '''
        Coordinate change into probit space
        
        Arguments:
            :float or np.ndarray samples: mass sample(s) to transform
        Returns:
            :float or np.ndarray: transformed sample(s)
        '''
        
        if self.m_min > 0:
            min = self.m_min*0.9999
        else:
            min = self.m_min*1.0001
        if self.m_max > 0:
            max = self.m_max*1.0001
        else:
            max = self.m_max*0.9999
        cdf_bounds = [min, max]
        
        cdf = (np.array(samples).T - np.atleast_2d([cdf_bounds[0]]).T)/np.array([cdf_bounds[1] - cdf_bounds[0]]).T
        new_samples = np.sqrt(2)*erfinv(2*cdf-1).T
        
        if len(new_samples) == 1:
            return new_samples[0]
        
        return new_samples
    
    def initialise_samplers(self, marker):
        '''
        Initialises n_parallel_threads instances of SE_Sampler class
        '''
        event_samplers = []
        for i in range(self.n_parallel_threads):
            if not self.seed == 0:
                rdstate = np.random.RandomState(seed = i)
            else:
                rdstate = np.random.RandomState()
            event_samplers.append(SE_Sampler.remote(
                                            burnin        = self.burnin_ev,
                                            n_draws       = self.n_draws_ev,
                                            n_steps       = self.n_steps_ev,
                                            dim           = self.dim,
                                            alpha0        = self.alpha0,
                                            a             = self.a_ev,
                                            V             = self.V_ev,
                                            glob_m_max    = self.m_max,
                                            glob_m_min    = self.m_min,
                                            output_folder = self.output_folder,
                                            verbose       = self.verbose,
                                            transformed   = True,
                                            var_names     = self.var_names,
                                            rdstate       = rdstate,
                                            restart       = self.restart,
                                            n_gridpoints  = self.n_gridpoints,
                                            initial_cluster_number = self.icn,
                                            hierarchical_flag = True,
                                            ))
        return ActorPool(event_samplers)
        
    def run_event_sampling(self):
        '''
        Runs all the single-event analysis.
        '''
        if self.verbose:
            try:
                ray.init(ignore_reinit_error=True, num_cpus = self.n_parallel_threads)
            except:
                # Handles memory error
                # ValueError: The configured object store size (XXX.XXX GB) exceeds /dev/shm size (YYY.YYY GB). This will harm performance. Consider deleting files in /dev/shm or increasing its size with --shm-size in Docker. To ignore this warning, set RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1.
                ray.init(ignore_reinit_error=True, num_cpus = self.n_parallel_threads, object_store_memory=10**9)
        else:
            try:
                ray.init(ignore_reinit_error=True, num_cpus = self.n_parallel_threads, log_to_driver = False)
            except:
                ray.init(ignore_reinit_error=True, num_cpus = self.n_parallel_threads, log_to_driver = False, object_store_memory=10**9)
        i = 0
        self.posterior_functions_events = []
        pool = self.initialise_samplers()
        for s in pool.map(lambda a, v: a.run.remote(v), [[t_ev, id, None, ev, None] for ev, id, t_ev in zip(self.events, self.names, self.transformed_events)]):
            self.posterior_functions_events.append(s)
            i += 1
            print('\rProcessed {0}/{1} events\r'.format(i, len(self.events)), end = '')
        ray.shutdown()
        return
    
    def load_mixtures(self):
        '''
        Loads results from previously analysed events
        '''
        print('Loading mixtures...')
        self.posterior_functions_events = []
        
        #Path -> str -> Path for sorting purposes.
        prob_files = [str(Path(self.output_recprob, f)) for f in os.listdir(self.output_recprob) if f.startswith('posterior_functions')]
        prob_files.sort(key = natural_keys)
        prob_files = [Path(p) for p in prob_files]
        
        for prob in prob_files:
            sampfile = open(prob, 'rb')
            samps = pickle.load(sampfile)
            self.posterior_functions_events.append(samps)
    
    def display_config(self):
        print('Collapsed Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Initial guesses:\nalpha0 = {0}\tgamma0 = {1}\tN = {2}'.format(self.alpha0, self.gamma0, self.icn))
        print('Single event hyperparameters: a = {0}, V = {1}'.format(self.a_ev, self.V_ev))
        for v, u, d in zip(self.var_names, self.m_max, self.m_min):
            print('{0} between {1} and {2}'.format(v, *np.round((d, u), decimals = 0)))
        print('Burn-in: {0} samples'.format(self.burnin_mf))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws_mf, self.n_steps_mf))
        print('Verbosity: {0} Diagnostic: {1} Reproducible run: {2}'.format(bool(self.verbose), bool(self.diagnostic), bool(self.seed)))
        print('------------------------')
        return
    
    def run_mass_function_sampling(self):
        '''
        Creates an instance of MF_Sampler class
        '''
        self.load_mixtures()
        self.mf_folder = Path(self.output_folder, 'mass_function')
        if not self.mf_folder.exists():
            self.mf_folder.mkdir()
        sampler = MF_Sampler(posterior_functions_events = self.posterior_functions_events,
                             dim                    = self.dim,
                             burnin                 = self.burnin_mf,
                             n_draws                = self.n_draws_mf,
                             n_steps                = self.n_steps_mf,
                             m_min                  = self.m_min,
                             m_max                  = self.m_max,
                             t_min                  = self.t_min,
                             t_max                  = self.t_max,
                             alpha0                 = self.gamma0,
                             output_folder          = self.mf_folder,
                             initial_cluster_number = self.icn,
                             true_samples           = self.true_samples,
                             sigma_min              = np.std(self.transf_ev, axis = 0)/16.,
                             sigma_max              = np.std(self.transf_ev, axis = 0)/3.,
                             n_parallel_threads     = self.n_parallel_threads,
                             transformed            = True,
                             var_names              = self.var_names,
                             n_samp_to_plot         = self.n_samp_to_plot,
                             rdstate                = self.rdstate,
                             restart                = self.restart,
                             )
        sampler.run()
    
    def run(self):
        '''
        Performs full analysis (single-event if required and mass function)
        '''
        init_time = perf_counter()
        self.display_config()
        if self.process_events:
            self.run_event_sampling()
        self.run_mass_function_sampling()
        end_time = perf_counter()
        seconds = int(end_time - init_time)
        h = int(seconds/3600.)
        m = int((seconds%3600)/60)
        s = int(seconds - h*3600-m*60)
        print('Elapsed time: {0}h {1}m {2}s'.format(h, m, s))
        return
        
    
@ray.remote
class SE_Sampler:
    '''
    Class to reconstruct a posterior density function given samples.
    
    Arguments:
        :iterable mass_samples:         mass samples (in probit or normal space)
        :str event_id:                  name to be given to outputs
        :int burnin:                    number of steps to be discarded
        :int n_draws:                   number of posterior density draws
        :int step:                      number of steps between draws
        :iterable real_masses:          mass samples before coordinate change.
        :float alpha0:                  initial guess for concentration parameter
        :float a:                       hyperprior on Gamma shape parameter (for NIG)
        :float V:                       hyperprior on Normal std (for NIG)
        :float m_min:                   mass prior lower bound for the specific event
        :float m_max:                   mass prior upper bound for the specific event
        :float t_min:                   prior lower bound in probit space
        :float t_max:                   prior upper bound in probit space
        :float glob_m_max:              mass function prior upper bound (required for transforming back from probit space)
        :float glob_m_min:              mass function prior lower bound (required for transforming back from probit space)
        :str output_folder:             output folder
        :bool verbose:                  displays analysis progress status
        :double initial_cluster_number: initial guess for the number of active clusters
        :double transformed:            mass samples are already in probit space
        :iterable var_names:            variable names (for corner plots)
    
    Returns:
        :SE_Sampler: instance of SE_Sampler class
    
    Example:
        sampler = SE_Sampler(*args)
        sampler.run()
    '''
    def __init__(self, burnin,
                       n_draws,
                       n_steps,
                       dim,
                       alpha0 = 1,
                       a = 1,
                       V = 1/4.,
                       glob_m_max = None,
                       glob_m_min = None,
                       output_folder = './',
                       verbose = True,
                       initial_cluster_number = 5.,
                       transformed = False,
                       var_names = None,
                       inital_assign = None,
                       sigma_max = None,
                       rdstate = None,
                       hierarchical_flag = None,
                       restart = False,
                       n_gridpoints = 20,
                       deltax = 1e-4,
                       ):
        
        if rdstate == None:
            self.rdstate = np.random.RandomState()
        else:
            self.rdstate = rdstate

        self.burnin  = burnin
        self.n_draws = n_draws
        self.n_steps = n_steps
        
        if hierarchical_flag and (glob_m_min is None or glob_m_max is None):
            raise Warning('Running a hierarchical inference with no global min/max specified.')
        
        self.glob_m_min = glob_m_min
        self.glob_m_max = glob_m_max
        
        
        if sigma_max is None:
            self.sigma_max_from_data = True
        else:
            self.sigma_max_from_data = False
            self.sigma_max = sigma_max
        
        self.dim = dim

        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.nu  = np.max([a,self.dim])
        self.k  = V
        # Miscellanea
        self.default_icn = initial_cluster_number
        self.SuffStat = namedtuple('SuffStat', 'mean cov N')
        self.hierarchical_flag = hierarchical_flag
        self.restart = restart
        self.deltax = deltax
        # Output
        self.output_folder = output_folder
        self.verbose = verbose
        self.transformed = transformed
        self.var_names = var_names
        self.n_gridpoints = np.atleast_1d(n_gridpoints)
        if len(self.n_gridpoints) == (1 or self.dim):
            self.n_gridpoints = self.n_gridpoints*np.ones(self.dim)
        else:
            print('n_gridpoints is not scalar but its dimensions does not match the data point dimensions.')
            exit()
        
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        cdf_normal is the cumulative distribution function of the unit normal distribution.
        Adjusting glob_min/max has to be done internally because this class can be called independently from the hierarchical one.
        
        t(m) = cdf_normal((m-m_min)/(m_max - m_min))
        
        Arguments:
            :float or np.ndarray samples: mass sample(s) to transform
        Returns:
            :float or np.ndarray: transformed sample(s)
        '''
        
        if self.glob_m_min > 0:
            min = self.glob_m_min*0.9999
        else:
            min = self.glob_m_min*1.0001
        if self.m_max > 0:
            max = self.glob_m_max*1.0001
        else:
            max = self.glob_m_max*0.9999
        
        cdf_bounds = [min, max]
        cdf = (np.array(samples).T - np.atleast_2d([cdf_bounds[0]]).T)/np.array([cdf_bounds[1] - cdf_bounds[0]]).T
        new_samples = np.sqrt(2)*erfinv(2*cdf-1).T
        
        if len(new_samples) == 1:
            return new_samples[0]
        
        return new_samples
    
        
    def initial_state(self, samples):
        '''
        Creates initial state -  a dictionary that stores a number of useful variables.
        Entries are:
            :list 'cluster_ids_':    list of labels for the maximum number of active cluster across the run
            :np.ndarray 'data_':     transformed samples
            :int 'num_clusters_':    number of active clusters
            :double 'alpha_':        actual value of concentration parameter
            :int 'Ntot':             total number of samples
            :dict 'hyperparameters': parameters of the hyperpriors
            :dict 'suffstats':       mean, variance and number of samples of each active cluster
            :list 'assignment':      list of cluster assignments (one for each sample)
        '''
        if self.restart:
            try:
                assign = np.genfromtxt(Path(self.output_assignment, 'assignment_{0}.txt'.format(self.e_ID))).astype(int)
            except:
                assign = np.array([int(a//(len(self.samples)/int(self.icn))) for a in range(len(self.samples))])
        elif self.initial_assign is not None:
            assign = self.initial_assign
        else:
            assign = np.array([int(a//(len(self.samples)/int(self.icn))) for a in range(len(self.samples))])
        cluster_ids = list(set(assign))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': self.samples,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(self.samples),
            'hyperparameters_': {
                "L": self.L,
                "k": self.k,
                "nu": self.nu,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': assign,
            }
        self.state = state
        self.update_suffstats()
        return
    
    def update_suffstats(self):
        '''
        Updates sufficient statistics for each cluster
        '''
        for cluster_id, N in Counter(self.state['assignment']).items():
            points_in_cluster = [x for x, cid in zip(self.state['data_'], self.state['assignment']) if cid == cluster_id]
            mean = np.atleast_2d(np.array(points_in_cluster).mean(axis = 0))
            cov  = np.cov(np.array(points_in_cluster), rowvar = False)
            M    = len(points_in_cluster)
            self.state['suffstats'][cluster_id] = self.SuffStat(mean, cov, M)
    
    def log_predictive_likelihood(self, data_id, cluster_id):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster - Eq. (2.30)
        
        Arguments:
            :int data_id:    index of the considered sample
            :int cluster_id: index of the considered cluster
        
        Returns:
            :double: log Likelihood
        '''
        if cluster_id == "new":
            ss = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        else:
            ss  = self.state['suffstats'][cluster_id]
            
        x = self.state['data_'][data_id]
        mean = ss.mean
        S    = ss.cov
        N    = ss.N
        # Update hyperparameters
        k_n  = self.state['hyperparameters_']["k"] + N
        mu_n = np.atleast_2d((self.state['hyperparameters_']["mu"]*self.state['hyperparameters_']["k"] + N*mean)/k_n)
        nu_n = self.state['hyperparameters_']["nu"] + N
        L_n  = self.state['hyperparameters_']["L"]*self.state['hyperparameters_']["k"] + S*N + self.state['hyperparameters_']["k"]*N*np.matmul((mean - self.state['hyperparameters_']["mu"]).T, (mean - self.state['hyperparameters_']["mu"]))/k_n
        # Update t-parameters
        t_df    = nu_n - self.dim + 1
        t_shape = L_n*(k_n+1)/(k_n*t_df)
        # Compute logLikelihood
        logL = my_student_t(df = t_df, t = np.atleast_2d(x), mu = mu_n, sigma = t_shape, dim = self.dim, s2max = self.sigma_max)
        return logL

    def add_datapoint_to_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        mean = (ss.mean*(ss.N)+x)/(ss.N+1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) + np.matmul(x.T, x))/(ss.N+1) - np.matmul(mean.T, mean)
        if (cov < 0).any(): # Numerical issue for clusters with one sample (variance = 0)
            cov[cov < 0] = 0
        return self.SuffStat(mean, cov, ss.N+1)


    def remove_datapoint_from_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        if ss.N == 1:
            return(self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) - np.matmul(x.T, x))/(ss.N-1) - np.matmul(mean.T, mean)
        if (cov < 0).any(): # Numerical issue for clusters with one sample (variance = 0)
            cov[cov < 0] = 0
        return self.SuffStat(mean, cov, ss.N-1)
        
    def cluster_assignment_distribution(self, data_id):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster. Eq. (2.39)
        
        Arguments:
            :int data_id: sample index
        
        Returns:
            :dict: p_i for each cluster
        """
        scores = {}
        cluster_ids = list(self.state['suffstats'].keys()) + ['new']
        for cid in cluster_ids:
            scores[cid] = self.log_predictive_likelihood(data_id, cid)
            scores[cid] += self.log_cluster_assign_score(cid)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def log_cluster_assign_score(self, cluster_id):
        """
        Log-likelihood that a new point generated will
        be assigned to cluster_id given the current state.
        """
        if cluster_id == "new":
            return np.log(self.state["alpha_"])
        else:
            return np.log(self.state['suffstats'][cluster_id].N)

    def create_cluster(self):
        state["num_clusters_"] += 1
        cluster_id = max(self.state['suffstats'].keys()) + 1
        self.state['suffstats'][cluster_id] = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        self.state['cluster_ids_'].append(cluster_id)
        return cluster_id

    def destroy_cluster(self, cluster_id):
        self.state["num_clusters_"] -= 1
        del self.state['suffstats'][cluster_id]
        self.state['cluster_ids_'].remove(cluster_id)
        
    def prune_clusters(self):
        for cid in self.state['cluster_ids_']:
            if self.state['suffstats'][cid].N == 0:
                self.destroy_cluster(cid)

    def sample_assignment(self, data_id):
        """
        Samples new assignment from marginal distribution.
        If cluster is "new", creates a new cluster.
        
        Arguments:
            :int data_id: index of the sample to be assigned
        
        Returns:
            :int: index of the selected cluster
        """
        scores = self.cluster_assignment_distribution(data_id).items()
        labels, scores = zip(*scores)
        cid = self.rdstate.choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster()
        else:
            return int(cid)

    def update_alpha(self, burnin = 200):
        '''
        Updates concentration parameter using a Metropolis-Hastings sampling scheme.
        
        Arguments:
            :int burnin: MH burnin
        
        Returns:
            :double: new concentration parametere value
        '''
        a_old = self.state['alpha_']
        n     = self.state['Ntot']
        K     = len(self.state['cluster_ids_'])
        for _ in range(burnin+self.rdstate.randint(100)):
            a_new = a_old + self.rdstate.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(self.rdstate.uniform()):
                    a_old = a_new
        return a_old

    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Gaussian Mixture Model
        """
        # alpha sampling
        self.state['alpha_'] = self.update_alpha()
        self.alpha_samples.append(self.state['alpha_'])
        pairs = zip(self.state['data_'], self.state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            self.state['suffstats'][cid] = self.remove_datapoint_from_suffstats(datapoint, self.state['suffstats'][cid])
            self.prune_clusters()
            cid = self.sample_assignment(data_id)
            self.state['assignment'][data_id] = cid
            self.state['suffstats'][cid] = self.add_datapoint_to_suffstats(self.state['data_'][data_id], self.state['suffstats'][cid])
        self.n_clusters.append(len(self.state['cluster_ids_']))
    
    def sample_mixture_parameters(self):
        '''
        Draws a mixture sample (weights, means and variances) using conditional probabilities. Eqs. (3.2) and (3.3)
        '''
        ss = self.state['suffstats']
        alpha = [ss[cid].N + self.state['alpha_'] / self.state['num_clusters_'] for cid in self.state['cluster_ids_']]
        weights = self.rdstate.dirichlet(alpha).flatten()
        components = {}
        for i, cid in enumerate(self.state['cluster_ids_']):
            mean = ss[cid].mean
            S    = ss[cid].cov
            N    = ss[cid].N
            k_n  = self.state['hyperparameters_']["k"] + N
            mu_n = np.atleast_2d((self.state['hyperparameters_']["mu"]*self.state['hyperparameters_']["k"] + N*mean)/k_n)
            nu_n = self.state['hyperparameters_']["nu"] + N
            L_n  = self.state['hyperparameters_']["L"] + S*N + self.state['hyperparameters_']["k"]*N*np.matmul((mean - self.state['hyperparameters_']["mu"]).T, (mean - self.state['hyperparameters_']["mu"]))/k_n
            # Update t-parameters
            s = stats.invwishart(df = nu_n, scale = L_n).rvs(random_state = self.rdstate)
            t_df    = nu_n - self.dim + 1
            t_shape = L_n*(k_n+1)/(k_n*t_df)
            m = multivariate_t(df = t_df, loc = mu_n.flatten(), shape = t_shape).rvs(random_state = self.rdstate)
            components[i] = {'mean': m, 'cov': s, 'weight': weights[i], 'N': N}
        self.mixture_samples.append(components)
    
    def save_assignment_state(self):
        z = self.state['assignment']
        np.savetxt(Path(self.output_assignment, 'assignment_{0}.txt'.format(self.e_ID)), np.array(z).T)
        return
    
    def run_sampling(self):
        """
        Runs the sampling algorithm - Listing 1
        """
        self.initial_state(self.samples)
        for i in range(self.burnin):
            if self.verbose:
                print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step()
        if self.verbose:
            print('\n', end = '')
        for i in range(self.n_draws):
            if self.verbose:
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.n_steps):
                self.gibbs_step()
            self.sample_mixture_parameters()
            if i%100 == 0:
                self.save_assignment_state()
        self.save_assignment_state()
        if self.verbose:
            print('\n', end = '')
        return

    def postprocess(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        
        lower_bound = np.maximum(self.m_min, self.glob_m_min)
        upper_bound = np.minimum(self.m_max, self.glob_m_max)
        
        points = [np.linspace(l, u, n) for l, u, n in zip(lower_bound, upper_bound, self.n_gridpoints)]
        log_vol_el = np.sum([np.log(v[1]-v[0]) for v in points])
        gridpoints = np.array(list(itertools.product(*points)))
        percentiles = [50]
        n_points = len(gridpoints)
        
        p = {}
        prob = []
        for i, ai in enumerate(gridpoints):
            a = self.transform([ai])
            #FIXME: scrivere log_norm in cython
            print('\rGrid evaluation: {0}/{1}'.format(i+1, n_points), end = '')
            logsum = np.sum([scalar_log_norm(par,0, 1) for par in a])
            prob.append([logsumexp([log_norm(a, component['mean'], component['cov']) + np.log(component['weight']) for component in sample.values()]) - logsum for sample in self.mixture_samples])
        prob = np.array(prob).reshape([n_points for _ in range(self.dim)] + [self.n_draws])
        
        log_draws_interp = []
        for i in range(self.n_draws):
            log_draws_interp.append(RegularGridInterpolator(points, prob[...,i] - logsumexp(prob[...,i] + log_vol_el)))
        
        picklefile = open(Path(self.output_posteriors, 'posterior_functions_{0}.pkl'.format(self.e_ID)), 'wb')
        pickle.dump(log_draws_interp, picklefile)
        picklefile.close()
        
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = -1)
        normalisation = logsumexp(p[50] + log_vol_el)
        for perc in percentiles:
            p[perc] = p[perc] - normalisation
        
        picklefile = open(Path(self.output_recprob, 'log_rec_prob_{0}.pkl'.format(self.e_ID)), 'wb')
        pickle.dump(RegularGridInterpolator(points, p[50]), picklefile)
        picklefile.close()
        
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = -1))
        for perc in percentiles:
            p[perc] = p[perc]/np.exp(normalisation)
            
        prob = np.array(prob)

        picklefile = open(Path(self.output_mixtures, 'posterior_functions_{0}.pkl'.format(self.e_ID)), 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        self.sample_probs = prob
        self.median = np.array(p[50])
        self.points = points
        
        samples_to_plot = np.array(MH_single_event(RegularGridInterpolator(points, p[50]), upper_bound, lower_bound, len(self.samples)))
        c = corner(self.initial_samples, color = 'orange', labels = self.var_names, hist_kwargs={'density':True})
        c = corner(samples_to_plot, fig = c, color = 'blue', labels = self.var_names, hist_kwargs={'density':True})
        c.savefig(Path(self.output_pltevents, '{0}.pdf'.format(self.e_ID)), bbox_inches = 'tight')
        
        # Plot numbers of clusters
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(Path(self.output_n_clusters, 'n_clusters_{0}.pdf'.format(self.e_ID)), bbox_inches='tight')
        
        # Plot concentration parameter
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))), histtype = 'step', density = True)
        fig.savefig(Path(self.alpha_folder, 'alpha_{0}.pdf'.format(self.e_ID)), bbox_inches='tight')
    
    def make_folders(self):
        """
        Creates directories.
        """
        self.output_events = Path(self.output_folder, 'reconstructed_events')
        dirs       = ['rec_prob', 'n_clusters', 'events', 'mixtures', 'posteriors', 'entropy', 'alpha', 'assignment']
        attr_names = ['output_recprob', 'output_n_clusters', 'output_pltevents', 'output_mixtures', 'output_posteriors', 'output_entropy', 'output_alpha', 'output_assignment']
        
        if not self.output_events.exists():
            self.output_events.mkdir()
        
        for d, attr in zip(dirs, attr_names):
            newfolder = Path(self.output_events, d)
            if not newfolder.exists():
                try:
                    newfolder.mkdir()
                except FileExistsError:
                    # This is to avoid that, while running several parallel single-event analysis,
                    # more than one instance of SE_Sampler attempts to create the folder.
                    # In that case, a (FileExistsError: [Errno 17] File exists: 'filename') is raised.
                    # This simply ignores the error and moves on with the inference.
                    pass
            setattr(self, attr, newfolder)
        return
        

    def run(self, args):
        """
        Runs the sampler, saves samples and produces output plots.
        
        Arguments:
            :iterable args: Iterable with arguments. These are (in order):
                                - mass samples: the samples to analyse
                                - event id: name to be given to the data set
                                - (m_min, m_max): lower and upper bound for mass (optional - if not provided, uses min(mass_samples) and max(mass_samples))
                                - real masses: samples in natural space, if mass samples are already transformed (optional - if it does not apply, use None)
                                - inital assignment: initial guess for cluster assignments (optional - if it does not apply, use None)
        """
        
        # Unpack arguments
        samples = args[0]
        event_id = args[1]
        real_samples = args[3]
        
        if args[2] is not None:
            m_min, m_max = args[2]
        else:
            if real_samples is not None:
                m_min = np.array([np.min(mi) for mi in np.array(real_samples).T])
                m_max = np.array([np.max(mi) for mi in np.array(real_samples).T])
            else:
                m_min = np.array([np.min(mi) for mi in np.array(samples).T])
                m_max = np.array([np.min(mi) for mi in np.array(samples).T])
        
        # Sanity check for zeros in bounds
        for i in range(self.dim):
            if self.m_min[i] == 0:
                if self.sample_min > self.deltax:
                    self.m_min[i] = self.deltax
                else:
                    self.m_min[i] = self.sample_min/2.
            elif self.m_max[i] == 0:
                if self.sample_min < -self.deltax:
                    self.m_max[i] = -self.deltax
                else:
                    self.m_max[i] = self.sample_min/2.
        
        initial_assign = args[4]
        
        # Store arguments
        if real_masses is None:
            self.initial_samples = samples
        else:
            self.initial_samples = real_samples
            
        self.initial_assign = initial_assign
        self.e_ID           = event_id
        
        self.m_min      = np.minimum(m_min, np.min(self.initial_samples))
        self.m_max      = np.maximum(m_max, np.max(self.initial_samples))
        self.m_min_plot = m_min
        self.m_max_plot = m_max

        if self.glob_m_min is None:
            self.glob_m_min = self.m_min
        if self.glob_m_max is None:
            self.glob_m_max = self.m_max
        
        # Check consistency
        if real_masses is None and self.transformed:
            raise ValueError('Samples are expected to be already transformed but no initial samples are provided.')
            exit()

        if self.transformed:
            self.samples = samples
            self.t_max   = np.max(samples)
            self.t_min   = np.min(samples)
        else:
            self.samples = self.transform(samples)
            self.t_max   = self.transform(self.m_max)
            self.t_min   = self.transform(self.m_min)
        
        if self.sigma_max_from_data:
            self.sigma_max = np.std(self.samples, axis = 0)/2.
        
        self.b  = self.a*(self.sigma_max/2.)**2
        self.mu = np.mean(self.samples, axis = 0)

        self.alpha_samples = []
        self.mixture_samples = []
        self.icn = np.min([len(samples), self.default_icn])
        self.n_clusters = [self.icn]
        
        # Run the analysis
        self.make_folders()
        self.run_sampling()
        self.postprocess()
        return

class MF_Sampler():
    '''
    Class to reconstruct the mass function given a set of single-event posterior distributions
    
    Arguments:
        :iterable posterior_functions_events: mixture draws for each event
        :int burnin:                    number of steps to be discarded
        :int n_draws:                   number of posterior density draws
        :int step:                      number of steps between draws
        :float alpha0: initial guess for concentration parameter
        :float m_min:                   mass prior lower bound for the specific event
        :float m_max:                   mass prior upper bound for the specific event
        :float t_min:                   prior lower bound in probit space
        :float t_max:                   prior upper bound in probit space
        :str output_folder: output folder
        :double initial_cluster_number: initial guess for the number of active clusters
        :function injected_density:     python function with simulated density
        :iterable true_masses:          draws from injected_density around which are drawn simulated samples
        :double sigma_min: sigma prior lower bound
        :double sigma_max: sigma prior upper bound
        :double m_max_plot: upper mass limit for output plots
        :int n_parallel_threads:        number of parallel actors to spawn
        :int ncheck: number of draws between checkpoints
        :double transformed:            mass samples are already in probit space
        
    Returns:
        :MF_Sampler: instance of CGSampler class
    
    Example:
        sampler = MF_Sampler(*args)
        sampler.run()
        
    '''
    def __init__(self, posterior_functions_events,
                       dim,
                       burnin,
                       n_draws,
                       n_steps,
                       m_min,
                       m_max,
                       t_min = -4,
                       t_max = 4,
                       alpha0 = 1,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       true_samples = None,
                       sigma_min = 0.003,
                       sigma_max = 0.7,
                       n_parallel_threads = 1,
                       ncheck = 5,
                       transformed = False,
                       var_names = None,
                       n_samp_to_plot = 2000,
                       n_gridpoints = 20,
                       rdstate = None,
                       restart = False,
                       deltax = 1e-4,
                       ):
                       
        if rdstate == None:
            self.rdstate = np.random.RandomState()
        else:
            self.rdstate = rdstate
    
        self.posterior_functions_events = posterior_functions_events
        self.true_samples = true_samples
        self.dim = dim
        
        # Priors
        self.m_min   = m_min
        self.m_max   = m_max
        # Sanity check for zeros in bounds
        for i in range(self.dim):
            if self.m_min[i] == 0:
                self.m_min[i] = deltax
            elif self.m_max[i] == 0:
                self.m_max[i] = -deltax

        if transformed:
            self.t_min = t_min
            self.t_max = t_max
        else:
            self.t_min = self.transform([m_min])
            self.t_max = self.transform([m_max])
        self.m_min_plot = m_min
        self.m_max_plot = m_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Settings
        self.n_parallel_threads = n_parallel_threads
        self.int_bounds = [[tmin, tmax] for tmin, tmax in zip(self.t_min, self.t_max)] + [[smin, smax] for smin, smax in zip(self.sigma_min, self.sigma_max)]
        self.p       = ActorPool([ScoreComputer.remote(self.int_bounds, self.dim, self.output_folder, self.rdstate) for _ in range(n_parallel_threads)])
        self.burnin  = burnin
        self.n_draws = n_draws
        self.n_steps = n_steps
        self.ncheck  = ncheck
        self.restart = restart
        
        # DP parameters
        self.alpha0 = alpha0
        self.icn    = np.min([len(posterior_functions_events), initial_cluster_number])
        # Output
        self.output_folder = Path(output_folder)
        if not self.output_folder.exists():
            self.output_folder.mkdir()

        self.mixture_samples = []
        self.n_clusters = [self.icn]
        self.alpha_samples = []
        self.points = [np.linspace(l, u, n_points) for l, u in zip(self.lower_bounds, self.upper_bounds)]
        self.log_vol_el = np.sum([np.log(v[1]-v[0]) for v in self.points])
        self.gridpoints = np.array(list(itertools.product(*self.points)))
        self.n_samp_to_plot = n_samp_to_plot
        self.var_names = var_names
        
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        
        Arguments:
            :float or np.ndarray samples: mass sample(s) to transform
        Returns:
            :float or np.ndarray: transformed sample(s)
        '''
        if self.m_min > 0:
            min = self.m_min*0.9999
        else:
            min = self.m_min*1.0001
        if self.m_max > 0:
            max = self.m_max*1.0001
        else:
            max = self.m_max*0.9999
        cdf_bounds = [min, max]
        
        cdf = (np.array(samples).T - np.atleast_2d([cdf_bounds[0]]).T)/np.array([cdf_bounds[1] - cdf_bounds[0]]).T
        new_samples = np.sqrt(2)*erfinv(2*cdf-1).T
        
        if len(new_samples) == 1:
            return new_samples[0]
        
        return new_samples
    
    def initial_state(self):
        '''
        Creates initial state
        '''
        self.update_draws()
        if self.restart:
            try:
                assign = np.genfromtxt(Path(self.output_folder, 'assignment_mf.txt')).astype(int)
            except:
                assign = np.array([int(a//(len(self.posterior_functions_events)/int(self.icn))) for a in range(len(self.posterior_functions_events))])
        else:
            assign = np.array([int(a//(len(self.posterior_functions_events)/int(self.icn))) for a in range(len(self.posterior_functions_events))])
        cluster_ids = list(set(assign))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': self.posterior_draws,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(self.posterior_draws),
            'assignment': assign,
            'ev_in_cl': {cid: list(np.where(np.array(assign) == cid)[0]) for cid in cluster_ids},
            'logL_D': {cid: None for cid in cluster_ids},
            'samples': {}
            }
        
        # Compute denominators
        self.integrator = cpnest.CPNest(Integrator(self.int_bounds, [], self.dim),
                                        verbose = 0,
                                        nlive   = 200, #FIXME: too few for a reliable estimate?
                                        maxmcmc = 1000,
                                        nensemble = 1,
                                        output = Path(self.output_folder, 'int_output'),
                                        )
        self.integrator.run()
        state['logL_D']["new"] = self.integrator.logZ
        state['samples']["new"] = np.fromiter(self.rdstate.choice(self.integrator.nested_samples), dtype = np.float64)[:-2]
        for cid in state['cluster_ids_']:
            self.integrator.user.events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
            self.integrator.run()
            state['logL_D'][cid] = self.integrator.logZ
            state['samples'][cid] = np.fromiter(self.rdstate.choice(self.integrator.nested_samples), dtype = np.float64)[:-2]
        
        self.state = state
        return
    
    
    def cluster_assignment_distribution(self, data_id):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
        """
        cluster_ids = list(self.state['ev_in_cl'].keys()) + ['new']
        output = self.p.map(lambda a, v: a.compute_score.remote(v), [[] for cid in cluster_ids])
        scores = {}
        for out in output:
            scores[out[0]] = out[1]
            self.numerators[out[0]] = out[2]
            self.starting_points[out[0]] = out[3]
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def create_cluster(self):
        self.state["num_clusters_"] += 1
        cluster_id = max(self.state['cluster_ids_']) + 1
        self.state['cluster_ids_'].append(cluster_id)
        self.state['ev_in_cl'][cluster_id] = []
        self.state['samples'][cluster_id] = self.state['samples']["new"]
        return cluster_id

    def destroy_cluster(self, cluster_id):
        self.state["num_clusters_"] -= 1
        self.state['cluster_ids_'].remove(cluster_id)
        self.state['ev_in_cl'].pop(cluster_id)
        
        
    def prune_clusters(self):
        for cid in self.state['cluster_ids_']:
            if len(self.state['ev_in_cl'][cid]) == 0:
                self.destroy_cluster(cid)

    def sample_assignment(self, data_id):
        """
        Sample new assignment from marginal distribution.
        If cluster is "new", create a new cluster.
        """
        self.numerators = {}
        self.samples = {}
        scores = self.cluster_assignment_distribution(data_id).items()
        labels, scores = zip(*scores)
        cid = random.choice(labels, p=scores)
        if cid == "new":
            new_cid = self.create_cluster()
            self.state['logL_D'][int(new_cid)] = self.numerators[cid]
            self.state['samples'][int(new_cid)] = self.samples[cid]
            return new_cid
        else:
            self.state['logL_D'][int(cid)] = self.numerators[int(cid)]
            self.state['samples'][int(cid)] = self.samples[int(cid)]
            return int(cid)

    def update_draws(self):
        draws = []
        for posterior_samples in self.posterior_functions_events:
            draws.append(posterior_samples[random.randint(len(posterior_samples))])
        self.posterior_draws = draws
    
    def drop_from_cluster(self, data_id, cid):
        self.state['ev_in_cl'][cid].remove(data_id)
        events = [self.posterior_draws[i] for i in self.state['ev_in_cl'][cid]]
        self.integrator.user.events = [self.posterior_draws[i] for i in self.state['ev_in_cl'][cid]]
        self.integrator.run()
        self.state['logL_D'][cid] = self.integrator.logZ
        self.state['samples'][cid] = np.fromiter(self.rdstate.choice(self.integrator.nested_samples), dtype = np.float64)[:-2]

    def add_to_cluster(self, data_id, cid):
        self.state['ev_in_cl'][cid].append(data_id)

    def update_alpha(self, burnin = 200):
        '''
        Updates concentration parameter using a Metropolis-Hastings sampling scheme.
        
        Arguments:
            :int burnin: MH burnin
        
        Returns:
            :double: new concentration parametere value
        '''
        a_old = self.state['alpha_']
        n     = self.state['Ntot']
        K     = len(self.state['cluster_ids_'])
        for _ in range(burnin+self.rdstate.randint(100)):
            a_new = a_old + self.rdstate.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = numba_gammaln(a_old) - numba_gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = numba_gammaln(a_new) - numba_gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(self.rdstate.uniform()):
                    a_old = a_new
        return a_old
    
    def gibbs_step(self):
        """
        Collapsed Gibbs sampler for Dirichlet Process Gaussian Mixture Model
        """
        self.update_draws()
        self.state['alpha_'] = self.update_alpha()
        self.alpha_samples.append(self.state['alpha_'])
        pairs = zip(self.state['data_'], self.state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            self.drop_from_cluster(data_id, cid)
            self.prune_clusters()
            cid = self.sample_assignment(data_id)
            self.add_to_cluster(data_id, cid)
            self.state['assignment'][data_id] = cid
        self.n_clusters.append(len(self.state['cluster_ids_']))
    
    def sample_mixture_parameters(self):
        alpha = [len(self.state['ev_in_cl'][cid]) + self.state['alpha_'] / self.state['num_clusters_'] for cid in self.state['cluster_ids_']]
        weights = self.rdstate.dirichlet(alpha).flatten()
        components = {}
        for i, cid in enumerate(self.state['cluster_ids_']):
            sample = self.state['samples'][cid]
            mean = np.array(sample[:self.dim])
            corr = np.identity(self.dim)/2.
            corr[np.triu_indices(self.dim, 1)] = sample[2*self.dim:]
            corr         = corr + corr.T
            sigma        = sample[self.dim:2*self.dim]
            ss           = np.outer(sigma,sigma)
            cov = ss@corr
            components[i] = {'mean': m, 'cov': cov, 'weight': weights[i]}
        self.mixture_samples.append(components)

    def postprocess(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        
        lower_bound = self.m_min_plot
        upper_bound = self.m_max_plot
        percentiles = [50]
        p = {}
        n_points = len(self.gridpoints)
        
        prob = []
        for i, ai in enumerate(self.gridpoints):
            a = self.transform([ai])
            #FIXME: scrivere log_norm in cython
            print('\rGrid evaluation: {0}/{1}'.format(i+1, n_points), end = '')
            logsum = np.sum([scalar_log_norm(par,0, 1) for par in a])
            prob.append([logsumexp([log_norm(a, component['mean'], component['cov']) + np.log(component['weight']) for component in sample.values()]) - logsum for sample in self.mixture_samples])
        prob = np.array(prob).reshape([n_points for _ in range(self.dim)] + [self.n_draws])

        log_draws_interp = []
        for i in range(self.n_draws):
            log_draws_interp.append(RegularGridInterpolator(self.points, prob[...,i] - logsumexp(prob[...,i] + self.log_vol_el)))
        
        # Saves interpolant functions into json file
        name = 'posterior_functions_mf_'
        extension ='.pkl'
        x = 0
        fileName = Path(self.output_folder, name + str(x) + extension)
        while fileName.exists():
            x = x + 1
            fileName = Path(self.output_folder, name + str(x) + extension)
        
        picklefile = open(fileName, 'wb')
        pickle.dump(log_draws_interp, picklefile)
        picklefile.close()
        
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = -1)
        normalisation = logsumexp(p[50] + log_vol_el)
        for perc in percentiles:
            p[perc] = p[perc] - normalisation
            
        self.sample_probs = prob
        self.median_mf = np.array(p[50])
        names = ['m'] + [str(perc) for perc in percentiles]
        
        picklefile = open(Path(self.output_folder, 'log_rec_prob_mf.pkl'), 'wb')
        pickle.dump(RegularGridInterpolator(self.points, p[50]), picklefile)
        picklefile.close()

        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = -1))
        for perc in percentiles:
            p[perc] = p[perc]/np.exp(normalisation)
        
        samples_to_plot = MH_single_event(RegularGridInterpolator(points, p[50]), self.upper_bounds, self.lower_bounds, self.n_samp_to_plot)
        c = corner(samples_to_plot, color = 'blue', labels = self.var_names, hist_kwargs={'density':True})
        if self.true_masses is not None:
            c = corner(self.true_masses, fig = c, color = 'orange', labels = self.var_names, hist_kwargs={'density':True})
        c.savefig(self.output_pltevents + '/obs_mass_function.pdf', bbox_inches = 'tight')
        
        name = 'posterior_mixtures_mf_'
        extension ='.pkl'
        x = 0
        fileName = Path(self.output_folder, name + str(x) + extension)
        while fileName.exists():
            x = x + 1
            fileName = Path(self.output_folder, name + str(x) + extension)
        
        picklefile = open(fileName, 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(Path(self.output_folder, 'n_clusters_mf.pdf'), bbox_inches='tight')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(Path(self.output_folder, 'gamma_mf.pdf'), bbox_inches='tight')
        
    
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        self.run_sampling()
        self.postprocess()
        return

    def checkpoint(self):
        try:
            picklefile = open(Path(self.output_events, 'checkpoint.pkl'), 'rb')
            samps = pickle.load(picklefile)
            picklefile.close()
        except:
            samps = []
        
        prob = []
        for ai in self.gridpoints:
            a = self.transform([ai])
            #FIXME: scrivere log_norm in cython
            print('\rGrid evaluation: {0}/{1} (checkpoint)'.format(i+1, n_points), end = '')
            logsum = np.sum([log_norm(par,0, 1) for par in a])
            prob.append([logsumexp([log_norm(a, component['mean'], component['cov']) + np.log(component['weight']) for component in sample.values()]) - logsum for sample in self.mixture_samples[-self.ncheck:]])
        prob = np.array(prob).reshape([n_points for _ in range(self.dim)] + [self.n_draws])

        log_draws_interp = []
        for i in range(self.n_draws):
            log_draws_interp.append(RegularGridInterpolator(self.points, prob[...,i] - logsumexp(prob[...,i] + self.log_vol_el)))
        
        samps = samps + log_draws_interp
        picklefile = open(Path(self.output_folder, 'checkpoint.pkl'), 'wb')
        pickle.dump(samps, picklefile)
        picklefile.close()

    def run_sampling(self):
        self.state = self.initial_state()
        for i in range(self.burnin):
            print('\rBURN-IN MF: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step()
        print('\n', end = '')
        for i in range(self.n_draws):
            print('\rSAMPLING MF: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.n_steps):
                self.gibbs_step()
            self.sample_mixture_parameters()
            if (i+1) % self.ncheck == 0:
                self.checkpoint()
        print('\n', end = '')

#------------#
# Integrator #
#------------#
class Integrator(cpnest.model.Model):
    
    def __init__(self, bounds, events, logN_cnst, dim):
        
        super(Integrator, self).__init__()
        self.events    = events
        self.logN_cnst = logN_cnst
        self.dim       = dim
        self.names     = ['m{0}'.format(i+1) for i in range(self.dim)] + ['s{0}'.format(i+1) for i in range(self.dim)] + ['r{0}'.format(j) for j in range(int(self.dim*(self.dim-1)/2.))]
        self.bounds    = bounds + [[-1,1] for _ in range(int(self.dim*(self.dim-1)/2.))]
    
    def log_prior(self, x):
        logP = super(Integrator, self).log_prior(x)
        if not np.isfinite(logP):
            return -np.inf
            
        self.mean = np.array(x.values[:self.dim])
        
        corr = np.identity(self.dim)/2.
        corr[np.triu_indices(self.dim, 1)] = x.values[2*self.dim:]
        corr         = corr + corr.T
        sigma        = x.values[self.dim:2*self.dim]
        ss           = np.outer(sigma,sigma)
        self.cov_mat = ss@corr
        
        if not np.linalg.slogdet(self.cov_mat)[0] > 0:
            return -np.inf
        
        return logP
    
    def log_likelihood(self, x):
        return integrand(self.mean, self.covariance, self.events, self.logN_cnst, self.dim)

@ray.remote
class ScoreComputer:
    def __init__(self, bounds, dim, output_folder, rdstate):
        
        self.bounds = bounds
        self.dim    = dim
        self.output_folder = output_folder
        self.rdstate = rdstate
        self.integrator = cpnest.CPNest(Integrator(self.bounds, [], self.dim),
                                        verbose = 0,
                                        nlive   = 200, #FIXME: too few for a reliable estimate?
                                        maxmcmc = 1000,
                                        nensemble = 1,
                                        output = Path(self.output_folder, 'int_output'),
                                        )


    def compute_score(self, args):
        """
        Wrapper for log_predictive_likelihood and log_cluster_assign_score
        (parallelized with Ray)
        
        Arguments:
            :list args: list of arguments. Contains:
                args[0]: sample index
                args[1]: cluster index
                args[2]: current state
                args[3]: posterior draws (list)
                args[4]: bounds (tuple (t_min, t_max, sigma_min, sigma_max))
        Returns:
            :list: list of computed values. Entries are:
                ret[0]: cluster index
                ret[1]: p_i for the considered cluster
                ret[2]: log Likelihood
                ret[3]: starting point for Metropolis-Hastings
        """
        data_id = args[0]
        cid     = args[1]
        state   = args[2]
        posterior_draws = args[3]
        
        score, logL_N, sample = self.log_predictive_likelihood(data_id, cid, state, posterior_draws)
        score += self.log_cluster_assign_score(cid, state)
        score = np.exp(score)
        return [cid, score, logL_N, sample]

    def log_cluster_assign_score(self, cluster_id, state):
        """
        Log-likelihood that a new point generated will
        be assigned to cluster_id given the current state. Eqs. (2.26) and (2.27)
        
        Arguments:
            :int cluster_id: index of the considered cluster
            :dict state:     current state
        
        Returns:
            :double: log Likelihood
        """
        if cluster_id == "new":
            return np.log(state["alpha_"])
        else:
            if len(state['ev_in_cl'][cluster_id]) == 0:
                return -np.inf
            return np.log(len(state['ev_in_cl'][cluster_id]))

    def log_predictive_likelihood(self, data_id, cluster_id, state, posterior_draws):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster - part of Eq. (2.39)
        
        Arguments:
            :int data_id:    index of the considered sample
            :int cluster_id: index of the considered cluster
            :dict state:     current state
        
        Returns:
            :double: log Likelihood
        '''
        if cluster_id == "new":
            events = []
        else:
            events = [posterior_draws[i] for i in state['ev_in_cl'][cluster_id]]
        events.append(posterior_draws[data_id])
        logL_D = state['logL_D'][cluster_id] #denominator
        self.integrator.user.events = events
        self.integrator.run()
        logL_N = self.integrator.logZ #numerator
        sample = np.fromiter(self.rdstate.choice(self.integrator.nested_samples), dtype = np.float64)[:-2]
        return logL_N - logL_D, logL_N, sample

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
