import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

from collections import namedtuple, Counter
from numpy import random

from scipy import stats
from scipy.stats import entropy, gamma
from scipy.spatial.distance import jensenshannon as js
from scipy.special import logsumexp, betaln, gammaln, erfinv
from scipy.interpolate import interp1d
from scipy.integrate import dblquad

from hdpgmm.sampler_component_pars import sample_point

from time import perf_counter

import ray
from ray.util import ActorPool
from ray.util.multiprocessing import Pool

from utils import integrand, compute_norm_const, log_norm

from matplotlib import rcParams
from numba import jit, njit
from numba.extending import get_cython_function_address
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@njit
def numba_gammaln(x):
  return gammaln_float64(x)

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
Paper: https://arxiv.org/pdf/xxxx.xxxxx.pdf
An excellent handbook on conjugate priors: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
"""

def sort_matrix(a, axis = -1):
    '''
    2d matrix sorting algorithm.
    
    Arguments:
        :np.ndarray a: the matrix to be sorted
        :int axis:     sorting axis
        
    Returns:
        :np.ndarray: first sorted column
        :np.ndarray: second sorted column
    '''
    mat = np.array([[m, f] for m, f in zip(a[0], a[1])])
    keys = np.array([x for x in mat[:,axis]])
    sorted_keys = np.copy(keys)
    sorted_keys = np.sort(sorted_keys)
    indexes = [np.where(el == keys)[0][0] for el in sorted_keys]
    sorted_mat = np.array([mat[i] for i in indexes])
    return sorted_mat[:,0], sorted_mat[:,1]
    

def atoi(text):
    '''
    Converts string to integers (if digit).
    
    Arguments:
        :str text: string to be converted
    
    Returns:
        :int or str: the converted string
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    natural sorting.
    list.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class CGSampler:
    '''
    Class to analyse a set of mass posterior samples and reconstruct the mass distribution.
    WARNING: despite being suitable to solve many different inference problems, this algorithm was implemented to infer the black hole mass function. Both variable names and documentation are written accordingly.
    
    Arguments:
        :iterable events:               list of single-event posterior samples
        :list samp_settings:            settings for mass function chain (burnin, number of draws and thinning)
        :list samp_settings_ev:         settings for single event chain (see above)
        :float alpha0:                  initial guess for single-event concentration parameter
        :float gamma0:                  initial guess for mass function concentration parameter
        :list prior_ev:                 parameters for single-event NIG prior
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
        :bool seed:                     fixes seed to a default value (1) for reproducibility
    
    Returns:
        :CGSampler: instance of CGSampler class
    
    Example:
        sampler = CGSampler(*args)
        sampler.run()
    '''
    def __init__(self, events,
                       samp_settings, # burnin, draws, step (list)
                       samp_settings_ev = None,
                       alpha0 = 1,
                       gamma0 = 1,
                       prior_ev = [1,1/4.], #a, V
                       m_min = 5,
                       m_max = 70,
                       verbose = True,
                       diagnostic = False,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       process_events = True,
                       n_parallel_threads = 8,
                       injected_density = None,
                       true_masses = None,
                       names = None,
                       seed = False,
                       inj_post = None,
                       var_symbol = 'M',
                       unit = 'M_{\\odot}'
                       ):
        
        # Settings
        self.burnin_mf, self.n_draws_mf, self.step_mf = samp_settings
        
        if samp_settings_ev is not None:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings_ev
        else:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings
            
        self.verbose            = verbose
        self.diagnostic         = diagnostic
        self.process_events     = process_events
        self.n_parallel_threads = n_parallel_threads
        self.events             = events
        self.m_max_plot         = m_max
        self.event_samplers     = []
        
        if seed:
            self.rdstate = np.random.RandomState(seed = 1)
        else:
            self.rdstate = np.random.RandomState()
            
        self.seed = seed
        
        # Priors
        self.a_ev, self.V_ev = prior_ev
        sample_min           = np.min([np.min(a) for a in self.events])
        sample_max           = np.max([np.max(a) for a in self.events])
        self.m_min           = min([m_min, sample_min])
        self.m_max           = max([m_max, sample_max])
        
        # Probit
        self.transformed_events = [self.transform(ev) for ev in events]
        self.t_min              = self.transform(self.m_min)
        self.t_max              = self.transform(self.m_max)
        
        # Dirichlet Process
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        self.icn    = initial_cluster_number
        
        # Output
        self.output_folder      = output_folder
        self.injected_density   = injected_density
        self.true_masses        = true_masses
        self.output_recprob     = self.output_folder + '/reconstructed_events/mixtures/'
        self.inj_post           = inj_post
        self.var_symbol         = var_symbol
        self.unit               = unit
        
        if names is not None:
            self.names = names
        else:
            self.names = [str(i+1) for i in range(len(self.events))]
            
    def transform(self, samples):
        '''
        Coordinate change into probit space.
        cdf_normal is the cumulative distribution function of the unit normal distribution.
        
        t(m) = cdf_normal((m-m_min)/(m_max - m_min))

        
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
        cdf = (samples - cdf_bounds[0])/(cdf_bounds[1]-cdf_bounds[0])
        new_samples = np.sqrt(2)*erfinv(2*cdf-1)
        return new_samples
    
    def initialise_samplers(self, marker):
        '''
        Initialises n_parallel_threads instances of SE_Sampler class
        
        Arguments:
            :float marker: index from where to begin with index slicing
        
        Returns:
            :list: list of SE_samplers ready to run
        '''
        event_samplers = []
        for i, (ev, t_ev) in enumerate(zip(self.events[marker:marker+self.n_parallel_threads], self.transformed_events[marker:marker+self.n_parallel_threads])):
            event_samplers.append(SE_Sampler.remote(
                                            mass_samples  = t_ev,
                                            event_id      = self.names[marker+i],
                                            burnin        = self.burnin_ev,
                                            n_draws       = self.n_draws_ev,
                                            step          = self.step_ev,
                                            real_masses   = ev,
                                            alpha0        = self.alpha0,
                                            a             = self.a_ev,
                                            V             = self.V_ev,
                                            m_min         = np.min(ev),
                                            m_max         = np.max(ev),
                                            t_min         = np.min(t_ev),
                                            t_max         = np.max(t_ev),
                                            glob_m_max    = self.m_max,
                                            glob_m_min    = self.m_min,
                                            output_folder = self.output_folder,
                                            verbose       = self.verbose,
                                            diagnostic    = self.diagnostic,
                                            transformed   = True,
                                            seed          = self.seed,
                                            inj_post      = self.inj_post[self.names[marker+i]],
                                            var_symbol    = self.var_symbol,
                                            unit          = self.unit,
                                            initial_cluster_number = self.icn,
                                            ))
        return event_samplers
        
    def run_event_sampling(self):
        '''
        Runs all the single-event analysis.
        '''
        if self.verbose:
            ray.init(ignore_reinit_error=True, num_cpus = self.n_parallel_threads)
        else:
            ray.init(ignore_reinit_error=True, num_cpus = self.n_parallel_threads, log_to_driver = False)
        i = 0
        self.posterior_functions_events = []
        for n in range(int(len(self.events)/self.n_parallel_threads)+1):
            tasks = self.initialise_samplers(n*self.n_parallel_threads)
            pool = ActorPool(tasks)
            for s in pool.map(lambda a, v: a.run.remote(), range(len(tasks))):
                self.posterior_functions_events.append(s)
                i += 1
                print('\rProcessed {0}/{1} events\r'.format(i, len(self.events)), end = '')
        ray.shutdown()
        return
    
    def load_mixtures(self):
        '''
        Loads results from previously analysed events.
        '''
        print('Loading mixtures...')
        self.posterior_functions_events = []
        prob_files = [self.output_recprob+f for f in os.listdir(self.output_recprob) if f.startswith('posterior_functions')]
        prob_files.sort(key = natural_keys)
        for prob in prob_files:
            sampfile = open(prob, 'r')
            samps = json.load(sampfile)
            self.posterior_functions_events.append(np.array([d for d in samps.values()]))
    
    def display_config(self):
        print('Collapsed Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Concentration parameters:\nalpha0 = {0}\tgamma0 = {1}'.format(self.alpha0, self.gamma0))
        print('Burn-in: {0} samples'.format(self.burnin_mf))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws_mf, self.step_mf))
        print('------------------------')
        return
    
    def run_mass_function_sampling(self):
        '''
        Creates an instance of MF_Sampler class.
        '''
        self.load_mixtures()
        self.mf_folder = self.output_folder+'/mass_function/'
        if not os.path.exists(self.mf_folder):
            os.mkdir(self.mf_folder)
        flattened_transf_ev = np.array([x for ev in self.transformed_events for x in ev])
        sampler = MF_Sampler(
                       posterior_functions_events = self.posterior_functions_events,
                       burnin                     = self.burnin_mf,
                       n_draws                    = self.n_draws_mf,
                       step                       = self.step_mf,
                       alpha0                     = self.gamma0,
                       m_min                      = self.m_min,
                       m_max                      = self.m_max,
                       t_min                      = self.t_min,
                       t_max                      = self.t_max,
                       output_folder              = self.mf_folder,
                       injected_density           = self.injected_density,
                       true_masses                = self.true_masses,
                       sigma_min                  = np.std(flattened_transf_ev)/16.,
                       sigma_max                  = np.std(flattened_transf_ev)/3.,
                       m_max_plot                 = self.m_max_plot,
                       n_parallel_threads         = self.n_parallel_threads,
                       transformed                = True,
                       seed                       = self.seed,
                       initial_cluster_number     = min([self.icn, len(self.posterior_functions_events)]),
                       var_symbol                 = self.var_symbol,
                       unit                       = self.unit,
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
        

def my_student_t(df, t):
    '''
    Student-t log pdf
    
    Arguments:
        :float df: degrees of freedom
        :float t:  variable
        
    Returns:
        :float: student_t(df).logpdf(t)
    '''
    lnB = numba_gammaln(0.5)+ numba_gammaln(0.5*df) - numba_gammaln(0.5 + 0.5*df)
    return -0.5*np.log(df*np.pi) - lnB - ((df+1)*0.5)*np.log1p(t*t/df)
    
@ray.remote
class SE_Sampler:
    '''
    Class to reconstruct a posterior density function given samples.
    
    Arguments:
        :iterable mass_samples:         mass samples (in probit or natural space)
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
        :bool diagnostic:               run diagnostic routines
        :dict inj_post:                 injected posterior (interpolant)
        
    Returns:
        :SE_Sampler: instance of SE_Sampler class
    
    Example:
        sampler = SE_Sampler(*args)
        sampler.run()
    '''
    def __init__(self, mass_samples,
                       event_id,
                       burnin,
                       n_draws,
                       step,
                       real_masses = None,
                       alpha0 = 1,
                       a = 3,
                       V = 1/4.,
                       m_min = 5,
                       m_max = 50,
                       t_min = -4,
                       t_max = 4,
                       glob_m_max = None,
                       glob_m_min = None,
                       output_folder = './',
                       verbose = True,
                       diagnostic = False,
                       initial_cluster_number = 5.,
                       transformed = False,
                       inj_post = None,
                       seed = False,
                       var_symbol = 'M',
                       unit       = 'M_{\\odot}',
                       sigma_max  = None,
                       initial_assign = None,
                       ):
        if seed:
            self.rdstate = np.random.RandomState(seed = 1)
        else:
            self.rdstate = np.random.RandomState()
        if real_masses is None:
            self.initial_samples = mass_samples
        else:
            self.initial_samples = real_masses
        
        self.initial_assign = initial_assign
        self.e_ID    = event_id
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = np.min([m_min, np.min(mass_samples)])
        self.m_max   = np.max([m_max, np.max(mass_samples)])
        self.m_min_plot = m_min
        self.m_max_plot = m_max
        if glob_m_min is None:
            self.glob_m_min = self.m_min
        else:
            self.glob_m_min = glob_m_min
            
        if glob_m_max is None:
            self.glob_m_max = self.m_max
        else:
            self.glob_m_max = glob_m_max
        
        if transformed:
            self.mass_samples = mass_samples
            self.t_max   = t_max
            self.t_min   = t_min
        else:
            self.mass_samples = self.transform(mass_samples)
            self.t_max        = self.transform(self.m_max)
            self.t_min        = self.transform(self.m_min)
            
        if sigma_max is None:
            self.sigma_max = np.std(self.mass_samples)/2.
        else:
            self.sigma_max = sigma_max
        # DP parameters
        self.alpha0 = alpha0
        # NIG prior parameters
        self.b  = a*(self.sigma_max/2.)**2
        self.mu = np.mean(self.mass_samples)
        self.a  = a
        self.V  = V
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean var N')
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = [self.icn]
        self.verbose = verbose
        self.diagnostic = diagnostic
        self.alpha_samples = []
        self.inj_post = inj_post
        self.draws_z  = []
        self.data_to_follow = [100]
        self.var_symbol = var_symbol
        self.unit = unit
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        cdf_normal is the cumulative distribution function of the unit normal distribution.
        
        t(m) = cdf_normal((m-m_min)/(m_max - m_min))
        
        Arguments:
            :float or np.ndarray samples: mass sample(s) to transform
        Returns:
            :float or np.ndarray: transformed sample(s)
        '''
        if self.m_min > 0:
            min = self.glob_m_min*0.9999
        else:
            min = self.glob_m_min*1.0001
        if self.m_max > 0:
            max = self.glob_m_max*1.0001
        else:
            max = self.glob_m_max*0.9999
        cdf_bounds = [min, max]
        cdf = (samples - cdf_bounds[0])/(cdf_bounds[1]-cdf_bounds[0])
        new_samples = np.sqrt(2)*erfinv(2*cdf-1)
        return new_samples
    
        
    def initial_state(self, samples):
        '''
        Create initial state -  a dictionary that stores a number of useful variables
        
        Arguments:
            :np.ndarray samples: transformed samples
        
        Returns:
            :dict: new state. Entries are:
                :list 'cluster_ids_':    list of active cluster labels
                :np.ndarray 'data_':     transformed samples
                :int 'num_clusters_':    number of active clusters
                :double 'alpha_':        actual value of concentration parameter
                :int 'Ntot':             total number of samples
                :dict 'hyperparameters': parameters of the hyperpriors
                :dict 'suffstats':       mean, variance and number of samples of each active cluster
                :list 'assignment':      list of cluster assignments (one for each sample)
        '''
        if self.initial_assign is None:
            assign = [int(a//(len(samples)/int(self.icn))) for a in range(len(samples))]
        else:
            assign = list(self.initial_assign)
        cluster_ids = list(np.arange(int(np.max(assign)+1)))
        samp = np.copy(samples)
        state = {
            'cluster_ids_': cluster_ids,
            'data_': samp,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(samples),
            'hyperparameters_': {
                "b": self.b,
                "a": self.a,
                "V": self.V,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': assign
            }
        self.update_suffstats(state)
        self.draws_z.append(np.array(state['assignment']))
        return state
    
    def update_suffstats(self, state):
        '''
        Updates sufficient statistics for each cluster
        
        Arguments:
            :dict state: the current state
        '''
        for cluster_id, N in Counter(state['assignment']).items():
            points_in_cluster = [x for x, cid in zip(state['data_'], state['assignment']) if cid == cluster_id]
            mean = np.array(points_in_cluster).mean()
            var  = np.array(points_in_cluster).var()
            M    = len(points_in_cluster)
            state['suffstats'][cluster_id] = self.SuffStat(mean, var, M)
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster - Eq. (2.30)
        
        Arguments:
            :int data_id:    index of the considered sample
            :int cluster_id: index of the considered cluster
            :dict state:     current state
        
        Returns:
            :double: log Likelihood
        '''
        if cluster_id == "new":
            ss = self.SuffStat(0,0,0)
        else:
            ss  = state['suffstats'][cluster_id]
            
        x = state['data_'][data_id]
        mean = ss.mean
        sigma = ss.var
        N     = ss.N
        # Update hyperparameters
        V_n  = 1/(1/state['hyperparameters_']["V"] + N)
        mu_n = (state['hyperparameters_']["mu"]/state['hyperparameters_']["V"] + N*mean)*V_n
        b_n  = state['hyperparameters_']["b"] + (state['hyperparameters_']["mu"]**2/state['hyperparameters_']["V"] + (sigma + mean**2)*N - mu_n**2/V_n)/2.
        a_n  = state['hyperparameters_']["a"] + N/2.
        # Update t-parameters
        t_sigma = np.sqrt(b_n*(1+V_n)/a_n)
        t_sigma = min([t_sigma, self.sigma_max])
        t_x     = (x - mu_n)/t_sigma
        # Compute logLikelihood
        logL = my_student_t(df = 2*a_n, t = t_x)
        if not np.isfinite(logL):
            print(self.e_ID, logL, mean, sigma, x)
        return logL

    def add_datapoint_to_suffstats(self, x, ss):
        '''
        Updates single cluster sufficient statistics after sample assignment
        
        Arguments:
            :double x:                 posterior sample
            :SuffStat (NamedTuple) ss: old sufficient statistics
        
        Returns:
            :SuffStat: new sufficient statistics
        '''
        mean = (ss.mean*(ss.N)+x)/(ss.N+1)
        var  = (ss.N*(ss.var + ss.mean**2) + x**2)/(ss.N+1) - mean**2
        if var < 0: # Numerical issue for clusters with one sample (variance = 0)
            var = 0
        return self.SuffStat(mean, var, ss.N+1)


    def remove_datapoint_from_suffstats(self, x, ss):
        '''
        Updates single cluster sufficient statistics after sample removal
        
        Arguments:
            :double x:                 posterior sample
            :SuffStat (NamedTuple) ss: old sufficient statistics
        
        Returns:
            :SuffStat: new sufficient statistics
        '''
        if ss.N == 1:
            return(self.SuffStat(0,0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        var  = (ss.N*(ss.var + ss.mean**2) - x**2)/(ss.N-1) - mean**2
        if var < 0:
            var = 0
        return self.SuffStat(mean, var, ss.N-1)
    
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster. Eq. (2.39)
        
        Arguments:
            :int data_id: sample index
            :dict state:  current state
        
        Returns:
            :dict: p_i for each cluster
        """
        scores = {}
        cluster_ids = list(state['suffstats'].keys()) + ['new']
        for cid in cluster_ids:
            scores[cid] = self.log_predictive_likelihood(data_id, cid, state)
            scores[cid] += self.log_cluster_assign_score(cid, state)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

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
            return np.log(state['suffstats'][cluster_id].N)

    def create_cluster(self, state):
        '''
        Creates a new cluster when a sample is assigned to "new".
        
        Arguments:
            :dict state: current state to update
        
        Returns:
            :int: new cluster label
        '''
        state["num_clusters_"] += 1
        cluster_id = max(state['suffstats'].keys()) + 1
        state['suffstats'][cluster_id] = self.SuffStat(0, 0, 0)
        state['cluster_ids_'].append(cluster_id)
        return cluster_id

    def destroy_cluster(self, state, cluster_id):
        """
        Removes an empty cluster
        
        Arguments:
            :dict state:     current state to update
            :int cluster_id: label of the target empty cluster
        """
        state["num_clusters_"] -= 1
        del state['suffstats'][cluster_id]
        state['cluster_ids_'].remove(cluster_id)
        
    def prune_clusters(self,state):
        """
        Selects empty cluster(s) and removes them.
        
        Arguments:
            :dict state: current state to update
        """
        for cid in state['cluster_ids_']:
            if state['suffstats'][cid].N == 0:
                self.destroy_cluster(state, cid)

    def sample_assignment(self, data_id, state):
        """
        Samples new assignment from marginal distribution.
        If cluster is "new", creates a new cluster.
        
        Arguments:
            :int data_id: index of the sample to be assigned
            :dict state:  current state
        
        Returns:
            :int: index of the selected cluster
        """
        scores = self.cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = self.rdstate.choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster(state)
        else:
            return int(cid)

    def update_alpha(self, state, burnin = 100):
        '''
        Updates concentration parameter using a Metropolis-Hastings sampling scheme.
        
        Arguments:
            :dict state: current state
            :int burnin: MH burnin
        
        Returns:
            :double: new concentration parametere value
        '''
        a_old = state['alpha_']
        n     = state['Ntot']
        K     = len(state['cluster_ids_'])
        for _ in range(burnin):
            a_new = a_old + self.rdstate.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(random.uniform()):
                    a_old = a_new
        return a_old

    def gibbs_step(self, state):
        """
        Computes a single Gibbs step (updates all the sample assignments using conditional probabilities)
        
        Arguments:
            :dict state: current state to update
        """
        # alpha sampling
        state['alpha_'] = self.update_alpha(state)
        self.alpha_samples.append(state['alpha_'])
        pairs = zip(state['data_'], state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            state['suffstats'][cid] = self.remove_datapoint_from_suffstats(datapoint, state['suffstats'][cid])
            self.prune_clusters(state)
            cid = self.sample_assignment(data_id, state)
            state['assignment'][data_id] = cid
            state['suffstats'][cid] = self.add_datapoint_to_suffstats(state['data_'][data_id], state['suffstats'][cid])
#            self.draws_z.append(np.array(state['assignment']))
        self.n_clusters.append(len(state['cluster_ids_']))
        self.draws_z.append(np.array(state['assignment']))
    
    def sample_mixture_parameters(self, state):
        '''
        Draws a mixture sample (weights, means and variances) using conditional probabilities. Eqs. (3.2) and (3.3)
        
        Arguments:
            :dict state: current state
        '''
        ss = state['suffstats']
        alpha = [ss[cid].N + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = self.rdstate.dirichlet(alpha).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            mean = ss[cid].mean
            sigma = ss[cid].var
            N     = ss[cid].N
            V_n  = 1/(1/state['hyperparameters_']["V"] + N)
            mu_n = (state['hyperparameters_']["mu"]/state['hyperparameters_']["V"] + N*mean)*V_n
            b_n  = state['hyperparameters_']["b"] + (state['hyperparameters_']["mu"]**2/state['hyperparameters_']["V"] + (sigma + mean**2)*N - mu_n**2/V_n)/2.
            a_n  = state['hyperparameters_']["a"] + N/2.
            # Update t-parameters
            s = stats.invgamma(a_n, scale = b_n).rvs(random_state = self.rdstate)
            m = stats.norm(mu_n, s*V_n).rvs(random_state = self.rdstate)
            components[i] = {'mean': m, 'sigma': np.sqrt(s), 'weight': weights[i]}
        self.mixture_samples.append(components)
    
    def run_sampling(self):
        """
        Runs the sampling algorithm - Listing 1
        """
        state = self.initial_state(self.mass_samples)
        if self.diagnostic:
            self.sample_mixture_parameters(state)
        for i in range(self.burnin):
            if self.verbose:
                print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step(state)
        if self.verbose:
            print('\n', end = '')
        for i in range(self.n_draws):
            if self.verbose:
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.step):
                self.gibbs_step(state)
            self.sample_mixture_parameters(state)
        if self.verbose:
            print('\n', end = '')
        return
    
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Concentration parameters:\nalpha0 = {0}\tgamma0 = {1}'.format(self.alpha0, self.gamma0))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('------------------------')
        return

    def postprocess(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        # mass values
        lower_bound = max([self.m_min-1, self.glob_m_min])
        upper_bound = min([self.m_max+1, self.glob_m_max])
        app  = np.linspace(lower_bound, upper_bound, 1000)
        da   = app[1]-app[0]
        percentiles = [5,16, 50, 84, 95]
        p = {}
        
        # plots samples histogram
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(self.initial_samples, bins = int(np.sqrt(len(self.initial_samples))), histtype = 'step', density = True, label = r"\textsc{Mass samples}")
        
        # evaluates log probabilities in mass space
        prob = []
        for ai in app:
            a = self.transform(ai)
            prob.append([logsumexp([log_norm(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) - log_norm(a, 0, 1) for sample in self.mixture_samples])
        self.prob_draws         = np.exp(np.array(prob[1:-1]).T)
        if self.inj_post is not None:
            self.injected_posterior = self.inj_post(app[1:-1])
        self.dm_vals            = da
        self.m_vals             = app
        
        # saves interpolant functions into json file
        j_dict = {str(m): list(draws) for m, draws in zip(app, prob)}
        with open(self.output_posteriors + '/posterior_functions_{0}.json'.format(self.e_ID), 'w') as jsonfile:
            json.dump(j_dict, jsonfile)
        
        # computes percentiles
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = 1)
        normalisation = logsumexp(p[50] + np.log(da))
        for perc in percentiles:
            p[perc] = p[perc] - normalisation
        
        # Saves median and CR
        names = ['m'] + [str(perc) for perc in percentiles]
        np.savetxt(self.output_recprob + '/log_rec_prob_{0}.txt'.format(self.e_ID), np.array([app, p[5], p[16], p[50], p[84], p[95]]).T, header = ' '.join(names))
        
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        for perc in percentiles:
            p[perc] = p[perc]/np.exp(normalisation)
        prob = np.array(prob)
        
        # Computes entropy between samples and median
        ent = []
        for i in range(np.shape(prob)[1]):
            sample = np.exp(prob[:,i])
            ent.append(js(sample,p[50]))
        mean_ent = np.mean(ent)
        np.savetxt(self.output_entropy + '/KLdiv_{0}.txt'.format(self.e_ID), np.array(ent), header = 'mean JS distance = {0}'.format(mean_ent))
        
        # saves mixture samples into json file
        j_dict = {str(i): sample for i, sample in enumerate(self.mixture_samples)}
        with open(self.output_mixtures + '/posterior_functions_{0}.json'.format(self.e_ID), 'w') as jsonfile:
            json.dump(j_dict, jsonfile)
        
        # plots median and CR of reconstructed probability density
        self.sample_probs = prob
        self.median_mf = np.array(p[50])
        
        ax.fill_between(app, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        if self.inj_post is not None:
            ax.plot(app, self.inj_post(app), lw = 0.5, color = 'r', label = r"\textsc{Simulated}")
        ax.plot(app, p[50], marker = '', color = 'steelblue', label = r"\textsc{Reconstructed}", zorder = 100)
        if not self.unit == '':
            ax.set_xlabel('${0}\ [{1}]$'.format(self.var_symbol, self.unit))
        else:
            ax.set_xlabel('${0}$'.format(self.var_symbol))
        ax.set_ylabel('$p({0})$'.format(self.var_symbol))
        ax.set_xlim(self.m_min_plot, self.m_max_plot)
        ax.grid(True,dashes=(1,3))
        ax.legend(loc=0,frameon=False,fontsize=10)
        plt.savefig(self.output_pltevents + '/{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        ax.set_yscale('log')
        plt.savefig(self.output_pltevents + '/log_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        
        # plots number of clusters
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(self.n_clusters)), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_n_clusters+'n_clusters_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
        
        # plots concentration parameter
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(self.alpha_folder+'/alpha_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
    
    def make_folders(self):
        """
        Creates directories.
        WARNING: While running for the very first time a hierarchical inference with more than one parallel single-event analysis, this function could lead to an error:
        
        (FileExistsError: [Errno 17] File exists: 'filename').
        
        This is due to the fact that two different samplers are trying to create one of the shared folders at the same time.
        To cure this, just re-run the inference with the same output path.
        """
        self.output_events = self.output_folder + '/reconstructed_events/'
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        if not os.path.exists(self.output_events + '/rec_prob/'):
            os.mkdir(self.output_events + '/rec_prob/')
        self.output_recprob = self.output_events + '/rec_prob/'
        if not os.path.exists(self.output_events + '/n_clusters/'):
            os.mkdir(self.output_events + '/n_clusters/')
        self.output_n_clusters = self.output_events + '/n_clusters/'
        if not os.path.exists(self.output_events + '/events/'):
            os.mkdir(self.output_events + '/events/')
        self.output_pltevents = self.output_events + '/events/'
        if not os.path.exists(self.output_events + '/mixtures/'):
            os.mkdir(self.output_events + '/mixtures/')
        self.output_mixtures = self.output_events + '/mixtures/'
        if not os.path.exists(self.output_events + '/posteriors/'):
            os.mkdir(self.output_events + '/posteriors/')
        self.output_posteriors = self.output_events + '/posteriors/'
        if not os.path.exists(self.output_events + '/entropy/'):
            os.mkdir(self.output_events + '/entropy/')
        self.output_entropy = self.output_events + '/entropy/'
        if not os.path.exists(self.output_events + '/alpha/'):
            os.mkdir(self.output_events + '/alpha/')
        self.alpha_folder = self.output_events + '/alpha/'
        if self.diagnostic:
            if not os.path.exists(self.output_events + '/autocorrelation/'):
                os.mkdir(self.output_events + '/autocorrelation/')
            self.autocorrelation_folder = self.output_events + '/autocorrelation/'
            if not os.path.exists(self.output_events + '/convergence/'):
                os.mkdir(self.output_events + '/convergence/')
            self.convergence_folder = self.output_events + '/convergence/'
        return
    
    def run_diagnostic(self):
        self.autocorrelation()
        self.convergence_data()
        if self.inj_post is not None:
            self.convergence_true()
        return

    def autocorrelation(self):
        # FIXME: autocorrelation
        
        pass
    
    def convergence_data(self):
        dist = np.zeros(len(self.prob_draws)-1)
        idx  = np.arange(len(self.prob_draws)-1)
        for i in idx:
            dist[i] = js(self.prob_draws[i], self.prob_draws[i+1])
        avg = np.mean(dist[len(dist)//2:])
        dev = np.std(dist[len(dist)//2:])
        
        fig_conv, ax_conv = plt.subplots()
        ax_conv.plot(idx, np.ones(len(idx))*avg, marker = '', lw = 0.5, color = 'green', alpha = 0.4)
        ax_conv.fill_between(idx, np.ones(len(idx))*(avg-dev), np.ones(len(idx))*(avg+dev), color = 'palegreen', alpha = 0.2)
        ax_conv.plot(idx, dist, marker = '', ls = '--', lw = 0.5)
        ax_conv.set_xlabel('$n$')
        ax_conv.set_ylabel('$D_{JS}(q_{n}(M), q_{n+1}(M))$')
        ax_conv.grid(True,dashes=(1,3))
        fig_conv.savefig(self.convergence_folder + '/convergence_data_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        return
        
    def convergence_true(self):
        dist = np.zeros(len(self.prob_draws))
        idx  = np.arange(len(self.prob_draws))
        for i in idx:
            dist[i] = js(self.prob_draws[i], self.injected_posterior)
        avg = np.mean(dist[len(dist)//2:])
        dev = np.std(dist[len(dist)//2:])
        
        fig_conv, ax_conv = plt.subplots()
        ax_conv.plot(idx, np.ones(len(idx))*avg, marker = '', lw = 0.8, color = 'green', alpha = 0.4)
        ax_conv.fill_between(idx, np.ones(len(idx))*(avg-dev), np.ones(len(idx))*(avg+dev), color = 'palegreen', alpha = 0.2)
        ax_conv.plot(idx, dist, marker = '', ls = '--', lw = 0.5)
        ax_conv.set_xlabel('$n$')
        ax_conv.set_ylabel('$D_{JS}(q_{n}(M), q_{sim}(M))$')
        ax_conv.grid(True,dashes=(1,3))
        fig_conv.savefig(self.convergence_folder + '/convergence_true_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        return
        
    def run(self):
        """
        Runs the sampler, saves samples and produces output plots.
        """
        self.make_folders()
        self.run_sampling()
        self.postprocess()
        if self.diagnostic:
            self.run_diagnostic()
        return

class MF_Sampler():
    '''
    Class to reconstruct the mass function given a set of single-event posterior distributions
    
    Arguments:
        :iterable posterior_functions_events: mixture draws for each event
        :int burnin:                    number of steps to be discarded
        :int n_draws:                   number of posterior density draws
        :int step:                      number of steps between draws
        :float alpha0:                  initial guess for concentration parameter
        :float m_min:                   mass prior lower bound for the specific event
        :float m_max:                   mass prior upper bound for the specific event
        :float t_min:                   prior lower bound in probit space
        :float t_max:                   prior upper bound in probit space
        :str output_folder:             output folder
        :double initial_cluster_number: initial guess for the number of active clusters
        :function injected_density:     python function with simulated density
        :iterable true_masses:          draws from injected_density around which are drawn simulated samples
        :double sigma_min:              sigma prior lower bound
        :double sigma_max:              sigma prior upper bound
        :double m_max_plot:             upper mass limit for output plots
        :int n_parallel_threads:        number of parallel actors to spawn
        :int ncheck:                    number of draws between checkpoints
        :double transformed:            mass samples are already in probit space
        :bool diagnostic:               run diagnostic routines
        
    Returns:
        :MF_Sampler: instance of CGSampler class
    
    Example:
        sampler = MF_Sampler(*args)
        sampler.run()
        
    '''
    def __init__(self, posterior_functions_events,
                       burnin,
                       n_draws,
                       step,
                       alpha0 = 1,
                       m_min = 5,
                       m_max = 50,
                       t_min = -4,
                       t_max = 4,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       injected_density = None,
                       true_masses = None,
                       sigma_min = 0.005,
                       sigma_max = 0.7,
                       m_max_plot = 50,
                       n_parallel_threads = 1,
                       ncheck = 5,
                       transformed = False,
                       diagnostic = False,
                       seed = False,
                       var_symbol = 'M',
                       unit = 'M_{\\odot}'
                       ):
        
        if seed:
            self.rdstate = np.random.RandomState(seed = 1)
        else:
            self.rdstate = np.random.RandomState()
        
        self.seed = seed
            
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        if transformed:
            self.t_min = t_min
            self.t_max = t_max
        else:
            self.t_min = self.transform(m_min)
            self.t_max = self.transform(m_max)
         
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.posterior_functions_events = posterior_functions_events
        self.m_min_plot = m_min
        self.m_max_plot = m_max
        # DP parameters
        self.alpha0 = alpha0
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.injected_density = injected_density
        self.true_masses = true_masses
        self.n_parallel_threads = n_parallel_threads
        self.alpha_samples = []
        self.ncheck = ncheck
        self.diagnostic = diagnostic
        self.var_symbol = var_symbol
        self.unit = unit
        
        self.p = Pool(n_parallel_threads)
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        cdf_normal is the cumulative distribution function of the unit normal distribution.
        
        t(m) = cdf_normal((m-m_min)/(m_max - m_min))

        
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
        cdf = (samples - cdf_bounds[0])/(cdf_bounds[1]-cdf_bounds[0])
        new_samples = np.sqrt(2)*erfinv(2*cdf-1)
        return new_samples
    
    def initial_state(self):
        '''
        Create initial state -  a dictionary that stores a number of useful variables
        
        Arguments:
            :np.ndarray samples: transformed samples
        
        Returns:
            :dict: new state. Entries are:
                :list 'cluster_ids_':    list of active cluster labels
                :np.ndarray 'data_':     transformed samples
                :int 'num_clusters_':    number of active clusters
                :double 'alpha_':        actual value of concentration parameter
                :int 'Ntot':             total number of samples
                :dict 'hyperparameters': parameters of the hyperpriors
                :dict 'suffstats':       mean, variance and number of samples of each active cluster
                :list 'assignment':      list of cluster assignments (one for each sample)
                :dict 'ev_in_cl':        list of sample indices assigned to each cluster
                :dict logL_D:            log-Likelihood for samples in cluster
        '''
        self.update_draws()
        assign = [int(a//(len(self.posterior_functions_events)/int(self.icn))) for a in range(len(self.posterior_functions_events))]
        cluster_ids = list(np.arange(int(np.max(assign)+1)))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': self.posterior_draws,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(self.posterior_draws),
            'assignment': assign,
            'ev_in_cl': {cid: list(np.where(np.array(assign) == cid)[0]) for cid in cluster_ids},
            'logL_D': {cid: None for cid in cluster_ids}
            }
        for cid in state['cluster_ids_']:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
            n = len(events)
            state['logL_D'][cid] = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max)
        state['logL_D']["new"] = self.log_numerical_predictive([], self.t_min, self.t_max, self.sigma_min, self.sigma_max)
        return state
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
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
            return -np.log(self.t_max-self.t_min), -np.log(self.t_max-self.t_min)
        else:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cluster_id]]
        n = len(events)
        events.append(self.posterior_draws[data_id])
        logL_D = state['logL_D'][cluster_id] #denominator
        logL_N = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max) #numerator
        return logL_N - logL_D, logL_N

    def log_numerical_predictive(self, events, t_min, t_max, sigma_min, sigma_max):
        """"
        Computes integral over cluster parameters (mean and std) in Eq. (2.39)
        Normalization constant is required to avoid underflow while working with a high number of events.
        Arguments:
            :list events:      single event posterior distributions associated with the cluster
            :double t_min:     lower bound of mean parameter uniform prior
            :double t_max:     upper bound of mean parameter uniform prior
            :double sigma_min: lower bound of sigma parameter uniform prior
            :double sigma_max: upper bound of sigma parameter uniform prior
        Returns:
            :double: log predictive likelihood
        """
        logN_cnst = compute_norm_const(0, 1, events) + np.log(t_max - t_min) + np.log(sigma_max - sigma_min)
        I, dI = dblquad(integrand, t_min, t_max, gfun = sigma_min, hfun = sigma_max, args = [events, logN_cnst])
        return np.log(I) + logN_cnst
    
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster. Eq. (2.39)
        
        Arguments:
            :int data_id: sample index
            :dict state:  current state
        
        Returns:
            :dict: p_i for each cluster
        """
        cluster_ids = list(state['ev_in_cl'].keys()) + ['new']
        # can't pickle injected density
        saved_injected_density = self.injected_density
        self.injected_density  = None
        # FIXME: ray.get()
        output = self.p.map(self.compute_score, [[data_id, cid, state] for cid in cluster_ids])
        scores = {out[0]: out[1] for out in output}
        self.numerators = {out[0]: out[2] for out in output}
        self.injected_density = saved_injected_density
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores
        
    def compute_score(self, args):
        """
        Wrapper for log_predictive_likelihood and log_cluster_assign_score
        (parallelized with Ray)
        
        Arguments:
            :list args: list of arguments. Contains:
                args[0]: sample index
                args[1]: cluster index
                args[2]: current state
        Returns:
            :list: list of computed values. Entries are:
                ret[0]: cluster index
                ret[1]: p_i for the considered cluster
                ret[2]: log Likelihood
        """
        data_id = args[0]
        cid     = args[1]
        state   = args[2]
        score, logL_N = self.log_predictive_likelihood(data_id, cid, state)
        score += self.log_cluster_assign_score(cid, state)
        score = np.exp(score)
        return [cid, score, logL_N]
        
        
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

    def create_cluster(self, state):
        '''
        Creates a new cluster when a sample is assigned to "new".
        
        Arguments:
            :dict state: current state to update
        
        Returns:
            :int: new cluster label
        '''
        state["num_clusters_"] += 1
        cluster_id = max(state['cluster_ids_']) + 1
        state['cluster_ids_'].append(cluster_id)
        state['ev_in_cl'][cluster_id] = []
        return cluster_id

    def destroy_cluster(self, state, cluster_id):
        """
        Removes an empty cluster
        
        Arguments:
            :dict state:     current state to update
            :int cluster_id: label of the target empty cluster
        """
        state["num_clusters_"] -= 1
        state['cluster_ids_'].remove(cluster_id)
        state['ev_in_cl'].pop(cluster_id)
        
    def prune_clusters(self,state):
        """
        Selects empty cluster(s) and removes them.
        
        Arguments:
            :dict state: current state to update
        """
        for cid in state['cluster_ids_']:
            if len(state['ev_in_cl'][cid]) == 0:
                self.destroy_cluster(state, cid)

    def sample_assignment(self, data_id, state):
        """
        Samples new assignment from marginal distribution.
        If cluster is "new", creates a new cluster.
        
        Arguments:
            :int data_id: index of the sample to be assigned
            :dict state:  current state
        
        Returns:
            :int: index of the selected cluster
        """
        self.numerators = {}
        scores = self.cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = random.choice(labels, p=scores)
        if cid == "new":
            new_cid = self.create_cluster(state)
            state['logL_D'][int(new_cid)] = self.numerators[cid]
            return new_cid
        else:
            state['logL_D'][int(cid)] = self.numerators[int(cid)]
            return int(cid)

    def update_draws(self):
        """
        Draws a set of N single event posterior samples, one for each event, from the pools.
        Marginalisation over single event posterior distribution parameters - Eq. (2.39) - is carried   out drawing a new posterior distribution from the pool for every Gibbs
        """
        draws = []
        for posterior_samples in self.posterior_functions_events:
            draws.append(posterior_samples[random.randint(len(posterior_samples))])
        self.posterior_draws = draws
    
    def drop_from_cluster(self, state, data_id, cid):
        """
        Removes a sample from a cluster.
        
        Arguments:
            :dict state: current state
            :data_id:    sample index
            :cid:        cluster index
        """
        state['ev_in_cl'][cid].remove(data_id)
        events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
        n = len(events)
        state['logL_D'][cid] = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max)

    def add_to_cluster(self, state, data_id, cid):
        """
        Adds a sample to a cluster.
        
        Arguments:
            :dict state: current state
            :data_id:    sample index
            :cid:        cluster index
        """
        state['ev_in_cl'][cid].append(data_id)

    def update_alpha(self, state, burnin = 100):
        '''
        Updates concentration parameter using a Metropolis-Hastings sampling scheme.
        
        Arguments:
            :dict state: current state
            :int burnin: MH burnin
        
        Returns:
            :double: new concentration parametere value
        '''
        a_old = state['alpha_']
        n     = state['Ntot']
        K     = len(state['cluster_ids_'])
        for _ in range(burnin):
            a_new = a_old + self.rdstate.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(random.uniform()):
                    a_old = a_new
        return a_old
    
    def gibbs_step(self, state):
        """
        Computes a single Gibbs step (updates all the sample assignments using conditional probabilities)
        
        Arguments:
            :dict state: current state to update
        """
        self.update_draws()
        state['alpha_'] = self.update_alpha(state)
        self.alpha_samples.append(state['alpha_'])
        pairs = zip(state['data_'], state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            self.drop_from_cluster(state, data_id, cid)
            self.prune_clusters(state)
            cid = self.sample_assignment(data_id, state)
            self.add_to_cluster(state, data_id, cid)
            state['assignment'][data_id] = cid
        self.n_clusters.append(len(state['cluster_ids_']))
    
    def sample_mixture_parameters(self, state):
        '''
        Draws a mixture sample (weights, means and std) using conditional probabilities. Eq. (3.7)
        
        Arguments:
            :dict state: current state
        '''
        alpha = [len(state['ev_in_cl'][cid]) + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = self.rdstate.dirichlet(alpha).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            events = [self.posterior_draws[j] for j in state['ev_in_cl'][cid]]
            m, s = sample_point(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max, burnin = 1000, seed = self.seed)
            components[i] = {'mean': m, 'sigma': s, 'weight': weights[i]}
        self.mixture_samples.append(components)
    
    
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.mass_samples)))
        print('Concentration parameters:\ngamma0 = {0}'.format(self.alpha0))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('------------------------')
        return

    def postprocess(self):
        """
        Plots the inferred distribution and saves draws.
        """
        # mass values
        app  = np.linspace(self.m_min, self.m_max_plot, 1000)
        da = app[1]-app[0]
        percentiles = [50, 5,16, 84, 95]
        p = {}
        fig = plt.figure()
        fig.suptitle('Observed mass function')
        ax  = fig.add_subplot(111)
        
        # if provided (simulation) plots true masses histogram
        if self.true_masses is not None:
            truths = np.genfromtxt(self.true_masses, names = True)
            ax.hist(truths['m'], bins = int(np.sqrt(len(truths['m']))), histtype = 'step', density = True, label = r"\textsc{True masses}")
        
        # evaluates log probabilities in mass space
        prob = []
        for ai in app:
            a = self.transform(ai)
            prob.append([logsumexp([log_norm(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) - log_norm(a, 0, 1) for sample in self.mixture_samples])
        
        self.prob_draws = np.array(prob).T
        self.m_vals     = app

        # Saves interpolant functions into json file
        name = self.output_events + '/posterior_functions_mf_'
        extension ='.json'
        x = 0
        fileName = name + str(x) + extension
        while os.path.exists(fileName):
            x = x + 1
            fileName = name + str(x) + extension
        
        j_dict = {str(m): list(draws) for m, draws in zip(app, prob)}
        with open(fileName, 'w') as jsonfile:
            json.dump(j_dict, jsonfile)
        
        # computes percentiles
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = 1)
        normalisation = np.sum(np.exp(p[50])*da)
        for perc in percentiles:
            p[perc] = p[perc] - np.log(normalisation)
        self.sample_probs = prob
            
        #saves median and CR
        self.median_mf = np.array(p[50])
        names = ['m'] + [str(perc) for perc in percentiles]
        np.savetxt(self.output_events + '/log_rec_obs_prob_mf.txt', np.array([app, p[50], p[5], p[16], p[84], p[95]]).T, header = ' '.join(names))
        
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        for perc in percentiles:
            p[perc] = p[perc]/normalisation
        
        # plots median and CR of reconstructed probability density
        ax.fill_between(app, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'steelblue', label = r"\textsc{Reconstructed}", zorder = 100)
        
        # if simulation, plots true probability density
        if self.injected_density is not None:
            norm = np.sum([self.injected_density(a)*(app[1]-app[0]) for a in app])
            density = np.array([self.injected_density(a)/norm for a in app])
            ax.plot(app, density, color = 'k', marker = '', linewidth = 0.8, label = r"\textsc{Simulated - Observed}")
        if not self.unit == '':
            ax.set_xlabel('${0}\ [{1}]$'.format(self.var_symbol, self.unit))
        else:
            ax.set_xlabel('${0}$'.format(self.var_symbol))
        ax.set_ylabel('$p({0})$'.format(self.var_symbol))
        ax.set_xlim(self.m_min_plot, self.m_max_plot)
        ax.grid(True,dashes=(1,3))
        ax.legend(loc=0,frameon=False,fontsize=10)
        plt.savefig(self.output_events + '/obs_mass_function.pdf', bbox_inches = 'tight')
        ax.set_yscale('log')
        ax.set_ylim(np.min(p[50]))
        plt.savefig(self.output_events + '/log_obs_mass_function.pdf', bbox_inches = 'tight')
        
        # saves mixture samples into json file
        name = self.output_events + '/posterior_mixtures_mf_'
        extension ='.json'
        x = 0
        fileName = name + str(x) + extension
        while os.path.exists(fileName):
            x = x + 1
            fileName = name + str(x) + extension
            
        j_dict = {str(i): sample for i, sample in enumerate(self.mixture_samples)}
        with open(fileName, 'w') as jsonfile:
            json.dump(j_dict, jsonfile)
        
        # plots number of clusters
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_events+'n_clusters_mf.pdf', bbox_inches='tight')
        
        # plots concentration parameter
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(self.output_events+'/gamma_mf.pdf', bbox_inches='tight')
        
        # if simulation, computes Jensen-Shannon distance
        if self.injected_density is not None:
            ent = [js(np.exp(s(app)), density) for s in log_draws_interp]
            JSD = {}
            for perc in percentiles:
                JSD[perc] = np.percentile(ent, perc, axis = 0)
            print('Jensen-Shannon distance: {0}+{1}-{2} nats'.format(JSD[50], JSD[95]-JSD[50], JSD[50]-JSD[5]))
            np.savetxt(output + '/JSD.txt', np.array([JSD[50], JSD[5], JSD[16], JSD[84], JSD[95]]), header = '50 5 16 84 95')
        
    
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        self.output_events = self.output_folder
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.run_sampling()
        self.postprocess()
        if self.diagnostic:
            self.run_diagnostic()
        return

    def checkpoint(self):
        """
        Saves to file recent draws
        """

        app  = np.linspace(self.m_min, self.m_max_plot, 1000)
        da = app[1]-app[0]
        try:
            with open(self.output_events + '/checkpoint.json', 'r') as jsonfile:
                samps = json.load(jsonfile)
        except:
            samps = {str(m):[] for m in app}
        
        # evaluates probabilities in mass space
        prob = []
        for ai in app:
            a = self.transform(ai)
            prob.append([logsumexp([log_norm(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) - log_norm(a, 0, 1) for sample in self.mixture_samples[-self.ncheck:]])
        
        # saves new samples
        for m, p in zip(app, prob):
            samps[str(m)] = samps[str(m)] + p
        with open(self.output_events + '/checkpoint.json', 'w') as jsonfile:
            json.dump(samps, jsonfile)

    def run_sampling(self):
        """
        Runs the sampling algorithm - Listing 2
        """
        self.state = self.initial_state()
        for i in range(self.burnin):
            print('\rBURN-IN MF: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step(self.state)
        print('\n', end = '')
        for i in range(self.n_draws):
            print('\rSAMPLING MF: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.step):
                self.gibbs_step(self.state)
            self.sample_mixture_parameters(self.state)
            if (i+1) % self.ncheck == 0:
                self.checkpoint()
        print('\n', end = '')
        return

    def run_diagnostic(self):
        self.convergence()
        self.autocorrelation()
        return
    
    def convergence(self):
        dist = np.zeros(len(self.prob_draws) - 1)
        idx  = np.arange(len(self.prob_draws) - 1)
        for i in idx:
            dist[i] = entropy(np.exp(self.prob_draws[i]), np.exp(self.prob_draws[i+1]))
        fig_conv, ax_conv = plt.subplots()
        ax_conv.plot(idx + 1, dist, ls = '--')
        ax_conv.set_xlabel('$N$')
        ax_conv.set_ylabel('$D_{JS}(p_{n}, p_{n+1})$')
        ax.grid(True,dashes=(1,3))
        fig_conv.savefig(self.output_events + '/convergence_mf.pdf', bbox_inches = 'tight')
        return


