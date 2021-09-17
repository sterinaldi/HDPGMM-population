import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle

from mpl_toolkits.mplot3d import Axes3D
from corner import corner

from collections import namedtuple, Counter

from numpy import random
from numpy.linalg import det, inv

from scipy import stats
from scipy.stats import entropy, gamma
from scipy.stats import multivariate_t as student_t
from scipy.stats import multivariate_normal as mn
from scipy.spatial.distance import jensenshannon as js
from scipy.special import logsumexp, betaln, gammaln, erfinv
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import nquad
from scipy.linalg import det

from sampler_component_pars import sample_point, MH_single_event
import itertools

from time import perf_counter

import ray
from ray.util import ActorPool
from ray.util.multiprocessing import Pool

from numba import jit

from utils import integrand, compute_norm_const, log_norm, scalar_log_norm, make_sym_matrix, cartesian_to_celestial

import cpnest.model

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
        return logP
    
    def log_likelihood(self, x):
#        vals = np.array([x[name] for name in self.names])
        return integrand(x.values, self.events, self.logN_cnst, self.dim)

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
                       m_min,
                       m_max,
                       samp_settings, # burnin, draws, step (list)
                       samp_settings_ev = None,
                       alpha0 = 1,
                       gamma0 = 1,
                       hyperpriors_ev = [1,1], #a, V
                       verbose = True,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       process_events = True,
                       n_parallel_threads = 8,
                       injected_density = None,
                       true_masses = None,
                       names = None,
                       var_names = None,
                       n_samp_to_plot = 2000
                       ):
        
        self.burnin_mf, self.n_draws_mf, self.step_mf = samp_settings
        if samp_settings_ev is not None:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings_ev
        else:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings
        self.events = events
        sample_min = np.min([np.min(a, axis = 0) for a in self.events], axis = 0)
        sample_max = np.max([np.max(a, axis = 0) for a in self.events], axis = 0)
        self.m_min   = np.minimum(m_min, sample_min)
        self.m_max   = np.maximum(m_max, sample_max)
        self.m_max_plot = m_max
        # probit
        self.upper_bounds = np.array([x if not x == 0 else -1e-4 for x in np.array([x*(1+1e-4) if x > 0 else x*(1-1e-4) for x in self.m_max])])
        self.lower_bounds = np.array([x if not x == 0 else 1e-4 for x in np.array([x*(1-1e-4) if x > 0 else x*(1+1e-4) for x in self.m_min])])
        self.transformed_events = [self.transform(ev) for ev in events]
        self.t_min = self.transform([self.m_min])
        self.t_max = self.transform([self.m_max])
        # DP
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        # student-t
        if hyperpriors_ev is not None:
            self.a_ev, self.V_ev = hyperpriors_ev
        else:
            self.a_ev, self.V_ev = [1,1]
        # miscellanea
        self.output_folder = output_folder
        self.icn = initial_cluster_number
        self.event_samplers = []
        self.verbose = verbose
        self.process_events = process_events
        self.n_parallel_threads = n_parallel_threads
        self.injected_density = injected_density
        self.true_masses = true_masses
        self.output_recprob = self.output_folder + '/reconstructed_events/pickle/'
        self.dim = len(self.events[-1][-1])
        self.n_samp_to_plot = n_samp_to_plot
        self.var_names = var_names
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
        
        cdf = (np.array(samples).T - np.atleast_2d([self.lower_bounds]).T)/np.array([self.upper_bounds - self.lower_bounds]).T
        new_samples = np.sqrt(2)*erfinv(2*cdf-1).T
        if len(new_samples) == 1:
            return new_samples[0]
        return new_samples
    
    def initialise_samplers(self, marker):
        '''
        Initialises n_parallel_threads instances of SE_Sampler class
        '''
        event_samplers = []
        for i, (ev, t_ev) in enumerate(zip(self.events[marker:marker+self.n_parallel_threads], self.transformed_events[marker:marker+self.n_parallel_threads])):
            event_samplers.append(SE_Sampler.remote(
                                            mass_samples           = t_ev,
                                            event_id               = self.names[marker+i],
                                            burnin                 = self.burnin_ev,
                                            n_draws                = self.n_draws_ev,
                                            step                   = self.step_ev,
                                            m_min                  = np.min(ev, axis = 0),
                                            m_max                  = np.max(ev, axis = 0),
                                            t_min                  = np.min(t_ev, axis = 0),
                                            t_max                  = np.max(t_ev, axis = 0),
                                            real_masses            = ev,
                                            alpha0                 = self.alpha0,
                                            a                      = self.a_ev,
                                            V                      = self.V_ev,
                                            glob_m_max             = self.m_max,
                                            glob_m_min             = self.m_min,
                                            upper_bounds           = self.upper_bounds,
                                            lower_bounds           = self.lower_bounds,
                                            output_folder          = self.output_folder,
                                            verbose                = self.verbose,
                                            initial_cluster_number = self.icn,
                                            transformed            = True,
                                            var_names              = self.var_names
                                            ))
        return event_samplers
        
    def run_event_sampling(self):
        '''
        Performs analysis on each event
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
            #guardare ray.wait
            for s in pool.map(lambda a, v: a.run.remote(), range(len(tasks))):
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
        prob_files = [self.output_recprob+f for f in os.listdir(self.output_recprob) if f.startswith('posterior_functions')]
        prob_files.sort(key = natural_keys)
        for prob in prob_files:
            sampfile = open(prob, 'rb')
            samps = pickle.load(sampfile)
            self.posterior_functions_events.append(samps)
    
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
        Creates an instance of MF_Sampler class
        '''
        self.load_mixtures()
        self.mf_folder = self.output_folder+'/mass_function/'
        if not os.path.exists(self.mf_folder):
            os.mkdir(self.mf_folder)
        flattened_transf_ev = np.array([x for ev in self.transformed_events for x in ev])
        sampler = MF_Sampler(self.posterior_functions_events,
                       self.dim,
                       self.burnin_mf,
                       self.n_draws_mf,
                       self.step_mf,
                       alpha0 = self.gamma0,
                       m_min = self.m_min,
                       m_max = self.m_max,
                       t_min = self.t_min,
                       t_max = self.t_max,
                       output_folder = self.mf_folder,
                       initial_cluster_number = min([self.icn, len(self.posterior_functions_events)]),
                       injected_density = self.injected_density,
                       true_masses = self.true_masses,
                       sigma_min = np.std(flattened_transf_ev)/16.,
                       sigma_max = np.std(flattened_transf_ev)/3.,
                       n_parallel_threads = self.n_parallel_threads,
                       transformed = True,
                       n_samp_to_plot = self.n_samp_to_plot,
                       upper_bounds = self.upper_bounds,
                       lower_bounds = self.lower_bounds
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
        

@jit(forceobj=True)
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
    A = gammaln(x)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return float(A - B - C - D + E)
    
#@ray.remote
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
    def __init__(self, mass_samples,
                       event_id,
                       burnin,
                       n_draws,
                       step,
                       m_min,
                       m_max,
                       t_min = -1,
                       t_max = 1,
                       real_masses = None,
                       alpha0 = 1,
                       a = 1,
                       V = 1,
                       glob_m_max = None,
                       glob_m_min = None,
                       upper_bounds = None,
                       lower_bounds = None,
                       output_folder = './',
                       verbose = True,
                       initial_cluster_number = 5.,
                       transformed = False,
                       var_names = None,
                       vol_rec = False
                       ):
        # New seed for each subprocess
        random.RandomState(seed = os.getpid())
        if real_masses is None:
            self.initial_samples = mass_samples
        else:
            self.initial_samples = real_masses
        self.e_ID    = event_id
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        if glob_m_min is None:
            self.glob_m_min = m_min
        else:
            self.glob_m_min = glob_m_min
            
        if glob_m_max is None:
            self.glob_m_max = m_max
        else:
            self.glob_m_max = glob_m_max
        
        if upper_bounds is None:
            self.upper_bounds = np.array([x if not x == 0 else -1e-9 for x in np.array([x*(1+1e-9) if x > 0 else x*(1-1e-9) for x in self.glob_m_max])])
        else:
            self.upper_bounds = upper_bounds
        if lower_bounds is None:
            self.lower_bounds = np.array([x if not x == 0 else 1e-9 for x in np.array([x*(1-1e-9) if x > 0 else x*(1+1e-9) for x in self.glob_m_min])])
        else:
            self.lower_bounds = lower_bounds
        
        if transformed:
            self.mass_samples = mass_samples
            self.t_max   = t_max
            self.t_min   = t_min
        else:
            self.mass_samples = self.transform(mass_samples)
            self.t_max        = self.transform([self.m_max])
            self.t_min        = self.transform([self.m_min])
            
        self.sigma_max = np.var(self.mass_samples, axis = 0)
        self.dim = len(mass_samples[-1])
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.L  = (np.std(self.mass_samples, axis = 0)/9.)**2*np.identity(self.dim)
        self.nu  = np.max([a,self.dim])
        self.k  = V
        self.mu = np.atleast_2d(np.mean(self.mass_samples, axis = 0))
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean cov N')
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.verbose = verbose
        self.alpha_samples = []
        self.var_names = var_names
        self.vol_rec = vol_rec
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        
        Arguments:
            :float or np.ndarray samples: mass sample(s) to transform
        Returns:
            :float or np.ndarray: transformed sample(s)
        '''
        cdf = (np.array(samples).T - np.atleast_2d(self.lower_bounds).T)/np.array([self.upper_bounds - self.lower_bounds]).T
        new_samples = np.sqrt(2)*erfinv(2*cdf-1).T
        if len(new_samples) == 1:
            return new_samples[0]
        return new_samples
    
        
    def initial_state(self, samples):
        '''
        Create initial state
        '''
        assign = [a%int(self.icn) for a in range(len(samples))]
        cluster_ids = list(np.arange(int(np.max(assign)+1)))
        samp = np.copy(samples)
        state = {
            'cluster_ids_': cluster_ids,
            'data_': samp,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(samples),
            'hyperparameters_': {
                "L": self.L,
                "k": self.k,
                "nu": self.nu,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': assign,
            'pi': {cid: self.alpha0 / self.icn for cid in cluster_ids},
            }
        self.update_suffstats(state)
        return state
    
    def update_suffstats(self, state):
        for cluster_id, N in Counter(state['assignment']).items():
            points_in_cluster = [x for x, cid in zip(state['data_'], state['assignment']) if cid == cluster_id]
            mean = np.atleast_2d(np.array(points_in_cluster).mean(axis = 0))
            cov  = np.cov(np.array(points_in_cluster), rowvar = False)
            M    = len(points_in_cluster)
            state['suffstats'][cluster_id] = self.SuffStat(mean, cov, M)
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster
        '''
        if cluster_id == "new":
            ss = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        else:
            ss  = state['suffstats'][cluster_id]
            
        x = state['data_'][data_id]
        mean = ss.mean
        S = ss.cov
        N = ss.N
        # Update hyperparameters
        k_n  = state['hyperparameters_']["k"] + N
        mu_n = np.atleast_2d((state['hyperparameters_']["mu"]*state['hyperparameters_']["k"] + N*mean)/k_n)
        nu_n = state['hyperparameters_']["nu"] + N
        L_n  = state['hyperparameters_']["L"]*state['hyperparameters_']["k"] + S*N + state['hyperparameters_']["k"]*N*np.matmul((mean - state['hyperparameters_']["mu"]).T, (mean - state['hyperparameters_']["mu"]))/k_n
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
        return self.SuffStat(mean, cov, ss.N+1)


    def remove_datapoint_from_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        if ss.N == 1:
            return(self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) - np.matmul(x.T, x))/(ss.N-1) - np.matmul(mean.T, mean)
        return self.SuffStat(mean, cov, ss.N-1)
        
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
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
        be assigned to cluster_id given the current state.
        """
        if cluster_id == "new":
            return np.log(state["alpha_"])
        else:
            return np.log(state['suffstats'][cluster_id].N)

    def create_cluster(self, state):
        state["num_clusters_"] += 1
        cluster_id = max(state['suffstats'].keys()) + 1
        state['suffstats'][cluster_id] = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        state['cluster_ids_'].append(cluster_id)
        return cluster_id

    def destroy_cluster(self, state, cluster_id):
        state["num_clusters_"] -= 1
        del state['suffstats'][cluster_id]
        state['cluster_ids_'].remove(cluster_id)
        
    def prune_clusters(self,state):
        for cid in state['cluster_ids_']:
            if state['suffstats'][cid].N == 0:
                self.destroy_cluster(state, cid)

    def sample_assignment(self, data_id, state):
        """
        Sample new assignment from marginal distribution.
        If cluster is "new", create a new cluster.
        """
        scores = self.cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = random.RandomState().choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster(state)
        else:
            return int(cid)

    def update_alpha(self, state, thinning = 100):
        '''
        Update concentration parameter
        '''
        a_old = state['alpha_']
        n     = state['Ntot']
        K     = len(state['cluster_ids_'])
        for _ in range(thinning):
            a_new = a_old + random.RandomState().uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old)
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new)
                if logP_new - logP_old > np.log(random.uniform()):
                    a_old = a_new
        return a_old

    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Gaussian Mixture Model
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
        self.n_clusters.append(len(state['cluster_ids_']))
    
    def sample_mixture_parameters(self, state):
        '''
        Draws a mixture sample
        '''
        ss = state['suffstats']
        alpha = [ss[cid].N + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = stats.dirichlet(alpha).rvs(size=1).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            mean = ss[cid].mean
            S = ss[cid].cov
            N     = ss[cid].N
            k_n  = state['hyperparameters_']["k"] + N
            mu_n = np.atleast_2d((state['hyperparameters_']["mu"]*state['hyperparameters_']["k"] + N*mean)/k_n)
            nu_n = state['hyperparameters_']["nu"] + N
            L_n  = state['hyperparameters_']["L"] + S*N + state['hyperparameters_']["k"]*N*np.matmul((mean - state['hyperparameters_']["mu"]).T, (mean - state['hyperparameters_']["mu"]))/k_n
            # Update t-parameters
            s = stats.invwishart(df = nu_n, scale = L_n).rvs()
            t_df    = nu_n - self.dim + 1
            t_shape = L_n*(k_n+1)/(k_n*t_df)
            m = student_t(df = t_df, loc = mu_n.flatten(), shape = t_shape).rvs()
            components[i] = {'mean': m, 'cov': s, 'weight': weights[i], 'N': N}
        self.mixture_samples.append(components)
    
    def run_sampling(self):
        state = self.initial_state(self.mass_samples)
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

    def postprocess(self, n_points = 30):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        
        lower_bound = np.maximum(self.m_min, self.glob_m_min)
        upper_bound = np.minimum(self.m_max, self.glob_m_max)
        points = [np.linspace(l, u, n_points) for l, u in zip(lower_bound, upper_bound)]
        log_vol_el = np.sum([np.log(v[1]-v[0]) for v in points])
        gridpoints = np.array(list(itertools.product(*points)))
        percentiles = [50] #[5,16, 50, 84, 95]
        
        p = {}
        
#        fig = plt.figure()
#        ax  = fig.add_subplot(111)
#        ax.hist(self.initial_samples, bins = int(np.sqrt(len(self.initial_samples))), histtype = 'step', density = True)
        prob = []
        for i, ai in enumerate(gridpoints):
            a = self.transform([ai])
            #FIXME: scrivere log_norm in cython
            print('\rGrid evaluation: {0}/{1}'.format(i+1, n_points**self.dim), end = '')
            logsum = np.sum([scalar_log_norm(par,0, 1) for par in a])
            prob.append([logsumexp([log_norm(a, component['mean'], component['cov']) + np.log(component['weight']) for component in sample.values()]) - logsum for sample in self.mixture_samples])
        prob = np.array(prob).reshape([n_points for _ in range(self.dim)] + [self.n_draws])
        
        log_draws_interp = []
        for i in range(self.n_draws):
            log_draws_interp.append(RegularGridInterpolator(points, prob[...,i] - logsumexp(prob[...,i] + log_vol_el)))
        
        picklefile = open(self.output_posteriors + '/posterior_functions_{0}.pkl'.format(self.e_ID), 'wb')
        pickle.dump(log_draws_interp, picklefile)
        picklefile.close()
        
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = -1)
        normalisation = logsumexp(p[50] + log_vol_el)
        for perc in percentiles:
            p[perc] = p[perc] - normalisation
            
        names = ['m'] + [str(perc) for perc in percentiles]
        
#        np.savetxt(self.output_recprob + '/log_rec_prob_{0}.txt'.format(self.e_ID), np.array([app, p[5], p[16], p[50], p[84], p[95]]).T, header = ' '.join(names))
        picklefile = open(self.output_recprob + '/log_rec_prob_{0}.pkl'.format(self.e_ID), 'wb')
        pickle.dump(RegularGridInterpolator(points, p[50]), picklefile)
        picklefile.close()
        
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = -1))
        for perc in percentiles:
            p[perc] = p[perc]/np.exp(normalisation)
            
        prob = np.array(prob)
        
        #FIXME: Jensen-Shannon distance in n dimensions? js works only on one axis
#        ent = []
#        for i in range(np.shape(prob)[1]):
#            sample = np.exp(prob[:,i])
#            ent.append(js(sample,p[50]))
#        mean_ent = np.mean(ent)
#        np.savetxt(self.output_entropy + '/KLdiv_{0}.txt'.format(self.e_ID), np.array(ent), header = 'mean JS distance = {0}'.format(mean_ent))
        
        picklefile = open(self.output_pickle + '/posterior_functions_{0}.pkl'.format(self.e_ID), 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        self.sample_probs = prob
        self.median = np.array(p[50])
        self.points = points
        
        samples_to_plot = np.array(MH_single_event(RegularGridInterpolator(points, p[50]), upper_bound, lower_bound, len(self.mass_samples)))
        if self.vol_rec:
            self.initial_samples = np.array([cartesian_to_celestial(np.array([x,y,z])) for x,y,z in zip(self.initial_samples[:,0], self.initial_samples[:,1], self.initial_samples[:,2])])
            samples_to_plot = np.array([cartesian_to_celestial(np.array([x,y,z])) for x,y,z in zip(samples_to_plot[:,0], samples_to_plot[:,1], samples_to_plot[:,2])])
        c = corner(self.initial_samples, color = 'orange', labels = self.var_names, hist_kwargs={'density':True})
        c = corner(samples_to_plot, fig = c, color = 'blue', labels = self.var_names, hist_kwargs={'density':True})
        c.savefig(self.output_pltevents + '/{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_n_clusters+'n_clusters_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(self.alpha_folder+'/alpha_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
    
    def make_folders(self):
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
        if not os.path.exists(self.output_events + '/pickle/'):
            os.mkdir(self.output_events + '/pickle/')
        self.output_pickle = self.output_events + '/pickle/'
        if not os.path.exists(self.output_events + '/posteriors/'):
            os.mkdir(self.output_events + '/posteriors/')
        self.output_posteriors = self.output_events + '/posteriors/'
        if not os.path.exists(self.output_events + '/entropy/'):
            os.mkdir(self.output_events + '/entropy/')
        self.output_entropy = self.output_events + '/entropy/'
        if not os.path.exists(self.output_events + '/alpha/'):
            os.mkdir(self.output_events + '/alpha/')
        self.alpha_folder = self.output_events + '/alpha/'
        return
        

    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
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
                       step,
                       m_min,
                       m_max,
                       t_min,
                       t_max,
                       alpha0 = 1,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       upper_bounds = None,
                       lower_bounds = None,
                       injected_density = None,
                       true_masses = None,
                       sigma_min = 0.003,
                       sigma_max = 0.7,
                       n_parallel_threads = 1,
                       ncheck = 5,
                       transformed = False,
                       var_names = None,
                       n_samp_to_plot = 2000,
                       n_points = 30 # np.linspace(min, max, n_points) for each variable
                       ):
                       
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        if transformed:
            self.t_min = t_min
            self.t_max = t_max
        else:
            self.t_min = self.transform([m_min])
            self.t_max = self.transform([m_max])
         
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.posterior_functions_events = posterior_functions_events
        # DP parameters
        self.alpha0 = alpha0
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.true_masses = true_masses
        self.n_parallel_threads = n_parallel_threads
        self.alpha_samples = []
        self.ncheck = ncheck
        self.points = [np.linspace(l, u, n_points) for l, u in zip(self.lower_bounds, self.upper_bounds)]
        self.log_vol_el = np.sum([np.log(v[1]-v[0]) for v in self.points])
        self.gridpoints = np.array(list(itertools.product(*self.points)))
        self.dim = dim
        self.n_samp_to_plot = n_samp_to_plot
        self.p = Pool(n_parallel_threads)
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        
        Arguments:
            :float or np.ndarray samples: mass sample(s) to transform
        Returns:
            :float or np.ndarray: transformed sample(s)
        '''
        cdf = (np.array(samples).T - np.atleast_2d([self.lower_bounds]).T)/np.array([self.upper_bounds - self.lower_bounds]).T
        new_samples = np.sqrt(2)*erfinv(2*cdf-1).T
        if len(new_samples) == 1:
            return new_samples[0]
        return new_samples
    
    def initial_state(self):
        '''
        Creates initial state
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
            'pi': {cid: self.alpha0 / self.icn for cid in cluster_ids},
            'ev_in_cl': {cid: list(np.where(np.array(assign) == cid)[0]) for cid in cluster_ids},
            'logL_D': {cid: None for cid in cluster_ids}
            'starting_points': {}
            }
        for cid in state['cluster_ids_']:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
            n = len(events)
            state['logL_D'][cid] = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max)
        state['logL_D']["new"] = self.log_numerical_predictive([], self.t_min, self.t_max, self.sigma_min, self.sigma_max)
        return state
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster
        '''
        if cluster_id == "new":
            events = []
            return -logsumexp([np.log(tmax - tmin) for tmin, tmax in zip(t_min, t_max)]), -logsumexp([np.log(tmax - tmin) for tmin, tmax in zip(t_min, t_max)])
        else:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cluster_id]]
        n = len(events)
        events.append(self.posterior_draws[data_id])
        logL_D = state['logL_D'][cluster_id] #denominator
        logL_N, starting_point = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max) #numerator
        state['starting_points'][cluster_id] = starting_point
        return logL_N - logL_D, logL_N

    def log_numerical_predictive(self, events, t_min, t_max, sigma_min, sigma_max):
        logN_cnst = compute_norm_const(np.zeros(self.dim), np.identity(self.dim), events) + logsumexp([np.log(tmax - tmin) for tmin, tmax in zip(t_min, t_max)]) + np.log(sigma_max - sigma_min)*self.dim*(self.dim - 1)/2.
        bounds = [[tmin, tmax] for tmin, tmax in zip(t_min, t_max)] + [[sigma_min, sigma_max] for _ in range(int(self.dim*(self.dim - 1)/2.))]
        integrator = Integrator(bounds, events, logN_cnst, self.dim)
        work = cpnest.CPNest(integrator, verbose = 0, nlive = self.dim*(self.dim+3)+1, maxmcmc = 1000, nensemble = 1, output = self.output_folder)
        work.run()
        return work.NS.logZ, work.posterior_samples[-1]
        #I, dI, d = nquad(integrand, bounds, args = [events, logN_cnst, self.dim])
        #return np.log(I) + logN_cnst
    
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
        """
        cluster_ids = list(state['ev_in_cl'].keys()) + ['new']
        # can't pickle injected density
#        saved_injected_density = self.injected_density
        self.injected_density  = None
#        with Pool(self.n_parallel_threads) as p:
        output = self.p.map(self.compute_score, [[data_id, cid, state] for cid in cluster_ids])
        scores = {out[0]: out[1] for out in output}
        self.numerators = {out[0]: out[2] for out in output}
#        self.injected_density = saved_injected_density
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores
        
    def compute_score(self, args):
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
        be assigned to cluster_id given the current state.
        """
        if cluster_id == "new":
            return np.log(state["alpha_"])
        else:
            if len(state['ev_in_cl'][cluster_id]) == 0:
                return -np.inf
            return np.log(len(state['ev_in_cl'][cluster_id]))

    def create_cluster(self, state):
        state["num_clusters_"] += 1
        cluster_id = max(state['cluster_ids_']) + 1
        state['cluster_ids_'].append(cluster_id)
        state['ev_in_cl'][cluster_id] = []
        state['starting_points'][cluster_id] = state['starting_points']["new"]
        return cluster_id

    def destroy_cluster(self, state, cluster_id):
        state["num_clusters_"] -= 1
        state['cluster_ids_'].remove(cluster_id)
        state['ev_in_cl'].pop(cluster_id)
        
    def prune_clusters(self,state):
        for cid in state['cluster_ids_']:
            if len(state['ev_in_cl'][cid]) == 0:
                self.destroy_cluster(state, cid)

    def sample_assignment(self, data_id, state):
        """
        Sample new assignment from marginal distribution.
        If cluster is "new", create a new cluster.
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
        draws = []
        for posterior_samples in self.posterior_functions_events:
            draws.append(posterior_samples[random.randint(len(posterior_samples))])
        self.posterior_draws = draws
    
    def drop_from_cluster(self, state, data_id, cid):
        state['ev_in_cl'][cid].remove(data_id)
        events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
        n = len(events)
        state['logL_D'][cid] = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max)

    def add_to_cluster(self, state, data_id, cid):
        state['ev_in_cl'][cid].append(data_id)

    def update_alpha(self, state, trimming = 100):
        '''
        Updetes concentration parameter
        '''
        a_old = state['alpha_']
        n     = state['Ntot']
        K     = len(state['cluster_ids_'])
        for _ in range(trimming):
            a_new = a_old + random.RandomState().uniform(-1,1)*0.5#.gamma(1)
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old)
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new)
                if logP_new - logP_old > np.log(random.uniform()):
                    a_old = a_new
        return a_old
    
    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Gaussian Mixture Model
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
        alpha = [len(state['ev_in_cl'][cid]) + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = stats.dirichlet(alpha).rvs(size=1).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            events = [self.posterior_draws[j] for j in state['ev_in_cl'][cid]]
            m, s = sample_point(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max, self.n_dim, state['starting_points'][cid])
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

    def postprocess(self, n_points = 30):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        
#        app  = np.linspace(self.m_min*1.1, self.m_max_plot, 1000)
#        da = app[1]-app[0]
        percentiles = [50]#[50, 5,16, 84, 95]
        
        p = {}
        
        prob = []
        for i, ai in enumerate(self.gridpoints):
            a = self.transform([ai])
            #FIXME: scrivere log_norm in cython
            print('\rGrid evaluation: {0}/{1}'.format(i+1, n_points**self.dim), end = '')
            logsum = np.sum([scalar_log_norm(par,0, 1) for par in a])
            prob.append([logsumexp([log_norm(a, component['mean'], component['cov']) + np.log(component['weight']) for component in sample.values()]) - logsum for sample in self.mixture_samples])
        prob = np.array(prob).reshape([n_points for _ in range(self.dim)] + [self.n_draws])

        log_draws_interp = []
        for i in range(self.n_draws):
            log_draws_interp.append(RegularGridInterpolator(self.points, prob[...,i] - logsumexp(prob[...,i] + self.log_vol_el)))
        
        name = self.output_events + '/posterior_functions_mf_'
        extension ='.pkl'
        x = 0
        fileName = name + str(x) + extension
        while os.path.exists(fileName):
            x = x + 1
            fileName = name + str(x) + extension
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
#        np.savetxt(self.output_events + '/log_rec_obs_prob_mf.txt', np.array([app, p[50], p[5], p[16], p[84], p[95]]).T, header = ' '.join(names))
        picklefile = open(self.output_recprob + '/log_rec_prob_mf.pkl', 'wb')
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
        
        name = self.output_events + '/posterior_mixtures_mf_'
        extension ='.pkl'
        x = 0
        fileName = name + str(x) + extension
        while os.path.exists(fileName):
            x = x + 1
            fileName = name + str(x) + extension
        picklefile = open(fileName, 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_events+'n_clusters_mf.pdf', bbox_inches='tight')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(self.output_events+'/gamma_mf.pdf', bbox_inches='tight')
        #FIXME: JS multidimensionale? (Vedi sopra)
#        inj = np.array([self.injected_density(ai)/norm for ai in app])
#        ent = js(p[50], inj)
#        print('Jensen-Shannon distance: {0} nats'.format(ent))
#        np.savetxt(self.output_events + '/relative_entropy.txt', np.array([ent]))
        
    
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """

        # reconstructed events
        self.output_events = self.output_folder
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.run_sampling()
        self.postprocess()
        return

    def checkpoint(self):

        try:
            picklefile = open(self.output_events + '/checkpoint.pkl', 'rb')
            samps = pickle.load(picklefile)
            picklefile.close()
        except:
            samps = []
        
        prob = []
        for ai in self.gridpoints:
            a = self.transform([ai])
            #FIXME: scrivere log_norm in cython
            logsum = np.sum([log_norm(par,0, 1) for par in a])
            prob.append([logsumexp([log_norm(a, component['mean'], component['cov']) + np.log(component['weight']) for component in sample.values()]) - logsum for sample in self.mixture_samples[-self.ncheck:]])
        prob = np.array(prob).reshape([n_points for _ in range(self.dim)] + [self.n_draws])

        log_draws_interp = []
        for i in range(self.n_draws):
            log_draws_interp.append(RegularGridInterpolator(self.points, prob[...,i] - logsumexp(prob[...,i] + self.log_vol_el)))
        
        samps = samps + log_draws_interp
        picklefile = open(self.output_events + '/checkpoint.pkl', 'wb')
        pickle.dump(samps, picklefile)
        picklefile.close()

    def run_sampling(self):
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
        
