import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle

from collections import namedtuple, Counter
from numpy import random

from scipy import stats
from scipy.stats import entropy, gamma
from scipy.spatial.distance import jensenshannon as js
from scipy.special import logsumexp, betaln, gammaln, erfinv
from scipy.interpolate import interp1d
from scipy.integrate import dblquad

from sampler_component_pars import sample_point

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
                       hyperpriors_ev = [1,1/4.], #a, V
                       m_min = 5,
                       m_max = 70,
                       verbose = True,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       process_events = True,
                       n_parallel_threads = 8,
                       injected_density = None,
                       true_masses = None,
                       names = None,
                       ):
        
        self.burnin_mf, self.n_draws_mf, self.step_mf = samp_settings
        if samp_settings_ev is not None:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings_ev
        else:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings
        self.events = events
        sample_min = np.min([np.min(a) for a in self.events])
        sample_max = np.max([np.max(a) for a in self.events])
        self.m_min   = min([m_min, sample_min])
        self.m_max   = max([m_max, sample_max])
        self.m_max_plot = m_max
        # probit
        self.transformed_events = [self.transform(ev) for ev in events]
        self.t_min = self.transform(self.m_min)
        self.t_max = self.transform(self.m_max)
        # DP
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        # student-t
        if hyperpriors_ev is not None:
            self.a_ev, self.V_ev = hyperpriors_ev
        else:
            self.a_ev, self.V_ev = [1,1/4.]
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
        cdf = (samples - cdf_bounds[0])/(cdf_bounds[1]-cdf_bounds[0])
        new_samples = np.sqrt(2)*erfinv(2*cdf-1)
        return new_samples
    
    def initialise_samplers(self, marker):
        '''
        Initialises n_parallel_threads instances of SE_Sampler class
        '''
        event_samplers = []
        for i, (ev, t_ev) in enumerate(zip(self.events[marker:marker+self.n_parallel_threads], self.transformed_events[marker:marker+self.n_parallel_threads])):
            event_samplers.append(SE_Sampler.remote(
                                            t_ev,
                                            self.names[marker+i],
                                            self.burnin_ev,
                                            self.n_draws_ev,
                                            self.step_ev,
                                            ev,
                                            self.alpha0,
                                            self.a_ev,
                                            self.V_ev,
                                            np.min(ev),
                                            np.max(ev),
                                            np.min(t_ev),
                                            np.max(t_ev),
                                            self.m_max,
                                            self.m_min,
                                            self.output_folder,
                                            self.verbose,
                                            self.icn,
                                            transformed = True
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
                       m_max_plot = self.m_max_plot,
                       n_parallel_threads = self.n_parallel_threads,
                       transformed = True
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
    lnB =numba_gammaln(0.5)+ numba_gammaln(0.5*df) - numba_gammaln(0.5 + 0.5*df)
    return -0.5*np.log(df*np.pi) - lnB - ((df+1)*0.5)*np.log1p(t*t/df)
    
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
                       initial_cluster_number = 5.,
                       transformed = False
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
        
        if transformed:
            self.mass_samples = mass_samples
            self.t_max   = t_max
            self.t_min   = t_min
        else:
            self.mass_samples = self.transform(mass_samples)
            self.t_max        = self.transform(self.m_max)
            self.t_min        = self.transform(self.m_min)
            
        self.sigma_max = np.std(self.mass_samples)/2.
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.b  = a*(np.std(self.mass_samples)/4.)**2
        self.a  = a
        self.V  = V
        self.mu = np.mean(self.mass_samples)
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean var N')
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.verbose = verbose
        self.alpha_samples = []
        
    def transform(self, samples):
        '''
        Coordinate change into probit space
        
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
                "b": self.b,
                "a": self.a,
                "V": self.V,
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
            mean = np.array(points_in_cluster).mean()
            var  = np.array(points_in_cluster).var()
            M    = len(points_in_cluster)
            state['suffstats'][cluster_id] = self.SuffStat(mean, var, M)
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster
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
        mean = (ss.mean*(ss.N)+x)/(ss.N+1)
        var  = (ss.N*(ss.var + ss.mean**2) + x**2)/(ss.N+1) - mean**2
        return self.SuffStat(mean, var, ss.N+1)


    def remove_datapoint_from_suffstats(self, x, ss):
        if ss.N == 1:
            return(self.SuffStat(0,0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        var  = (ss.N*(ss.var + ss.mean**2) - x**2)/(ss.N-1) - mean**2
        return self.SuffStat(mean, var, ss.N-1)
    
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
        state['suffstats'][cluster_id] = self.SuffStat(0, 0, 0)
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
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
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
        weights = random.RandomState().dirichlet(alpha).flatten()
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
            s = stats.invgamma(a_n, scale = b_n).rvs()
            m = stats.norm(mu_n, s*V_n).rvs()
            components[i] = {'mean': m, 'sigma': np.sqrt(s), 'weight': weights[i]}
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

    def postprocess(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        
        lower_bound = max([self.m_min-1, self.glob_m_min])
        upper_bound = min([self.m_max+1, self.glob_m_max])
        app  = np.linspace(lower_bound, upper_bound, 1000)
        da   = app[1]-app[0]
        percentiles = [5,16, 50, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(self.initial_samples, bins = int(np.sqrt(len(self.initial_samples))), histtype = 'step', density = True, label = r"\textsc{Mass samples}")
        prob = []
        for ai in app:
            a = self.transform(ai)
            prob.append([logsumexp([log_norm(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) - log_norm(a, 0, 1) for sample in self.mixture_samples])
        
        log_draws_interp = []
        for pr in np.array(prob).T:
            log_draws_interp.append(interp1d(app, pr - logsumexp(pr + np.log(da))))
        
        picklefile = open(self.output_posteriors + '/posterior_functions_{0}.pkl'.format(self.e_ID), 'wb')
        pickle.dump(log_draws_interp, picklefile)
        picklefile.close()
        
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = 1)
        normalisation = logsumexp(p[50] + np.log(da))
        for perc in percentiles:
            p[perc] = p[perc] - normalisation
            
        names = ['m'] + [str(perc) for perc in percentiles]
        np.savetxt(self.output_recprob + '/log_rec_prob_{0}.txt'.format(self.e_ID), np.array([app, p[5], p[16], p[50], p[84], p[95]]).T, header = ' '.join(names))
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        for perc in percentiles:
            p[perc] = p[perc]/np.exp(normalisation)
            
        prob = np.array(prob)
        
        ent = []
        
        for i in range(np.shape(prob)[1]):
            sample = np.exp(prob[:,i])
            ent.append(js(sample,p[50]))
        mean_ent = np.mean(ent)
        np.savetxt(self.output_entropy + '/KLdiv_{0}.txt'.format(self.e_ID), np.array(ent), header = 'mean JS distance = {0}'.format(mean_ent))
        
        picklefile = open(self.output_pickle + '/posterior_functions_{0}.pkl'.format(self.e_ID), 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        self.sample_probs = prob
        self.median_mf = np.array(p[50])
        
        ax.fill_between(app, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'steelblue', label = r"\textsc{Reconstructed}", zorder = 100)
        ax.set_xlabel('$M\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        ax.set_xlim(lower_bound, upper_bound)
        ax.grid(True,dashes=(1,3))
        ax.legend(loc=0,frameon=False,fontsize=10)
        plt.savefig(self.output_pltevents + '/{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
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
                       transformed = False
                       ):
                       
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
        self.m_max_plot = m_max_plot
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
        
        self.p = Pool(n_parallel_threads)
        
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
        cdf = (samples - cdf_bounds[0])/(cdf_bounds[1]-cdf_bounds[0])
        new_samples = np.sqrt(2)*erfinv(2*cdf-1)
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
            return -np.log(self.t_max-self.t_min), -np.log(self.t_max-self.t_min)
        else:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cluster_id]]
        n = len(events)
        events.append(self.posterior_draws[data_id])
        logL_D = state['logL_D'][cluster_id] #denominator
        logL_N = self.log_numerical_predictive(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max) #numerator
        return logL_N - logL_D, logL_N

    def log_numerical_predictive(self, events, t_min, t_max, sigma_min, sigma_max):
        logN_cnst = compute_norm_const(0, 1, events) + np.log(t_max - t_min) + np.log(sigma_max - sigma_min)
        I, dI = dblquad(integrand, t_min, t_max, gfun = sigma_min, hfun = sigma_max, args = [events, t_min, t_max, sigma_min, sigma_max, logN_cnst])
        return np.log(I) + logN_cnst
    
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
        """
        cluster_ids = list(state['ev_in_cl'].keys()) + ['new']
        # can't pickle injected density
        saved_injected_density = self.injected_density
        self.injected_density  = None
#        with Pool(self.n_parallel_threads) as p:
        output = self.p.map(self.compute_score, [[data_id, cid, state] for cid in cluster_ids])
        scores = {out[0]: out[1] for out in output}
        self.numerators = {out[0]: out[2] for out in output}
        self.injected_density = saved_injected_density
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
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
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
            m, s = sample_point(events, self.t_min, self.t_max, self.sigma_min, self.sigma_max, burnin = 1000)
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
        Plots samples [x] for each event in separate plots along with inferred distribution and saves draws.
        """
        
        app  = np.linspace(self.m_min, self.m_max_plot, 1000)
        da = app[1]-app[0]
        percentiles = [50, 5,16, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        fig.suptitle('Observed mass function')
        ax  = fig.add_subplot(111)
        if self.true_masses is not None:
            truths = np.genfromtxt(self.true_masses, names = True)
            ax.hist(truths['m'], bins = int(np.sqrt(len(truths['m']))), histtype = 'step', density = True, label = r"\textsc{True masses}")
        prob = []
        for ai in app:
            a = self.transform(ai)
            prob.append([logsumexp([log_norm(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) - log_norm(a, 0, 1) for sample in self.mixture_samples])
        
        log_draws_interp = []
        for pr in np.array(prob).T:
            log_draws_interp.append(interp1d(app, pr - logsumexp(pr + np.log(da))))
        
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
            p[perc] = np.percentile(prob, perc, axis = 1)
        normalisation = np.sum(np.exp(p[50])*da)
        for perc in percentiles:
            p[perc] = p[perc] - np.log(normalisation)
            
        self.sample_probs = prob
        self.median_mf = np.array(p[50])
        names = ['m'] + [str(perc) for perc in percentiles]
        np.savetxt(self.output_events + '/log_rec_obs_prob_mf.txt', np.array([app, p[50], p[5], p[16], p[84], p[95]]).T, header = ' '.join(names))
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        for perc in percentiles:
            p[perc] = p[perc]/normalisation
        
        ax.fill_between(app, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'steelblue', label = r"\textsc{Reconstructed}", zorder = 100)
        if self.injected_density is not None:
            norm = np.sum([self.injected_density(a)*(app[1]-app[0]) for a in app])
            density = np.array([self.injected_density(a)/norm for a in app])
            ax.plot(app, density, color = 'k', marker = '', linewidth = 0.8, label = r"\textsc{Simulated - Observed}")
        ax.set_xlabel('$M\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        ax.set_xlim(self.m_min*1.1, self.m_max_plot)
        ax.grid(True,dashes=(1,3))
        ax.legend(loc=0,frameon=False,fontsize=10)
        plt.savefig(self.output_events + '/obs_mass_function.pdf', bbox_inches = 'tight')
        ax.set_yscale('log')
        ax.set_ylim(np.min(p[50]))
        plt.savefig(self.output_events + '/log_obs_mass_function.pdf', bbox_inches = 'tight')
        
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
        inj = np.array([self.injected_density(ai)/norm for ai in app])
        ent = js(p[50], inj)
        print('Jensen-Shannon distance: {0} nats'.format(ent))
        np.savetxt(self.output_events + '/relative_entropy.txt', np.array([ent]))
        
    
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
        
        app  = np.linspace(self.m_min*1.1, self.m_max_plot, 1000)
        da = app[1]-app[0]
        prob = []
        for ai in app:
            a = self.transform(ai)
            prob.append([logsumexp([log_norm(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) - log_norm(a, 0, 1) for sample in self.mixture_samples[-self.ncheck:]])

        log_draws_interp = []
        for pr in np.array(prob).T:
            log_draws_interp.append(interp1d(app, pr - logsumexp(pr + np.log(da))))
        
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
        return



