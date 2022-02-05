import numpy as np
import os
import hdpgmm.multidim.collapsed_gibbs as HDPGMM
import optparse as op
import configparser
import sys
import importlib.util
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon as js
from hdpgmm.multidim.preprocessing import load_single_event
import ray
from ray.util import ActorPool
from pathlib import Path
from distutils.spawn import find_executable


if find_executable('latex'):
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

def save_options(options):
    """
    Saves options for the run (for reproducibility)
    
    Arguments:
        :dict options: options
    """
    logfile = open(Path(options.output, 'options_log.txt'), 'w')
    for key, val in zip(vars(options).keys(), vars(options).values()):
        logfile.write('{0}: {1}\n'.format(key,val))
    logfile.close()

def is_opt_provided (parser, dest):
    """
    Checks if an option is provided by the user
    
    Arguments:
        :obj parser: an instance of optparse.OptionParser with the user-provided options
        :str dest:   name of the option
    Returns:
        :bool: True if the option is provided, false otherwise
    """
    for opt in parser._get_all_options():
        try:
            if opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]):
                return True
        except:
            if opt.dest == dest and opt._long_opts[0] in sys.argv[1:]:
                return True
    return False


def main():
    '''
    Runs the analysis
    '''
    parser = op.OptionParser()
    
    # Input/Output
    parser.add_option("-i", "--input", type = "string", dest = "event_file", help = "Input file")
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder")
    parser.add_option("--optfile", type = "string", dest = "optfile", help = "Options file. Passing command line options overrides optfile. It must contains ALL options")
    parser.add_option("--par", type = "string", dest = "par", help = "Parameter from GW posterior", default = 'm1,m2')
    parser.add_option("--assign", type = "string", dest = "assign_file", help = "Initial guess for assignments", default = None)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = 'M_1,M_2')
    
    # Settings
    parser.add_option("--samp_settings", type = "string", dest = "samp_settings", help = "Burnin, number of draws and number of steps between draws", default = '10,100,10')
    parser.add_option("--icn", dest = "initial_cluster_number", type = "float", help = "Initial cluster number", default = 5.)
    parser.add_option("-s", "--seed", dest = "seed", type = "float", default = 1, help = "Fix seed for reproducibility")
    parser.add_option("--n_samps_dsp", dest = "n_samples_dsp", default = -1, help = "Number of samples to analyse (downsampling). Default: all")
    parser.add_option("-r", "--restart", dest = "restart", default = False, action = 'store_true', help = "Restart from checkpoint or last state. Requires the analysis to be run at least once before, otherwise the inital assignment will fall back to the default assignment")
    parser.add_option("--n_grid", dest = "n_gridpoints", type = "string", help = "Grid points for each parameter (single value or array)", default = '20')
    parser.add_option("-v", "--verbose", dest = "verbose", default = True, action = 'store_false', help = "Suppress output")
    
    # Priors
    parser.add_option("--prior", type = "string", dest = "prior", help = "Parameters for NIG prior (a0, V0). See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf sec. 6 for reference", default = '1,1')
    parser.add_option("--xmin", type = "string", dest = "xmin", help = "Low bounds for parameters", default = None)
    parser.add_option("--xmax", type = "string", dest = "xmax", help = "High bounds for parameters", default = None)
    parser.add_option("--alpha", type = "float", dest = "alpha0", help = "Internal (event) initial concentration parameter", default = 1.)
    parser.add_option("--sigma_max", type = "string", dest = "sigma_max", help = "Maximum std for clusters", default = None)
    
    # Others
    parser.add_option("--cosmology", type = "string", dest = "cosmology", help = "Cosmological parameters (h, om, ol). Default values from Planck (2021)", default = '0.674,0.315,0.685')
    
    (options, args) = parser.parse_args()
    
    # Converts relative paths to absolute paths
    options.event_file = Path(str(options.event_file)).resolve()
    options.output     = Path(str(options.output)).resolve()
    if options.assign_file is not None:
        options.assign_file = Path(str(options.assign_file)).resolve()
    
    # If provided, reads optfile. Command-line inputs override file options.
    if options.optfile is not None:
        config = configparser.ConfigParser()
        config.read(options.optfile)
        opts = config['DEFAULT']
        configfile_keys = [k for k in opts.keys()]
        for key, val in zip(vars(options).keys(), vars(options).values()):
            if not is_opt_provided(parser, key) and key in configfile_keys:
                vars(options)[key] = opts[key]

    if options.assign_file == 'None':
        options.assign_file = None
    if options.sigma_max == 'None':
        options.sigma_max = None
    elif not options.sigma_max == None:
        options.sigma_max = np.array([float(x) for x in options.sigma_max.split(',')])
        
    options.seed    = int(options.seed)
    options.restart = int(options.restart)
    
    # Read hyperpriors and sampling settings
    if options.prior is not None:
        options.a, options.V = [float(x) for x in options.prior.split(',')]
    options.burnin, options.n_draws, options.n_steps = (int(x) for x in options.samp_settings.split(','))
    
    # Read cosmology
    options.h, options.om, options.ol = (float(x) for x in options.cosmology.split(','))
    
    # Read par names and symbol
    options.symbol = ['${0}$'.format(x) for x in options.symbol.split(',')]
    options.par    = [x for x in options.par.split(',')]
    
    # Loads event
    event, name = load_single_event(event = options.event_file, seed = options.seed, par = options.par, n_samples = int(options.n_samples_dsp), h = options.h, om = options.om, ol = options.ol)
    
    dim = len(event[-1])
    
    # Check min and max
    if options.xmin is None or 'None':
        options.xmin = np.min(event, axis = 0)
    else:
        opt_min = np.array([float(x) for x in options.xmin.split(',')])
        options.xmin = np.minimum([opt_min, np.min(event, axis = 0)])
        
    if options.xmax is None or 'None':
        options.xmax = np.max(event, axis = 0)
    else:
        opt_max = np.array([float(x) for x in options.xmax.split(',')])
        options.xmax = np.maximum([opt_max, np.max(event, axis = 0)])

    # Loads initial assignment, if provided
    if options.assign_file is not None:
        assign = np.genfromtxt(options.assign_file)
        options.initial_cluster_number = int(np.max(assign) + 1)
    else:
        assign = None
    
    # Create a RandomState
    if options.seed == 0:
        rdstate = np.random.RandomState(seed = 1)
    else:
        rdstate = np.random.RandomState()
        
    # Read grid points
    options.n_gridpoints = [int(x) for x in options.n_gridpoints.split(',')]
    if len(options.n_gridpoints) == (1 or dim):
        options.n_gridpoints = options.n_gridpoints*np.ones(dim, dtype = int)
    else:
        print('n_gridpoints is not scalar but its dimensions does not match the data point dimensions.')
        exit()
    
    save_options(options)
    
    try:
        ray.init(num_cpus = 1)
    except:
        ray.init(num_cpus = 1, object_store_memory=10**9)
    
    sampler = HDPGMM.SE_Sampler.remote(
                                burnin = int(options.burnin),
                                n_draws = int(options.n_draws),
                                n_steps = int(options.n_steps),
                                dim = dim,
                                alpha0 = float(options.alpha0),
                                a = float(options.a),
                                V = float(options.V),
                                output_folder = options.output,
                                verbose = bool(options.verbose),
                                initial_cluster_number = int(options.initial_cluster_number),
                                transformed = False,
                                var_names = options.symbol,
                                sigma_max = options.sigma_max,
                                rdstate = rdstate,
                                restart = options.restart,
                                n_gridpoints = options.n_gridpoints,
                                )
    pool = ActorPool([sampler])
    bin = []
    for s in pool.map(lambda a, v: a.run.remote(v), [[event, name, (options.xmin, options.xmax), None, assign]]):
        bin.append(s)

if __name__=='__main__':
    main()

