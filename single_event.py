import numpy as np
import os
import hdpgmm.collapsed_gibbs as HDPGMM
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
from hdpgmm.preprocessing import load_single_event

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
    logfile = open(options.output + '/options_log.txt', 'w')
    for key, val in zip(vars(options).keys(), vars(options).values()):
        logfile.write('{0}: {1}\n'.format(key,val))
    logfile.close()

def main():
    '''
    Runs the analysis
    '''
    parser = op.OptionParser()
    
    # Input/Output
    parser.add_option("-i", "--input", type = "string", dest = "event_file", help = "Input file")
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder")
    parser.add_option("--optfile", type = "string", dest = "optfile", help = "Options file. Passing command line options overrides optfile. It must contains ALL options")
    parser.add_option("--inj", type = "string", dest = "inj_file", help = "File with injected single event posterior (two columns file: m p)", default = None)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = 'M')
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes. Use '' for dimensionless quantities", default = 'M_{\\odot}')
    
    # Settings
    parser.add_option("--samp_settings", type = "string", dest = "samp_settings_ev", help = "Burnin, samples and step for single event sampling", default = 10,1000,1)
    parser.add_option("--icn", dest = "initial_cluster_number", type = "float", help = "Initial cluster number", default = 5.)
    parser.add_option("-d", "--diagnostic", dest = "diagnostic", action = 'store_true', default = False, help = "Run diagnostic routines (Autocorrelation, quasi-convergence)")
    parser.add_option("-s", "--seed", dest = "seed", action = 'store_true', default = False, help = "Fix seed for reproducibility")
    parser.add_option("--n_samps_dsp", dest = "n_samples_dsp", default = -1, help = "Number of samples to analyse (downsampling). Default: all")
    
    # Priors
    parser.add_option("--prior", type = "string", dest = "prior", help = "Parameters for NIG prior (a0, V0). See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf sec. 6 for reference", default = '1,1')
    parser.add_option("--mmin", type = "float", dest = "mmin", help = "Minimum BH mass [Msun]", default = 3.)
    parser.add_option("--mmax", type = "float", dest = "mmax", help = "Maximum BH mass [Msun]", default = 120.)
    parser.add_option("--alpha", type = "float", dest = "alpha0", help = "Internal (event) initial concentration parameter", default = 1.)
    parser.add_option("--gamma", type = "float", dest = "gamma0", help = "External (MF) initial concentration parameter", default = 1.)

    (options, args) = parser.parse_args()
    
    # Converts relative paths to absolute paths
    options.event_file = os.path.abspath(options.event_file)
    options.output     = os.path.abspath(options.output)
    if options.inj_file is not None:
        options.inj_file = os.path.abspath(options.inj_file)
    
    # If provided, reads optfile. Command-line inputs override file options.
    if options.optfile is not None:
        config = configparser.ConfigParser()
        config.read(options.optfile)
        opts = config['DEFAULT']
        for key, val in zip(vars(options).keys(), vars(options).values()):
            if not is_opt_provided(parser, key):
                vars(options)[key] = opts[key]
    
    # Reads hyperpriors and sampling settings
    if options.prior is not None:
        options.a, options.V = [float(x) for x in options.prior.split(',')]
    options.burnin, options.n_draws, options.step = (int(x) for x in options.samp_settings.split(','))

    # Loads event
    event, name = load_single_event(event = options.event_file, seed = bool(options.seed), par = options.par, n_samples = int(options.n_samples_dsp))
    
    # Loads posterior injections and saves them as interpolants
    if options.inj_file is not None:
        post = np.genfromtxt(options.inj_file + '/' + name + '.txt', names = True)
        inj_post = interp1d(post['m'], post['p'], bounds_error = False, fill_value = 0)
    else:
        inj_post = None
    
    save_options(options)
    
    sampler = HDPGMM.SE_sampler(mass_samples = event,
                                event_id     = name,
                                burnin = int(options.burnin),
                                n_draws = int(options.n_draws),
                                step = int(options.step),
                                alpha0 = float(options.alpha0),
                                a = float(options.a),
                                V = float(options.V),
                                m_min = float(options.mmin),
                                m_max = float(options.mmax),
                                output_folder = options.output,
                                verbose = True,
                                diagnostic = bool(options.diagnostic),
                                initial_cluster_number = int(options.initial_cluster_number),
                                transformed = False,
                                inj_post = inj_post,
                                seed = bool(options.seed),
                                var_symbol = options.symbol,
                                unit = options.unit,
                                )
    sampler.run()

if __name__=='__main__':
    main()
