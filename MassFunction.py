import numpy as np
import os
import collapsed_gibbs as DPGMM
import optparse as op
import configparser
import sys
import importlib.util
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon as js


def is_opt_provided (parser, dest):
    for opt in parser._get_all_options():
        try:
            if opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]):
                return True
        except:
            if opt.dest == dest and opt._long_opts[0] in sys.argv[1:]:
                return True
    return False

def log_normal_density(x, x0, sigma):
    return (-(x-x0)**2/(2*sigma**2))-np.log(np.sqrt(2*np.pi)*sigma)

def plot_samples(samples, m_min, m_max, output, injected_density = None, filtered_density = None, true_masses = None):

        app  = np.linspace(m_min, m_max, 1000)
        da = app[1]-app[0]
        percentiles = [50, 5,16, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        fig.suptitle('Observed mass function')
        ax  = fig.add_subplot(111)
        if true_masses is not None:
            truths = np.genfromtxt(true_masses, names = True)
            ax.hist(truths['m'], bins = int(np.sqrt(len(truths['m']))), histtype = 'step', density = True, label = '$Masses$')
        prob = []
        for a in app:
            prob.append([sample(a) for sample in samples])
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = 1)
        sample_probs = prob
        median_mf = np.array(p[50])
        norm = np.sum(np.exp(p[50]))*da
        log_norm = np.log(norm)
        names = ['m'] + [str(perc) for perc in percentiles]
        np.savetxt(output+ '/log_joint_obs_prob_mf.txt', np.array([app, p[50] - log_norm, p[5] - log_norm, p[16] - log_norm, p[84] - log_norm, p[95] - log_norm]).T, header = ' '.join(names))
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        
        ax.fill_between(app, p[95]/norm, p[5]/norm, color = 'lightgreen', alpha = 0.5, label = '$90\%\ CI$')
        ax.fill_between(app, p[84]/norm, p[16]/norm, color = 'aqua', alpha = 0.5, label = '$68\%\ CI$')
        ax.plot(app, p[50]/norm, marker = '', color = 'r', label = '$Reconstructed$')
        if injected_density is not None:
            norm = np.sum([injected_density(a)*(app[1]-app[0]) for a in app])
            density = np.array([injected_density(a)/norm for a in app])
            ax.plot(app, density, color = 'm', marker = '', linewidth = 0.7, label = '$Simulated - Astro$')
            ent = [js(np.exp(s(app)), density) for s in samples]
            JSD = {}
            for perc in percentiles:
                JSD[perc] = np.percentile(ent, perc, axis = 0)
            print('Jensen-Shannon distance: {0}+{1}-{2} nats'.format(JSD[50], JSD[95]-JSD[50], JSD[50]-JSD[5]))
            np.savetxt(output + '/joint_relative_entropy.txt', np.array([JSD[50], JSD[5], JSD[16], JSD[84], JSD[95]]), header = '50 5 16 84 95')
        if filtered_density is not None:
            norm = np.sum([filtered_density(a)*(app[1]-app[0]) for a in app])
            f_density = np.array([filtered_density(a)/norm for a in app])
            ax.plot(app, f_density, color = 'k', marker = '', linewidth = 0.7, label = '$Simulated - Obs$')
            ent = np.array([js(np.exp(s(app)), f_density) for s in samples])
            JSD = {}
            for perc in percentiles:
                JSD[perc] = np.percentile(ent, perc)
            print('Jensen-Shannon distance: {0}+{1}-{2} nats (filtered)'.format(JSD[50], JSD[95]-JSD[50], JSD[50]-JSD[5]))
            np.savetxt(output + '/filtered_joint_relative_entropy.txt', np.array([JSD[50], JSD[5], JSD[16], JSD[84], JSD[95]]), header = '50 5 16 84 95')
        
        plt.legend(loc = 0)
        ax.set_xlabel('$M\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(output + '/joint_mass_function.pdf', bbox_inches = 'tight')
        ax.set_yscale('log')
        ax.set_ylim(np.min(p[50]))
        plt.savefig(output + '/log_joint_mass_function.pdf', bbox_inches = 'tight')

def main():
    '''
    Runs the analysis
    '''
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type = "string", dest = "events_path", help = "Input folder")
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder")
    parser.add_option("--mmin", type = "float", dest = "mmin", help = "Minimum BH mass [Msun]", default = 3.)
    parser.add_option("--mmax", type = "float", dest = "mmax", help = "Maximum BH mass [Msun]", default = 120.)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density")
    parser.add_option("--true_masses", type = "string", dest = "true_masses", help = "Simulated true masses")
    parser.add_option("--optfile", type = "string", dest = "optfile", help = "Options file. Passing command line options overrides optfile. It must contains ALL options")
    parser.add_option("--samp_settings", type = "string", dest = "samp_settings", help = "Burnin, samples and step for MF sampling", default = '10,1000,1')
    parser.add_option("--samp_settings_ev", type = "string", dest = "samp_settings_ev", help = "Burnin, samples and step for single event sampling. If None, uses MF settings")
    parser.add_option("--hyperpriors_ev", type = "string", dest = "hyperpriors_ev", help = "Event hyperpriors (a0, V0). See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf sec. 6 for reference", default = '1,1')
    parser.add_option("--alpha", type = "float", dest = "alpha0", help = "Internal (event) concentration parameter", default = 1.)
    parser.add_option("--gamma", type = "float", dest = "gamma0", help = "External (MF) concentration parameter", default = 1.)
    parser.add_option("-e", "--processed_events", dest = "process_events", action = 'store_false', default = True, help = "Disables event processing")
    parser.add_option("--icn", dest = "initial_cluster_number", type = "float", help = "Initial cluster number", default = 5.)
    parser.add_option("--nthreads", dest = "n_parallel_threads", type = "int", help = "Number of parallel threads to spawn", default = 8)
    parser.add_option("-v", "--verbose", dest = "verbose", action = 'store_true', default = False, help = "Display output")
    parser.add_option("-p", "--postprocessing", dest = "postprocessing", action = 'store_true', default = False, help = "Postprocessing - requires log_rec_prob_mf.txt")
    parser.add_option("--selfunc", dest = "selection_function", help = "Python module with selection function or text file with M_i and S(M_i) for interp1d")
    parser.add_option("--join", dest = "join", help = "Join samples from different runs", action = 'store_true', default = False)
    (options, args) = parser.parse_args()
    
    if options.optfile is not None:
        config = configparser.ConfigParser()
        config.read(options.optfile)
        opts = config['DEFAULT']
        for key, val in zip(vars(options).keys(), vars(options).values()):
            if not is_opt_provided(parser, key):
                vars(options)[key] = opts[key]
        if options.true_masses == 'None':
            options.true_masses = None
        if options.inj_density_file == 'None':
            options.inj_density_file = None
        if options.selection_function == 'None':
            options.selection_function = None
            
    if options.hyperpriors_ev is not None:
        options.hyperpriors_ev = [float(x) for x in options.hyperpriors_ev.split(',')]
    options.samp_settings = [int(x) for x in options.samp_settings.split(',')]
    if options.samp_settings_ev is not None:
        options.samp_settings_ev = [int(x) for x in options.samp_settings_ev.split(',')]
    
    event_files = [options.events_path+f for f in os.listdir(options.events_path) if not f.startswith('.')]
    events      = []
    names       = []
    
    for event in event_files:
        events.append(np.genfromtxt(event))
        names.append(event.split('/')[-1].split('.')[0])
    inj_density = None
    if options.inj_density_file is not None:
        inj_file_name = options.inj_density_file.split('/')[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(inj_file_name, options.inj_density_file)
        inj_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inj_module)
        inj_density = inj_module.injected_density
    
    if options.selection_function is not None:
        if options.selection_function.endswith('.py'):
            sel_func_name = options.selection_function.split('/')[-1].split('.')[0]
            spec = importlib.util.spec_from_file_location(sel_func_name, options.selection_function)
            sf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sf_module)
            sel_func = sf_module.selection_function
        else:
            sf = np.genfromtxt(options.selection_function)
            sel_func = interp1d(sf[:,0], sf[:,1], bounds_error = False, fill_value = (sf[:,1][0],sf[:,1][-1]))
    
    if options.selection_function is not None:
        def filtered_density(x):
            return sel_func(x)*inj_density(x)
    else:
        filtered_density = inj_density
        
    if not bool(options.postprocessing):
        sampler = DPGMM.CGSampler(events = events,
                              samp_settings = options.samp_settings,
                              samp_settings_ev = options.samp_settings_ev,
                              alpha0 = float(options.alpha0),
                              gamma0 = float(options.gamma0),
                              hyperpriors_ev = options.hyperpriors_ev,
                              m_min = float(options.mmin),
                              m_max = float(options.mmax),
                              verbose = bool(options.verbose),
                              output_folder = options.output,
                              initial_cluster_number = int(options.initial_cluster_number),
                              process_events = bool(options.process_events),
                              n_parallel_threads = int(options.n_parallel_threads),
                              injected_density = filtered_density,
                              true_masses = options.true_masses,
                              names = names,
                              )
        sampler.run()
    
    if bool(options.join):
        samples = []
        pickle_folder = options.output + '/mass_function/'
        pickle_files  = [pickle_folder + f for f in os.listdir(pickle_folder) if (f.startswith('posterior_functions_') or f.startswith('checkpoint'))]
        
        for file in pickle_files:
            openfile = open(file, 'rb')
            for d in pickle.load(openfile):
                samples.append(d)
            openfile.close()
        samples_set = []
        for s in samples:
            if not s in samples_set:
                samples_set.append(s)
        
        picklefile = open(pickle_folder + '/all_samples.pkl', 'wb')
        pickle.dump(samples_set, picklefile)
        picklefile.close()
        
        print('{0} MF samples'.format(len(samples_set)))
        if options.selection_function is not None:
            plot_samples(samples = samples_set, m_min = float(options.mmin), m_max = float(options.mmax), output = pickle_folder, injected_density = inj_density, filtered_density = filtered_density, true_masses = options.true_masses)
        else:
            plot_samples(samples = samples_set, m_min = float(options.mmin), m_max = float(options.mmax), output = pickle_folder, filtered_density = inj_density, true_masses = options.true_masses)
        
        
        
    if options.selection_function is None:
        exit()
    
    app = np.linspace(options.mmin, options.mmax, 1000)
    try:
        obs_mf = np.genfromtxt(options.output + '/mass_function/log_joint_obs_prob_mf.txt', names = True)
    except:
        obs_mf = np.genfromtxt(options.output + '/mass_function/log_rec_obs_prob_mf.txt', names = True)
    percentiles = [50, 5, 16, 84, 95]
    dm = obs_mf['m'][1]-obs_mf['m'][0]
    mf = {}
    for p in percentiles:
        mf[p] = np.array([omf - np.log(sel_func(m)) for m, omf in zip(obs_mf['m'], obs_mf[str(p)])])
    
    norm = np.exp(mf[50]).sum()*dm
    names = ['m']+[str(perc) for perc in percentiles]
    np.savetxt(options.output + '/mass_function/log_rec_prob_mf.txt',  np.array([app, mf[50], mf[5], mf[16], mf[84], mf[95]]).T, header = ' '.join(names))

    fig = plt.figure()
    fig.suptitle('Mass function')
    ax  = fig.add_subplot(111)
    ax.fill_between(obs_mf['m'], np.exp(mf[95])/norm, np.exp(mf[5])/norm, color = 'lightgreen', alpha = 0.5, label = '$90\%\ CI$')
    ax.fill_between(obs_mf['m'], np.exp(mf[84])/norm, np.exp(mf[16])/norm, color = 'aqua', alpha = 0.5, label = '$68\%\ CI$')
    ax.plot(obs_mf['m'], np.exp(mf[50])/norm, marker = '', color = 'r', label = 'Reconstructed (with sel. effects)')
    
    if inj_density is not None:
        norm_density = np.sum([inj_density(ai)*dm for ai in obs_mf['m']])
        ax.plot(obs_mf['m'], [inj_density(a)/norm_density for a in obs_mf['m']], marker = '', color = 'm', linewidth = 0.7, label = 'Simulated - Astrophysical')
    ax.set_ylim(np.min(np.exp(mf[50])))
    ax.set_xlabel('$M\ [M_\\odot]$')
    ax.set_ylabel('$p(M)$')
    ax.legend(loc = 0)
    plt.savefig(options.output + '/mass_function/mass_function.pdf', bbox_inches = 'tight')
    ax.set_yscale('log')
    ax.set_ylim([1e-3,10])
    plt.savefig(options.output + '/mass_function/log_mass_function.pdf', bbox_inches = 'tight')
    
        
    
if __name__=='__main__':
    main()
