import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletDistribution, DirichletProcess
import pickle
import cpnest
import corner
from scipy.special import erf, logsumexp
from scipy.stats import norm, dirichlet
from scipy.interpolate import interp1d
import os


def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

def PL(m, alpha, m_max, m_min, l_max, l_min):
    f = m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.
    return f

def gauss(x, x0, s):
    return np.exp(-((x-x0)**2/(2*s**2)))/(np.sqrt(2*np.pi)*s)

def bimodal(x,x0,s0,x1,s1):
    return (np.exp(-((x-x0)**2/(2*s0**2)))/(np.sqrt(2*np.pi)*s0) + np.exp(-((x-x1)**2/(2*s1**2)))/(np.sqrt(2*np.pi)*s1))/2.

def four_g(x,x0,x1,x2,x3,s0,s1,s2,s3,b0,b1,b2,b3):
    return (b0*np.exp(-((x-x0)**2/(2*s0**2)))/(np.sqrt(2*np.pi)*s0) + b1*np.exp(-((x-x1)**2/(2*s1**2)))/(np.sqrt(2*np.pi)*s1)+ b2*np.exp(-((x-x2)**2/(2*s2**2)))/(np.sqrt(2*np.pi)*s2)+ b3*np.exp(-((x-x3)**2/(2*s3**2)))/(np.sqrt(2*np.pi)*s3))/(4*(b1+b2+b3+b0))

def logPrior(*args):
    return 0
 
#
alpha = 1.1
mmax = 75
mmin = 15
lmax = 5
lmin = 5
mu = 50
sigma = 4
a = 20
#true_vals = [alpha, mmax, mmin, lmax, lmin, a, 1]
#true_vals = [mu, sigma, a, 1]
#true_vals = [50, 2, 100, 2.5, a, 1]
true_vals = [25,4,55,5, a, 1]
#true_vals = [38, 54, 45,60,6,4,5,7,0.4,0.1,0.2,0.3]
#true_vals = [60,54,45,38,7,4,5,6,0.3,0.1,0.2,0.4, a,1]

samp_file = '/Users/stefanorinaldi/Documents/mass_inference/bimodal_good/mass_function/all_samples.pkl'
#samp_file = '/Users/stefanorinaldi/Documents/mass_inference/DPGMM/reconstructed_events/posteriors/posterior_functions_gaussian.pkl'
openfile  = open(samp_file, 'rb')
samples   = pickle.load(openfile)
openfile.close()


rec_file = '/Users/stefanorinaldi/Documents/mass_inference/bimodal_good/mass_function/log_joint_obs_prob_mf.txt'
#rec_file = '/Users/stefanorinaldi/Documents/mass_inference/DPGMM/reconstructed_events/rec_prob/log_rec_prob_gaussian.txt'
rec = np.genfromtxt(rec_file, names = True)

x = np.linspace(20, 40,400)
dx = x[1]-x[0]

##
#ps = PL(x, alpha, mmax, mmin, lmax, lmin)
#
#samples  = dirichlet(a*ps*dx/np.sum(ps*dx)).rvs(size = 300)
#
out_folder  = '/Users/stefanorinaldi/Documents/parametric/DP'
#
#samples = np.array([s for s in samples if (s != 0.).all()])
#
#fig = plt.figure()
#ax  = fig.add_subplot(111)
#for s in samples:
#    ax.plot(x, s, linewidth = 0.3)
#
#ax.plot(x,ps*dx/np.sum(ps*dx))
#
#ax.set_xlabel('x')
#ax.set_ylabel('p(x)dx')
#fig.suptitle('Conc. par. = {0}'.format(a))
#plt.savefig(out_folder+'/draws.pdf', bbox_inches = 'tight')

#probs = []
#xmax = []
#for samp in samples:
#    p = np.ones(400) * -np.inf
#    for component in samp.values():
#        logW = np.log(component['weight'])
#        mu   = component['mean']
#        s    = component['sigma']
#        for i, mi in enumerate(x):
#            p[i] = log_add(p[i], logW + log_norm(mi, mu, s))
#    p = np.exp(p + np.log(dx) - logsumexp(p+np.log(dx)))
#    probs.append(p)
##    xmax.append(x[np.where(p == p.max())])
#
##print(np.mean(xmax))
#for p in probs:
#    plt.plot(x,p, lw = 0.3)
#
#plt.plot(x, gauss(x, 30.075, 5.945)*dx/np.sum(gauss(x, 30.075, 5.945)*dx), c = 'r')
#plt.show()

#names  = ['alpha', 'm_max', 'm_min', 'l_max', 'l_min']
#bounds = [[0,2], [60,80], [10,30], [3,10], [3,10]]
#labels  = ['\\alpha', 'm_{max}', 'm_{min}', '\\lambda_{max}', '\\lambda_{min}']
#
#names = ['mu_1', 'sigma_1', 'mu_2', 'sigma_2']
#bounds = [[40,60], [1,4], [90, 110], [1,4]]
#labels = ['\mu_1', '\sigma_1', '\mu_2', '\sigma_2']
names = ['mu1', 'sigma1', 'mu2','sigma2']
bounds = [[20,50], [2,7],[40,70],[2,7]]
labels = ['\mu_1', '\sigma_1','\mu_2','\sigma_2']
selected_model = bimodal

#names  = ['mu1','mu2','mu3', 'mu4', 's1', 's2', 's3', 's4', 'b1', 'b2','b3','b4']
#bounds = [[58,70], [50,58], [40,50],[30,40], [1,8], [1,8], [1,8],[1,8], [0,1],[0,1],[0,1],[0,1]]
#labels = ['\mu_1','\mu_2','\mu_3', '\mu_4', '\sigma_1', '\sigma_2', '\sigma_3', '\sigma_4', 'w_1','w_2','w_3','w_4']


PE = DirichletProcess(
    selected_model,
    names,
    bounds,
    samples,
    15,
    70,
    prior_pars = logPrior,
    max_a = 1e5,
    max_N = 100
    )

work = cpnest.CPNest(PE,
                    verbose = 2,
                    nlive = 1000,
                    maxmcmc = 1000,
                    nthreads = 4,
                    output  = out_folder
                    )
work.run()
print('log Evidence: {0}'.format(work.NS.logZ))

labels = labels + ['\\alpha', 'N']
par_names = names
names = names + ['a','N']
#x = np.genfromtxt('/Users/stefanorinaldi/Documents/parametric/DP/nested_samples.dat', names = True).ravel()
x = work.posterior_samples.ravel()
#x['b1'] = x['b1']/(x['b1']+x['b2']+x['b3']+x['b4'])
#x['b2'] = x['b2']/(x['b1']+x['b2']+x['b3']+x['b4'])
#x['b3'] = x['b3']/(x['b1']+x['b2']+x['b3']+x['b4'])
#x['b4'] = x['b4']/(x['b1']+x['b2']+x['b3']+x['b4'])
samps = np.column_stack([x[lab] for lab in names])
fig = corner.corner(samps,
       labels= [r'${0}$'.format(lab) for lab in labels],
       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
       use_math_text=True, truths=true_vals,
       filename=os.path.join(out_folder,'joint_posterior.pdf'))
fig.savefig(os.path.join(out_folder,'joint_posterior.pdf'), bbox_inches='tight')

p = np.percentile(np.column_stack([x[lab] for lab in par_names]), 50, axis = 0)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.fill_between(rec['m'], np.exp(rec['95']), np.exp(rec['5']), color = 'lightgreen', alpha = 0.5)
ax.fill_between(rec['m'], np.exp(rec['84']), np.exp(rec['16']), color = 'aqua', alpha = 0.5)
ax.plot(rec['m'], np.exp(rec['50']), color = 'r')
print(*p)
ax.plot(rec['m'], selected_model(rec['m'], *p), color = 'k')
fig.savefig(os.path.join(out_folder,'compare_50.pdf'), bbox_inches='tight')

