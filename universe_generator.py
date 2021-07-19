import numpy as np
from numpy.random import uniform, triangular, normal, gumbel
from scipy.special import erf
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

#def mass_function(m, alpha, m_max, m_min,scale_max=5, scale_min=5):
#    return m**(-alpha)*(-alpha+1)/(m_max**(-alpha+1) - m_min**(-alpha+1))*(1-np.exp(-(m-m_max)/scale_max))*(1-np.exp(-(m_min-m)/scale_min))
#def mass_function(m, alpha=1.2, m_max=70, m_min= 20, l_max = 5, l_min = 5):
#    return m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.

def mass_function(m, a = 2.63, mmin = 4.59, mmax = 86.22, l = 0, mu = 33.07, s = 5.69, d = 4.82):
    if m < mmin or m > mmax:
        return 0
    f = (((1-a)/(mmax**(1-a) - mmin**(1-a)))*m**(-a))# + l*np.exp(-(m-mu)**2/(2*s**2))/(s*np.sqrt(2*np.pi))
    if m > mmin + d:
        return f
    S = np.exp(d/(m-mmin) + d/(m-mmin -d))
    return f/(S+1)

#def mass_function(m, x01 = 25, sigma1 = 4, x02 = 55, sigma2 = 5):
#    return (0.5*np.exp(-(m-x01)**2/(2*sigma1**2))/(np.sqrt(2*np.pi)*sigma1) + 0.5*np.exp(-(m-x02)**2/(2*sigma2**2))/(np.sqrt(2*np.pi)*sigma2))/2.


def posterior_probability(m, m_true, k, b):
    norm = 2*b*np.sqrt(b/k)
    return (-k*(m-m_true)**2+b)/norm

def mass_sampler(m_max, m_min, sup):
    while 1:
        mass_try = uniform(m_min, m_max)
        if mass_function(mass_try) > uniform(0, sup):
            return mass_try

def sigma_sampler(s_min, s_max):
    return np.exp(uniform(np.log(s_min), np.log(s_max)))
def posterior_sampler(m_true, sigma):
    #return triangular(m_true*0.8, m_true, m_true*1.1)
    return normal(m_true, sigma)
    
if __name__ == '__main__':

    out_folder = '/Users/stefanorinaldi/Documents/mass_inference/uds/uds_10/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    post_folder = out_folder+'/events/'
    plot_folder = out_folder+'/plots/'

    if not os.path.exists(out_folder+'/events/'):
        os.mkdir(post_folder)

    data_sf = np.genfromtxt('/Users/stefanorinaldi/Documents/mass_inference/GWTC/seleff.txt', names = True)
    selfunc = interp1d(data_sf['m1'], data_sf['pdet'])

    n_bbh = 45
    n_samples = 200

    alpha = 1.1
    m_min = data_sf['m1'].min()
    m_max = data_sf['m1'].max()
    s_min = 2
    s_max = 4
    
    app = np.linspace(m_min, m_max, 1000)
    mf  = np.array([mass_function(ai) for ai in app])
    sup = mf.max()
    norm = np.sum(mf*(app[1]-app[0]))

    bbhs = []
    masses = []
    sigmas = []
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    i = 0
    j = 0
#    for i in range(n_bbh):
    while i < n_bbh:
        j += 1
        m1 = mass_sampler(m_max, m_min, sup)
        m2 = mass_sampler(m_max, m_min, sup)
        m1 = np.max([m1,m2])
        if selfunc(m1) > uniform():
            sigma = np.exp(uniform(np.log(5), np.log(8)))
            samples = [s for s in normal(loc = m1, scale = sigma, size = n_samples) if s > 0]
            np.savetxt(post_folder + '/event_{0}.txt'.format(i+1), np.array(samples))
            bbhs.append(samples)
            masses.append(m1)
            sigmas.append(sigma)
            i += 1
            
    print(j)


    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    appM = np.linspace(m_min, m_max, 1000)
    mf  = np.array([mass_function(ai) for ai in appM])
    norm = np.sum(mf*(appM[1]-appM[0]))
    ax1.hist(masses, bins = int(np.sqrt(len(masses))), density = True)
    ax1.plot(appM, mf*selfunc(appM)/np.sum(mf*selfunc(appM)*(appM[1]-appM[0])), color = 'r', label = 'Observed')
    ax1.plot(appM, mf/(np.sum(mf)*(appM[1]-appM[0])), color = 'k', label = 'Astrophysical')
#    ax1.plot(appM, selfunc(appM))
    ax1.set_xlabel('$M_1\ [M_\\odot]$')
    ax1.legend(loc = 0)
#    ax1.set_ylim(0, sup*(1.1))
    plt.tight_layout()
    plt.savefig(out_folder+'/truths.pdf', bbox_inches = 'tight')
    np.savetxt(out_folder+'/truths.txt', np.array([masses, sigmas]).T, header = 'm sigma')

    flattened_m = np.array([m for ev in bbhs for m in ev])

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(flattened_m, bins = 1000, density = True)
    ax.plot(appM, mf/norm, color = 'r', linewidth = 0.5)
    ax.set_xlabel('$M_1\ [M_\\odot]$')
    fig.savefig(out_folder+'/all_samples.pdf', bbox_inches = 'tight')
