from __future__ import division
cimport numpy as np
import numpy as np
from libc.math cimport log, sqrt, M_PI, exp, HUGE_VAL, atan2, acos, sin, cos
cimport cython
from numpy.linalg import det, inv
from scipy.stats import multivariate_normal as mn


cdef double LOGSQRT2 = log(sqrt(2*M_PI))

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double _scalar_log_norm(double x, double x0, double s) nogil: return -(x-x0)*(x-x0)/(2*s*s) - LOGSQRT2 - 0.5*log(s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_norm(np.ndarray x, np.ndarray x0, np.ndarray sigma):
    try:
        logP = mn(mean = x0, cov = sigma).logpdf(x)
    except:
        logP = -np.inf
    return logP
    #cdef np.ndarray diff = x-x0
    #cdef double D = det(sigma)
    #return -np.dot(diff.T, np.dot(inv(sigma), diff)) -n*0.5*LOGSQRT2 -0.5*log(D)


#solo un alias per scipy.stats.multivariate_normal
def log_norm(np.ndarray x, np.ndarray x0, np.ndarray sigma):
    return mn(mean = x0, cov = sigma).logpdf(x)
    #return _log_norm(x, x0, sigma, n)

def scalar_log_norm(double x, double x0, double s):
    return _scalar_log_norm(x,x0,s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_prob_component(np.ndarray mu, np.ndarray mean, np.ndarray sigma, double w):
    return log(w) + _log_norm(mu, mean, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_prob_mixture(np.ndarray mu, np.ndarray sigma, dict ev):
    cdef double logP = -HUGE_VAL
    cdef dict component
    for component in ev.values():
        logP = log_add(logP,_log_prob_component(mu, component['mean'], sigma, component['weight']))
    return logP

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrand(double[::1] values, list events, double logN_cnst, unsigned int dim):
    cdef unsigned int i,j
    cdef double logprob = 0.0
    cdef dict ev
    cdef double[::1] mu = values[:dim]
    cdef double[::1] sigma = values[dim:2*dim+1]#_make_sym_matrix(dim, values[dim:])
    cdef double[::1] rho = values[2*dim+1:]
    cdef np.ndarray[double,ndim=2,mode='c'] norm_cov = np.identity(dim)*0.5
    cdef double[:,:] norm_cov_view = norm_cov
    cdef np.ndarray[double,ndim=2,mode='c'] cov
    for i in range(dim):
        for j in range(dim):
            if i < j:
                norm_cov_view[i,j] = rho[dim*i + (j-i)]*sigma[i]*sigma[j]
    cov = norm_cov + norm_cov.T
    for ev in events:
        logprob += _log_prob_mixture(mu, cov, ev)
    return logprob - logN_cnst

def integrand(double[::1] values, list events, double logN_cnst, unsigned int dim):
    return _integrand(values, events, logN_cnst, dim)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _compute_norm_const(np.ndarray mu, np.ndarray sigma, list events):
    cdef double logprob = 0.0
    cdef dict ev
    for ev in events:
        logprob += _log_prob_mixture(mu, sigma, ev)
    return logprob

def compute_norm_const(np.ndarray mu, np.ndarray sigma, list events):
    return _compute_norm_const(mu, sigma, events)

#-------------#
# Coordinates #
#-------------#

cpdef  tuple eq2ang(double ra, double dec):
    """
    convert equatorial ra,dec in radians to angular theta, phi in radians
    parameters
    ----------
    ra: scalar or array
        Right ascension in radians
    dec: scalar or array
        Declination in radians
    returns
    -------
    theta,phi: tuple
        theta = pi/2-dec*D2R # in [0,pi]
        phi   = ra*D2R       # in [0,2*pi]
    """
    cdef double phi = ra
    cdef double theta = np.pi/2. - dec
    return theta, phi

cpdef  tuple ang2eq(double theta, double phi):
    """
    convert angular theta, phi in radians to equatorial ra,dec in radians
    ra = phi*R2D            # [0,360]
    dec = (pi/2-theta)*R2D  # [-90,90]
    parameters
    ----------
    theta: scalar or array
        angular theta in radians
    phi: scalar or array
        angular phi in radians
    returns
    -------
    ra,dec: tuple
        ra  = phi*R2D          # in [0,360]
        dec = (pi/2-theta)*R2D # in [-90,90]
    """
    
    cdef double ra = phi
    cdef double dec = np.pi/2. - theta
    return ra, dec

cpdef list cartesian_to_spherical(np.ndarray[np.float64_t, ndim=1] vector):

    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    cdef unsigned int i
    cdef double r
    r = vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]
    r = sqrt(r)
    # Unit vector
    cdef np.ndarray[np.float64_t, ndim=1] unit = vector / r
    # The polar angle.
    cdef double theta = acos(unit[2])

    # The azimuth.
    cdef double phi = atan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    cdef list coord = [r,theta,phi]
    return coord


cpdef  np.ndarray[np.float64_t, ndim=1] spherical_to_cartesian(np.ndarray[np.float64_t, ndim=1] spherical_vect):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """
    cdef np.ndarray[np.float64_t, ndim=1] cart_vect = np.zeros(3)
    # Trig alias.
    cdef double sin_theta = sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * cos(spherical_vect[1])
    return cart_vect

cpdef  np.ndarray[np.float64_t, ndim=1] celestial_to_cartesian(np.ndarray[np.float64_t, ndim=1] celestial_vect):
    """Convert the spherical coordinate vector [r, dec, ra] to the Cartesian vector [x, y, z]."""
    celestial_vect[1]=np.pi/2. - celestial_vect[1]
    return spherical_to_cartesian(celestial_vect)

cpdef  np.ndarray[np.float64_t, ndim=1] cartesian_to_celestial(np.ndarray[np.float64_t, ndim=1] cartesian_vect):
    """Convert the Cartesian vector [x, y, z] to the celestial coordinate vector [r, dec, ra]."""
    spherical_vect = cartesian_to_spherical(cartesian_vect)
    spherical_vect[1]=np.pi/2. - spherical_vect[1]
    cdef np.ndarray[np.float64_t, ndim=1] coord = np.zeros(3)
    coord[0] = spherical_vect[0]
    coord[1] = spherical_vect[1]
    coord[2] = spherical_vect[2]
    return coord

cpdef  double Jacobian(np.ndarray[np.float64_t, ndim=1] cartesian_vect):
    d = sqrt(cartesian_vect.dot(cartesian_vect))
    d_sin_theta = sqrt(cartesian_vect[:-1].dot(cartesian_vect[:-1]))
    return d*d_sin_theta
