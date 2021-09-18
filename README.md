# hdp-population
Source code for the inference of black hole mass function using (H)DPGMM.
This code implements a Gibbs sampler to explore a hierarchy of DPGMMs.
If you use this code, please cite the paper https://arxiv.org/pdf/2109.05960.pdf.

## Requirements:
* Numpy
* Scipy
* Matplotlib
* Numba
* Ray
* CPNest (https://github.com/johnveitch/cpnest - for multidim only)

Installation: 
  python setup.py build_ext --inplace
