import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

ext_modules=[
             Extension("hdpgmm.utils",
                       sources=[os.path.join("hdpgmm","utils.pyx")],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=['hdpgmm', numpy.get_include()]
                       )
             ]
ext_modules = cythonize(ext_modules)
setup(
      name = 'hdpgmm/utils',
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include()]
      )

setup(
    name = 'hdpgmm',
    use_scm_version=True,
    description = '(H)DPGMM: a Hierarchy of DPGMM',
    author = 'Walter Del Pozzo, Stefano Rinaldi',
    author_email = 'walter.delpozzo@unipi.it, stefano.rinaldi@phd.unipi.it',
    url = 'https://git.ligo.org/stefano.rinaldi/hdp-population',
    python_requires = '>=3.6',
    packages = ['hdpgmm'],
    include_dirs = [numpy.get_include()],
    setup_requires=['numpy', 'cython', 'setuptools_scm'],
    entry_points={},
    package_data={"": ['*.c', '*.pyx', '*.pxd']},
    ext_modules=ext_modules,
    )
