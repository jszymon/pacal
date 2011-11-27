#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    have_Cython = True
except ImportError:
    print "Warning: Cython could not be imported.  Will use slower version."
    have_Cython = False

if have_Cython:
    Cython_args = {
        "cmdclass" : {'build_ext': build_ext},
        "ext_modules" : [Extension("pacal.bary_interp", ["pacal/bary_interp.pyx"])],
        #"ext_modules" : [Extension("pacal.bary_interp", ["pacal/bary_interp.pyx"], include_dirs=["c:/Python27/Lib/site-packages/numpy/core/include/"])],
        }
else:
    Cython_args = {}
    
setup(
    name='PaCal',
    version='1.1',
    description ='PaCal - ProbAbilistic CALculator',
    author='Szymon Jaroszewicz, Marcin Korzen',
    author_email='s.jaroszewicz@ipipan.waw.pl, mkorzen@wi.zut.edu.pl',
    license='GNU General Public License V.3 or later',
    url='http://pacal.sf.net',
    long_description=open('README.txt').read(),
    requires=['Python (>=2.6,<3.0)', 'numpy (>=1.4)', 'matplotlib (>=1.0)', 'Cython', 'Sympy'],

    packages=['pacal', 'pacal.stats', 'pacal.examples', 'pacal.examples.springer_book', 'pacal.examples.depvars.two_variables'],
    **Cython_args
)
