#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import print_function

from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
    have_Cython = True
except ImportError:
    print("Warning: Cython could not be imported.  Will use slower pure Python version.")
    have_Cython = False

if have_Cython:
    import numpy as np
    Cython_args = {
        "cmdclass" : {'build_ext': build_ext},
        "ext_modules" : [Extension("pacal.bary_interp", ["pacal/bary_interp.pyx"], include_dirs=[np.get_include()])],
        }
else:
    Cython_args = {}
    
setup(
    name='PaCal',
    version='1.6.1',
    description ='PaCal - ProbAbilistic CALculator',
    author='Szymon Jaroszewicz, Marcin Korzeń',
    author_email='s.jaroszewicz@ipipan.waw.pl, mkorzen@wi.zut.edu.pl',
    license='GNU General Public License V.3 or later',
    url='http://pacal.sf.net',
    long_description=open('README.md').read(),
    python_requires=">=3.5",
    install_requires=['numpy>=1.13', 'matplotlib>=2.0.0', 'Cython', 'Sympy>=1.0.0', 'scipy>=1.1.0'],

    packages=['pacal', 'pacal.stats', 'pacal.depvars',
              'pacal.examples', 'pacal.examples.springer_book',
              'pacal.examples.dependent',
              'pacal.examples.dependent.two_variables'],
    **Cython_args
)
