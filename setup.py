#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

setup(
    name='PaCal',
    version='0.99',
    description='PaCal - ProbAbilistic CALculator',
    author = 'Szymon Jaroszewicz, Marcin Korzen',
    author_email = 's.jaroszewicz@itl.waw.pl, mkorzen@wi.zut.edu.pl',
    license='GNU General Public License',
    url='http://pacal.sf.net',
    long_description=open('README.txt').read(),

    requires=['Python (>=2.6,<3.0)', 'numpy (>=1.3)', 'matplotlib (>=0.99)', 'Cython'],

    packages=['pacal','pacal.examples'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pacal.bary_interp", ["pacal/bary_interp.pyx"])]
)
