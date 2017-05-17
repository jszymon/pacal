#!===============================
#! Demo of central limit theorem
#!===============================

from __future__ import print_function

import sys

from pylab import *
from pacal import *
from pacal import params
import time

params.general.warn_on_dependent = False
if __name__ == "__main__":
    colors = "kbgrcmy"
    def central_limit_demo(X, N = 5, xmin = None, xmax = None, ymax = None, **args):
        tic=time.time()
        figure()
        title("Limit of averages of " + X.getName())
        X.plot(linewidth = 4, color = "c", **args)
        Y = X
        print("Limit of averages of " + X.getName() + ": ", end=' ')
        for i in range(N-1):
            print(i+2, end=' ')
            sys.stdout.flush()
            Y += X
            (Y/(i+2)).plot(color = colors[i%len(colors)], **args)
        if xmin is not None:
            xlim(xmin = xmin)
        if xmax is not None:
            xlim(xmax = xmax)
        ylim(ymin = 0)
        if ymax is not None:
            ylim(ymax = ymax)
        print()
        print("time===", time.time()-tic)
        #show()

    #!----------------------
    #! uniform distribution
    #!----------------------
    X = UniformDistr(0,1)
    central_limit_demo(X, xmin=-0.1, xmax=1.1)

    #!----------------------
    #! Chi^2_1
    #!----------------------
    X = ChiSquareDistr(1)
    central_limit_demo(X, N=5, ymax=1.5, xmax=3)

    #!----------------------
    #! Student T w. 2df
    #!----------------------
    X = StudentTDistr(2)
    central_limit_demo(X, N = 5, xmin=-5, xmax=5)

    #!----------------------
    #! a ratio distribution
    #!----------------------
    X = UniformDistr(1,3) / UniformDistr(-2,1)
    central_limit_demo(X, N = 5, xmin=-5, xmax=5)

    #!----------------------
    #! Cauchy distribution
    #!----------------------
    X = CauchyDistr()
    central_limit_demo(X, xmin = -10, xmax = 10)

    #!----------------------
    #! Levy distribution
    #!----------------------
    X = LevyDistr()
    central_limit_demo(X, xmax=5, numberOfPoints = 10000)

    show()
