#!===================================
#! Demo of noncentral distributions.
#!===================================

from __future__ import print_function

from pacal import *
from pylab import figure, legend, title, xlim, ylim
import time

colors = "kbgrcmy"

def plot_nonc(d, titl = "", lim = None):
    figure()
    print("----------------------------------------------------------------")
    for i, nc in enumerate([0, 1, 2, 5, 10]):
        ncd = d(nc)
        print(ncd)
        ncd.summary(show_moments=True)
        ncd.plot(label = "nonc=" + str(nc), color = colors[i%len(colors)])
        print()

    if lim is not None:
        xlim(lim[0], lim[1])
        ylim(lim[2], lim[3])
    title(titl)
    legend()
    #show()

if __name__ == "__main__":
    tic = time.time()
    #!-------------
    #! Noncentral T
    #!-------------
    #!
    plot_nonc(lambda nc: NoncentralTDistr(2, nc), titl = "NoncentralT(2, nonc)")
    plot_nonc(lambda nc: NoncentralTDistr(10, nc), titl = "NoncentralT(10, nonc)")

    #!-----------------------
    #! Noncentral Chi square
    #!-----------------------
    #!
    plot_nonc(lambda nc: NoncentralChiSquareDistr(1, nc), titl = "NoncentralChiSquare(1, nonc)")
    plot_nonc(lambda nc: NoncentralChiSquareDistr(2, nc), titl = "NoncentralChiSquare(2, nonc)")
    plot_nonc(lambda nc: NoncentralChiSquareDistr(10, nc), titl = "NoncentralChiSquare(10, nonc)")

    #!-------------------------
    #! Noncentral Beta
    #!-------------------------
    #!
    plot_nonc(lambda nc: NoncentralBetaDistr(1, 1, nc), titl = "NoncentralBeta(1, 1, nonc)")
    plot_nonc(lambda nc: NoncentralBetaDistr(10, 15, nc), titl = "NoncentralBeta(10, 15, nonc)")

    #!-------------
    #! Noncentral F
    #!-------------
    #!
    plot_nonc(lambda nc: NoncentralFDistr(1, 1, nc), titl = "NoncentralF(1, 1, nonc)", lim = [-0.1, 3, 0, 0.9])
    plot_nonc(lambda nc: NoncentralFDistr(10, 20, nc), titl = "NoncentralF(10, 20, nonc)")
    print("toc=", time.time() - tic)
    show()
