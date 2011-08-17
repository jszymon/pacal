"""Demo of noncentral distributions."""

from pacal import *
from pylab import figure, legend


colors = "kbgrcmy"

def plot_nonc(d):
    figure()
    for i, nc in enumerate([0, 1, 2, 5, 10]):
        ncd = d(nc)
        print ncd
        ncd.summary()
        ncd.plot(label = "nonc=" + str(nc), color = colors[i%len(colors)])
    legend()


plot_nonc(lambda nc: NoncentralTDistr(2, nc))
plot_nonc(lambda nc: NoncentralTDistr(10, nc))
plot_nonc(lambda nc: NoncentralChiSquareDistr(2, nc))
plot_nonc(lambda nc: NoncentralChiSquareDistr(10, nc))
show()
