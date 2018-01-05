#!===================================================================
#! Examples of distributions with singularities
#!===================================================================

from __future__ import print_function

from functools import partial
from pylab import *
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset

from pacal import *
from pacal.distr import demo_distr

if __name__ == "__main__":
    #!-------------------------------------------
    #! Product of two shifted normal variables
    #!-------------------------------------------
    #! such a product always has a singularity at 0, but the further the factors' means are from zero, the 'lighter' the singularity becomes
    figure()
    d = NormalDistr(0,1) * NormalDistr(0,1)
    demo_distr(d, ymax=1.5, xmin=-5, xmax=5)
    #show()

    figure()
    d = NormalDistr(1,1) * NormalDistr(1,1)
    demo_distr(d)
    #show()

    figure()
    d = NormalDistr(2,1) * NormalDistr(2,1)
    demo_distr(d)
    #show()

    figure()
    d = NormalDistr(3,1) * NormalDistr(3,1)
    d.plot()
    d.hist()
    ax=gca()
    axins = zoomed_inset_axes(ax, 6, loc=1)
    d.plot(xmin=-1.5, xmax=1.5)
    axins.set_xlim(-1.5, 1.5)
    xticks(rotation="vertical")
    axins.set_ylim(0, 0.01)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    #show()

    figure()
    d = NormalDistr(4,1) * NormalDistr(4,1)
    d.plot()
    d.hist()
    ax=gca()
    axins = zoomed_inset_axes(ax, 12000, loc=1)
    d.plot(xmin=-.001, xmax=.001)
    axins.set_xlim(-.001, .001)
    xticks(rotation="vertical")
    axins.set_ylim(0.000072, 0.000075)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    #show()



    #   demo_distr(UniformDistr(0,1) * NormalDistr(0,1) * NormalDistr(0,1) * NormalDistr(0,1) * NormalDistr(0,1)* NormalDistr(0,1) * NormalDistr(0,1))

    #!-------------------------------------------
    #! Product of six uniform distributions
    #!-------------------------------------------
    def prod_uni_pdf(n, x):
        pdf = (-log(x)) ** (n-1)
        for i in range(2, n):
            pdf /= i
        return pdf
    figure()
    d = UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1)
    demo_distr(d, ymax=100, xmin=-0.01, xmax=0.3, theoretical = partial(prod_uni_pdf, 6))
    #show()

    #   figure()
    #   demo_distr(UniformDistr(0,1.1) * UniformDistr(0,1.1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1))

    show()
