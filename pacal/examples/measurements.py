#!==============================================
#! Examples of physical measurements
#!==============================================

from __future__ import print_function

import numpy
from pylab import *

from pacal import *
from pacal.distr import demo_distr

if __name__ == "__main__":
    ##!----------------------------------
    ##! coefficient of thermal expansion
    ##!----------------------------------
    #L0 = UniformDistr(9,10)
    #L1 = UniformDistr(11,12)
    #dT = NormalDistr(1,2)
    #k = (L1/L0 - 1)/dT
    #k.plot(linewidth=3, color = "k")
    #k.hist(xmin=-1,xmax=1,color="0.75")
    #k.summary()
    #print "P(K<0) - NormalDistr(1,2).cdf(0) =", k.cdf(0) - NormalDistr(1,2).cdf(0)
    #xlim(-1,1)
    #ylim(ymin=0)
    #title("Distribution of thermal expansion coefficient")


    #!----------------------------------------
    #! combining two independent measurements
    #!----------------------------------------

    try:
        from scipy.optimize.optimize import fminbound
    except ImportError:
        print("Scipy not available, exiting...")
        import sys
        sys.exit(10)

    E1 = UniformDistr(0.5,2.0)   # error of the fist measurement
    #E2 = NormalDistr()        # error of the second measurement
    #E2 = ParetoDistr() | Between(0,4)        # error of the second measurement
    E2 = UniformDistr(0.0,1.5)        # error of the second measurement

    # E1 = UniformDistr(0,2)
    # E2 = BetaDistr(2,2)
    # E2 = BetaDistr(0.5,0.5)


    E1.summary()
    E2.summary()


    def E(alpha):
        #"""Error of a linear combination of measurements."""
        alpha = numpy.squeeze(alpha)
        #return alpha*E1 + (1 - alpha)*E2
        """Error of a geometric combination of measurements."""
        return exp(log(E1)*alpha +  log(E2)*(1-alpha))

    print()
    print("Combining measurements for optimal variance")
    alphaOptVar = fminbound(lambda alpha: E(alpha).var(), 0, 1, xtol = 1e-16)
    print("alpha for optimal variance = ", alphaOptVar)
    dopt = E(alphaOptVar)
    dopt.summary()

    print()
    print("Combining measurements for optimal Median Absolute Deviance")
    alphaOptMad = fminbound(lambda alpha: E(alpha).medianad(), 0, 1, xtol = 1e-16)
    print("alpha for optimal MAD = ", alphaOptMad)
    dopt = E(alphaOptMad)
    dopt.summary()

    print()
    print("Combining measurements for optimal 95% confidence interval")
    alphaOptIQrange = fminbound(lambda alpha: E(alpha).iqrange(0.025), 0, 1, xtol = 1e-16)
    print("alpha for optimal 95% c.i. = ", alphaOptIQrange)
    dopt = E(alphaOptIQrange)
    dopt.summary()

    print()
    print("Combining measurements for optimal entropy")
    alphaOptEntropy = fminbound(lambda alpha: E(alpha).entropy(), 0, 1, xtol = 1e-16)
    print("alpha for optimal entropy = ", alphaOptEntropy)
    dopt = E(alphaOptEntropy)
    dopt.summary()

    print("-----------------------")
    figure()
    #E1.plot(color='k')
    #E2.plot(color='k')
    E(alphaOptVar).plot(color='r', label="optimal variance")
    E(alphaOptMad).plot(color='g', label="optimal MAD")
    E(alphaOptIQrange).plot(color='b', label="optimal 95% c.i.")
    E(alphaOptEntropy).plot(color='k', label="optimal entropy")
    title("Density of combined measurement error")
    legend()

    figure()
    E(alphaOptVar).get_piecewise_cdf().plot(color='r', label="optimal variance")
    E(alphaOptMad).get_piecewise_cdf().plot(color='g', label="optimal MAD")
    E(alphaOptIQrange).get_piecewise_cdf().plot(color='b', label="optimal 95% c.i.")
    E(alphaOptEntropy).get_piecewise_cdf().plot(color='k', label="optimal entropy")
    title("Cumulative distribution function of combined measurement error")
    legend(loc="lower right")

    show()
