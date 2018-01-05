#os.system("D:\prog\python_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf D:\m_\ecPro\pacal\demos\demo.py")
#! Discrete random variables using PaCal
#! =========================================
#$ This demo ilustrates hot to use **PaCal** with discrite random variables.
#$

from __future__ import print_function

from pylab import figure, show, subplot

from pacal import *
from pacal.distr import demo_distr, sign
from matplotlib.pyplot import legend

if __name__ == "__main__":
    #! Constuctor
    #!------------
    #!
    U = UniformDistr(0, 1)
    M = DiscreteDistr(xi=[0.0, 1.0, 2.0, 3.0], pi=[0.3, 0.4, 0.2, 0.1])
    S = M + U
    params.general.warn_on_dependent = False
    SM = min(S, S)
    params.general.warn_on_dependent = True

    #!
    M.summary()
    S.summary()
    SM.summary()

    figure()
    subplot(221)
    S.plot()
    subplot(222)
    SM.plot()
    subplot(223)
    S.get_piecewise_cdf().plot()
    subplot(224)
    SM.get_piecewise_cdf().plot()

    #! Binomial and Bernoulli distributions
    #! -------------------------------------
    b5 = ZeroDistr()
    for i in range(5):
        b5 += DiscreteDistr(xi=[0, 1], pi=[0.2, 0.8])
    figure()
    subplot(131)
    b5.plot()
    subplot(132)
    b5.get_piecewise_cdf().plot()
    b5.summary()
    subplot(133)
    b5.hist()

    #! Difference of binomial distributions
    #!-----------------------------------------
    #!
    #! A test for equality of two proportions
    figure()
    b1 = BinomialDistr(5, 0.4)
    b2 = BinomialDistr(3, 0.45)
    bd = b1 / 5 - b2 / 3
    bd.summary()
    subplot(311)
    bd.plot()
    subplot(312)
    bd.hist()
    subplot(313)
    bd.get_piecewise_cdf().plot()

    #! Mixing continuous and discrete distributions
    #!----------------------------------------------
    d = DiscreteDistr(xi=[1, 2], pi=[0.2, 0.8])
    U = UniformDistr(0, 2)
    A1 = d + U
    A2 = d * U
    A3 = d / U
    A4 = U / d
    figure()
    subplot(221)
    A1.plot()
    A1.hist()
    subplot(222)
    A2.plot()
    A2.hist()
    subplot(223)
    A3.plot(xmax=6.0)
    A3.hist(xmin=0.5, xmax=6)
    subplot(224)
    A4.plot()
    A4.hist()

    A1.summary()
    A2.summary()
    A3.summary()
    A4.summary()

    figure()
    subplot(221)
    A1.get_piecewise_cdf().plot()
    subplot(222)
    A2.get_piecewise_cdf().plot()
    subplot(223)
    A3.get_piecewise_cdf().plot(xmax=6.0)
    subplot(224)
    A4.get_piecewise_cdf().plot()


    #! Mixture distributions
    #!----------------------

    d = DiscreteDistr(xi=[-1, 2, 6], pi=[0.2, 0.4, 0.4])
    N = NormalDistr()
    GM = d + N
    figure()
    GM.summary()
    GM.plot()


    #! Sign, Abs
    #! ----------
    figure()
    N = NormalDistr(0.1,1)
    S = sign(N)
    A = abs(N)
    S.plot(color='r', linewidth=2.0, label="sign(N)")
    A.plot(color='g', linewidth=2.0, label="abs(N)")
    N.plot(color='b', linewidth=2.0, label="N")
    legend()
    figure()
    params.general.warn_on_dependent = False
    demo_distr(S * A)#, theoretical=N)
    params.general.warn_on_dependent = True

    #! min, max, Conditional distributions
    #! ------------------------------------
    #!
    #!
    figure()
    E = max(ExponentialDistr() - 1, ZeroDistr())
    subplot(211)
    E.plot(linewidth=2.0)
    E.summary()
    subplot(212)
    params.general.warn_on_dependent = False
    S = E + E + E + E + E + E
    params.general.warn_on_dependent = True
    S.plot(linewidth=2.0)
    S.summary()


    figure()
    E = ExponentialDistr()
    E1 = CondGtDistr(E, 1)
    E2 = CondLtDistr(E, 1)
    subplot(311)
    E.plot(linewidth=2.0)
    subplot(312)
    E1.plot(linewidth=2.0)
    E1.summary()
    subplot(313)
    E2.plot(linewidth=2.0)
    E2.summary()

    #! Memorylessness of exponential distribution
    figure()
    demo_distr(E1 - 1, theoretical=E)

    show()
