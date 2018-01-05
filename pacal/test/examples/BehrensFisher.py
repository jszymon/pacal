"""The Behrens-Fisher problem."""
from __future__ import print_function

from numpy import *
from pacal import *


def BF_distr(n1, n2, v1, v2):
    """The distribution of Behrens-Fisher statistic for known
    variances."""
    num = NormalDistr(0, sqrt(v1 + v2))
    den = sqrt((float(v1) / (n1-1)) * ChiSquareDistr(n1 - 1) + (float(v2) / (n2-1)) * ChiSquareDistr(n2 - 1))
    t = num / den
    return t

def Welch_approx(n1, n2, v1, v2):
    """Welch approximation."""
    df = (v1/n1 + v2/n2)**2 / (v1**2/(n1**2 * (n1-1)) + v2**2/(n2**2 * (n2-1)))
    return StudentTDistr(df)

def Compare_Welch(n1, n2, v1, v2):
    """Compare exact distribution with Welch approaximation."""
    t = BF_distr(n1, n2, v1, v2)
    w = Welch_approx(n1, n2, v1, v2)
    return t, w, t - w
    

def BF_pvalue(x1, x2, v1 = None, v2 = None, Welch = True):
    """Compute the p-value for the Behrens-Fisher problem for samples
    x1 and x2.

    if v1 and/or v2 are unknown, the p-value is conditional on sample
    variance(s).  This means that the test is safe."""
    n1  = len(x1)
    n2  = len(x2)
    mu1 = mean(x1)
    mu2 = mean(x2)
    if v1 is None:
        v1  = var(x1, ddof = 1)
    if v2 is None:
        v2  = var(x2, ddof = 1)
    print("x1: mean={0}, var={1}, n={2}".format(mu1, v1, n1))
    print("x2: mean={0}, var={1}, n={2}".format(mu2, v2, n2))

    T = (mu1 - mu2) / sqrt(v1 / n1 + v2 / n2)
    print("test statistic T={0}".format(T))

    # distribution of the statistic under null hypothesis
    t = BF_distr(n1, n2, v1, v2)
    tpdf = t.get_piecewise_pdf()
    pv = tpdf.integrate(-Inf, -abs(T)) + tpdf.integrate(abs(T), Inf)
    ret = pv, t
    if Welch:
        wd = Welch_approx(n1, n2, v1, v2)
        wdpdf = wd.get_piecewise_pdf()
        pvw = wdpdf.integrate(-Inf, -abs(T)) + wdpdf.integrate(abs(T), Inf)
        ret = ret + (pvw, wd)
    return ret

def compare_DF_Welch(n1, n2, v1, v2, nsamp = 10, alpha = 0.05):
    """Compare DF_distr with Welch_approx by counting the number of
    rejections of correct H0."""
    nrej_BF = 0
    nrej_W = 0
    for i in range(nsamp):
        x1 = normal(2, 1, n1)
        x2 = normal(2, 1, n2)
        pv, t, pvw, wd = BF_pvalue(x1, x2)
        if pv < alpha:
            nrej_BF += 1
        if pvw < alpha:
            nrej_W += 1
        print("pv, pvw=", pv, pvw)
    return float(nrej_BF) / nsamp, float(nrej_W) / nsamp

if __name__ == "__main__":
    from pylab import show, figure
    from numpy.random import seed, normal
    seed(1)

    #x1 = normal(2, 1, 4)
    #x2 = normal(2, 1.5, 6)

    # blad w zerze!!!
    x1 = normal(2, 1, 4)
    x2 = normal(2, 15, 60)

    #x1 = normal(2, 1, 4)
    #x2 = normal(2, 15, 6)

    #x1 = normal(2, 1, 4)
    #x2 = normal(3, 15, 6)

    print("x1 =", x1)
    print("x2 =", x2)
    pv, t, pvw, wd = BF_pvalue(x1, x2)
    print(pv, pvw)
    t.plot()
    t.summary()
    wd.plot()
    wd.summary()
    figure()
    (t.get_piecewise_pdf() - wd.get_piecewise_pdf()).plot()
    show()

    # print compare_DF_Welch(4, 6, 1.0, 1.5, alpha = 0.1, nsamp = 10)
