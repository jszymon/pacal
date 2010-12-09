from pacal import *

def noncentral_t(df, mu, x):
    nonc_t = NormalDistr(mu, 1) / sqrt(ChiSquareDistr(df) / df)
    pdf = nonc_t(x)
    cdf = nonc_t.cdf(x)
    return nonc_t, pdf[0], cdf[0]



import mpmath as mp
mp.mp.dps = 200
def multiprec_pdf(df, mu, x):
    df = mp.mpf(str(df))
    mu = mp.mpf(str(mu))
    x  = mp.mpf(str(x))
    pdf = (mp.exp(-mu**2 / 2) * 2**(-df) * df**(df/2) * mp.gamma(df+1) * \
           ((mp.sqrt(2) * x * mu * (x**2+df)**(-df/2-1) * mp.hyp1f1(df/2+1,1.5,(mu**2 * x**2)/(2 * (x**2+df))))/(mp.gamma((df+1)/2))+((x**2+df)**(-df/2-0.5) * mp.hyp1f1((df+1)/2,0.5,(mu**2 * x**2)/(2 * (x**2+df))))/(mp.gamma(df/2+1))))/(mp.gamma(df/2))
    return pdf
def multiprec_cdf(df, mu, x):
    return mp.quad(lambda x: multiprec_pdf(df, mu, x), [-mp.inf, mp.mpf(x)])


def test_noncentral_t(df, mu, x, exact_pdf = None, exact_cdf = None,
                      R_pdf = None, R_cdf = None,
                      Mathematica_pdf = None, Mathematica_cdf = None,
                      SAS_pdf = None, SAS_cdf = None,
                      ):
    if exact_pdf is None:
        exact_pdf = float(multiprec_pdf(df, mu, x))
        print "exact_pdf", repr(exact_pdf)
    if exact_cdf is None:
        exact_cdf = float(multiprec_cdf(df, mu, x))
        print "exact_cdf", repr(exact_cdf)
    d_, f, P = noncentral_t(df, mu, x)
    print f, P
    pdf_err = abs(f - exact_pdf)
    cdf_err = abs(P - exact_cdf)
    def str_errors(name, df, mu, x, pdf, cdf, exact_pdf, exact_cdf):
        if pdf is None and cdf is None:
            return ""
        r_err = name + " df={0},mu={1},x={2}  ".format(df, mu, x)
        if pdf is not None:
            r_err += " pdf_err={0}".format(abs(exact_pdf - pdf))
        if cdf is not None:
            r_err += " cdf_err={0}".format(abs(exact_cdf - cdf))
        return r_err + "\n"
    print str_errors("PaCal:      ", df, mu, x, f, P, exact_pdf, exact_cdf),
    print str_errors("R:          ", df, mu, x, R_pdf, R_cdf, exact_pdf, exact_cdf),
    print str_errors("Mathematica:", df, mu, x, Mathematica_pdf, Mathematica_cdf, exact_pdf, exact_cdf),
    print str_errors("SAS:        ", df, mu, x, SAS_pdf, SAS_cdf, exact_pdf, exact_cdf),


#SAS code:
#data _null_;
#p = pdf('T', -1000, 7, 10.3);
#p = format p e32.31;
#put p;
#run;

# Correct values are computed using sympy at multidigit precision.
test_noncentral_t(7, 10.3, 5.1, None, None,
                  R_pdf = 0.003581733882073113, R_cdf = 0.001191483320239349,
                  Mathematica_pdf = 0.0035817338822428215,
                  SAS_pdf = 3.5817338822428000000e-3, SAS_cdf = 1.1914833206298000000e-3)
test_noncentral_t(7, 10.3, -50.0, None, None,
                  R_pdf = 1.554312234475223e-17, R_cdf = 9.35918009759007e-14, # R gives a warning of low accuracy
                  Mathematica_pdf = 1.542256935359599e-21,
                  SAS_pdf = 1.3157713027446000000e-40, SAS_cdf = 9.400060780335900000e-40)
test_noncentral_t(7, 10.3, -1e3, None, None,
                  R_pdf = 0, R_cdf = 9.39248678832882e-14, # R gives a warning of low accuracy
                  Mathematica_pdf = 0,
                  SAS_cdf = 7.34842111248630000e-49,# SAS gives an error for the PDF in this case.  Strangely it works for x=-820 but not for -820.5 (and below)
                  )


#d, f, P = noncentral_t(4, 2, 5.0)
#from pylab import show
#d.plot()
#show()

