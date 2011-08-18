"""Base distribution classes."""

from numpy import array, zeros_like, unique, concatenate, isscalar
from numpy import sqrt, pi, exp, arctan, tan
from numpy.random import normal, uniform, chisquare
from scipy.stats import chi2;
#from scipy.integrate import quad, Inf, quadrature

from integration import integrate_clenshaw
from integration import integrate_fejer2, integrate_fejer2_pinf
from integration import integrate_fejer2_minf, integrate_fejer2_pminf

def plotdistr(d, l = -10, u = 10, numberOfPoints = 1000):
    X = linspace(l, u, numberOfPoints)
    #Y = [d.pdf(x) for x in X] # it should be vectorized
    Y = d.pdf(X) # doesn't work yet 
    plot(X,Y)
    
def histdistr(d, n = 1000000, l = None, u = None, bins = 50):
    if l is None and u is None:
        X = d.rand(n, None)
        allDrawn = len(X)
    else:
        X = []
        allDrawn = 0
        while len(X) < n:
            x = d.rand(n - len(X))
            allDrawn = allDrawn + len(x)
            if l is not None:
                x = x[(l <= x)]
            if u is not None:
                x = x[(x <= u)]
            X = hstack([X, x])
    dw = (X.max() - X.min()) / bins
    w = (float(n)/float(allDrawn)) / n / dw
    counts, binx = histogram(X, bins)
    width = binx[1] - binx[0]
    for c, b in zip(counts, binx):
        bar(b, float(c) * w, width = width, alpha = 0.25)


class Distr(object):
    def __init__(self, parents = []):
        self.parents = parents        
    def pdf(self,x):
        return None
    def logPdf(self,x):
        return log(self.pdf())
    def rand_raw(self, n = None):
        """Generates random numbers without tracking dependencies.

        This method will be implemented in subclasses implementing
        specific distributions.  Not intended to be used directly."""
        return None
    def rand(self, n = None, cache = None):
        """Generates random numbers while tracking dependencies.

        if n is None, return a scalar, otherwise, an array of given
        size."""
        if cache is None:
            cache = {}
        if id(self) not in cache:
            cache[id(self)] = self.rand_raw(n)
        return cache[id(self)]
    # 
    # TODO nalezy dodac funkcje copy, clone, lub cos w tym stylu
    # 
    # 

class NormDistr(Distr):
    def __init__(self, mu=0, sigma=1):        
        super(NormDistr, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.twosigma2 = 2 * sigma * sigma
        self.nrm = 1.0 / (self.sigma * sqrt(2*pi))
        self.type='normal({0},{1})#{2}'.format(mu, sigma, id(self))
        # put breaks at inflection points
        self.breaks = [self.mu - self.sigma, self.mu + self.sigma]
    def pdf(self,x):
        q = (x-self.mu)**2 / self.twosigma2
        f = self.nrm * exp(-q) 
        return f
    def rand_raw(self, n = None):  # None means return scalar
        return normal(self.mu, self.sigma, n)
class UniformDistr(Distr):
    def __init__(self, a, b):
        super(UniformDistr, self).__init__()
        a = float(a)
        b = float(b)
        self.a = a
        self.b = b
        self.p = 1.0 / (b-a)
        self.type='uniform({0},{1})#{2}'.format(self.a, self.b, id(self))
        self.breaks = [a, b]
    def pdf(self,x):
        return self.p * ((x<=self.b) & (x>=self.a))
    def rand_raw(self, n = None):
        return uniform(self.a, self.b, n)
    
class Chi2Distr(Distr):
    def __init__(self, df):
        super(Chi2Distr, self).__init__()
        df = float(df)
        self.df = df        
        self.type='chi2({0})#{1}'.format(self.df, id(self))
        self.breaks = [0]
    def pdf(self,x):
        return chi2(self.df).pdf(x);                      
    def rand_raw(self, n = None):
        return chisquare(self.df, n)
class ConstDistr(Distr):
    def __init__(self, c):
        super(ConstDistr, self).__init__()
        self.c = c        
        self.type='const={0}'.format(self.c)
    def pdf(self,x):
        if x==self.c:
            return 1
        else:
            return 0                      
    def rand_raw(self, n = None):
        return [self.c for i in range(n)]

class OpDistr(Distr):
    """Base class for operations on distributions.

    Currently only does caching for random number generation."""
    def rand(self, n = 1, cache = None):
        if cache is None:
            cache = {}
        if id(self) not in cache:
            cache[id(self)] = self.rand_op(n, cache)
        return cache[id(self)]
        

class NegDistr(OpDistr):
    def __init__(self, d):
        super(NegDistr, self).__init__([d])
        self.d = d
        self.type='negation'        
        self.breaks = [-b for b in d.breaks]
    def pdf(self,x):
        return self.d.pdf(-x)
    def rand_op(self, n, cache):
        return -self.d.rand(n, cache)
class ConstMulDistr(OpDistr):
    """Multiply random variable by a constant"""
    def __init__(self, d, c):
        super(ConstMulDistr, self).__init__(d)
        self.d = d
        self.c = c
        self._1_c = 1.0 / c
        self.type='mulByC={0}'.format(c)                
        self.breaks = [self.c * b for b in d.breaks]
    def pdf(self,x):
        f = self.d.pdf(x * self._1_c) * abs(self._1_c)
        return f
    def rand_op(self, n, cache):
        return self.c * self.d.rand(n, cache)
class FuncDistr(OpDistr):
    """Injective function of random variable"""
    def __init__(self, d, f, f_inv, f_inv_deriv):
        super(FuncDistr, self).__init__()
        self.d = d
        self.f = f
        self.f_inv = f_inv
        self.f_inv_deriv = f_inv_deriv
        self.type='fun'
        self.breaks = [self.f(b) for b in d.breaks]
    def pdf(self,x):
        f = self.d.pdf(self.f_inv(x)) * abs(self.f_inv_deriv(x))
        if not isfinite(f):
            f = 0
        return f
    def rand_op(self, n, cache):
        return self.f(self.d.rand(n, cache))
    
class SquareDistr(OpDistr):
    """Injective function of random variable"""
    def __init__(self, d):
        super(SquareDistr, self).__init__()
        self.d = d
        self.type='sqr'
        self.breaks = sorted(set([0] + [b*b for b in d.breaks]))
    def pdf(self,x):
        if x <= 0:
            f = 0
        else:
            f = (self.d.pdf(-sqrt(x)) + self.d.pdf(sqrt(x))) /(2*sqrt(x))
        return f
    def rand_op(self, n, cache):
        r = self.d.rand(n, cache)
        return r * r
    
class SumDistr(OpDistr):
    def __init__(self, d1, d2):
        super(SumDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
        self.type='sum'                
        breaks = [b1 + b2 for b1 in self.d1.breaks for b2 in self.d2.breaks]
        self.breaks = sorted(set(breaks))
    def __conv(self, x, tau):
        return self.d1.pdf(x-tau) * self.d2.pdf(tau)
    def pdf(self,x):
        if isscalar(x):
            breaks1 = self.d2.breaks
            breaks2 = [x - b for b in self.d1.breaks]
            #breaks = sorted(set(breaks1 + breaks2))
            breaks = unique(concatenate([breaks1, breaks2]))
            # debug
            if doplot__:
                X = linspace(-10,10,1000)
                Y = [self.__conv(x, tau) for tau in X]
                plot(X,Y)
                for b in breaks:
                    axvline(b)
                show()
            # end debug
            f = integrate(lambda tau: self.__conv(x, tau), points=breaks, limit=1000)[0]
        else:
            f = array([self.pdf(xsc) for xsc in x])
        return f
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 + r2
    
class MulDistr(OpDistr):
    def __init__(self, d1, d2):
        super(MulDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
        self.type='mul'                
        breaks = [b1 * b2 for b1 in self.d1.breaks for b2 in self.d2.breaks]
        self.breaks = sorted(set(breaks))
    def __conv(self, x, tau):
        if tau == 0: return 0
        return (1.0 / abs(tau)) * self.d1.pdf(tau) * self.d2.pdf(x / tau)
    def pdf(self,x):
        breaks1 = self.d1.breaks
        breaks2 = [x / b for b in self.d2.breaks if b != 0]
        breaks = [0] + breaks1 + breaks2
        f = integrate(lambda tau: self.__conv(x, tau), points=breaks, limit=100)[0]
        return f
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 * r2
    
class DivDistr(OpDistr):
    def __init__(self, d1, d2):
        super(DivDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
        self.type='div'                
        breaks = [b1 / b2 for b1 in self.d1.breaks for b2 in self.d2.breaks if b2 != 0]
        self.breaks = sorted(set(breaks))
    def __conv(self, x, tau):
        if isscalar(tau):
            if tau == 0: return 0
            return abs(tau) * self.d1.pdf(tau*x) * self.d2.pdf(tau)
        ret = abs(tau) * self.d1.pdf(tau*x) * self.d2.pdf(tau)
        ret[tau == 0] = 0
        return ret
    def pdf(self,x):
        if isscalar(x):
            breaks1 = self.d2.breaks
            if x != 0:
                #for small x's breaks2 become huge, and the distribution become a single spike which quad cannot handle well
                # the fix below is wrong, does removes good breakpoints e.g. for uniform distr
                # will need a better solution....
                #breaks2 = [b / x for b in self.d1.breaks if self.__conv(x, b/x) > 0] # not too pretty fix
                breaks2 = [b / x for b in self.d1.breaks]
            else:
                breaks2 = []
            breaks = unique(concatenate([[0], breaks1, breaks2])) # 0 comes from |tau| not having derivative there
            #print x, breaks, breaks1, breaks2, self.d1.breaks, self.d2.breaks
            f = integrate(lambda tau: self.__conv(x, tau), points=breaks, limit=100)[0]
        else:
            f = array([self.pdf(xsc) for xsc in x])
        return f
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 / r2

class InterpolatedDistr(Distr):
    """distribution with interpolated pdf."""
    def __init__(self, d):
        super(InterpolatedDistr, self).__init__([d])
        self.d = d
        self.breaks = d.breaks
        self.type='interp'
        if len(self.breaks) == 0:
            self.interp = [ChebyshevInterpolator_PMInf(self.d.pdf)]
        else:
            self.interp =  [ChebyshevInterpolator_MInf(self.d.pdf, self.breaks[0])]
            for i in xrange(len(self.breaks) - 1):
                interp = ChebyshevInterpolator(self.d.pdf, self.breaks[i], self.breaks[i+1])
                self.interp.append(interp)
            self.interp.append(ChebyshevInterpolator_PInf(self.d.pdf, self.breaks[-1]))
        self.err = max([i.err for i in self.interp])
        self.n_nodes = sum([len(i.Xs) for i in self.interp])
    def pdf(self,x):
        if len(self.breaks) == 0:
            return self.interp[0].interp_at(x)
        #TODO: do binary search here
        if isscalar(x):
            if x >= self.breaks[-1]:
                y =  self.interp[-1].interp_at(x)
            else:
                i = 0
                while x > self.breaks[i]:
                    i += 1
                y = self.interp[i].interp_at(x)
        else:
            y = zeros_like(x)
            y[x < self.breaks[0]] = self.interp[0].interp_at(x[x < self.breaks[0]])
            y[x >= self.breaks[-1]] = self.interp[-1].interp_at(x[x >= self.breaks[-1]])
            for i in xrange(len(self.breaks) - 1):
                mask = (x >= self.breaks[i]) & (x < self.breaks[i+1])
                y[mask] = self.interp[i+1].interp_at(x[mask])
        return y
    def rand(self, n, cache = None):
        return self.d.rand(n, cache)

"""
    TODO To najlepiej przeniesc gdyie indziej
    
"""
def integrate(f, points = [], limit = 1000, eps = 1e-12):
    """Integrate function with breaks on (-Inf,Inf) interval.  scipy
    cannot handle this directly."""
    epsabs = eps
    epsrel = 0
    tol = eps
    if len(points) == 0:
        #I, err = quad(f, -Inf, Inf, limit = limit, epsabs = epsabs, epsrel = epsrel)
        I, err = integrate_fejer2_pminf(f)
    else:
        breaks = sorted(points)
        #I1, err1 = quad(f, -Inf, breaks[0], limit = limit, epsabs = epsabs, epsrel = epsrel)
        I1, err1 = integrate_fejer2_minf(f, breaks[0])
        ### using quad
        #I2, err2 = quad(f, breaks[0], breaks[-1], points=breaks[1:-1], limit = limit, epsabs = epsabs, epsrel = epsrel)
        ### using quadrature
        #tol = epsabs
        #I2 = 0
        #err2 = 0
        #for i in xrange(len(breaks)-1):
        #    I2_, err2_ = quadrature(f, breaks[i], breaks[i+1], tol = tol, vec_func = False)
        #    I2 += I2_
        #    err2 += err2_
        #### using Clenshaw
        ##tol = epsabs
        #I2 = 0
        #err2 = 0
        #for i in xrange(len(breaks)-1):
        #    I2_, err2_ = integrate_clenshaw(f, breaks[i], breaks[i+1])#, tol = tol)#, vec_func = False)
        #    I2 += I2_
        #    err2 += err2_
        ####
        #### using Fejer2
        I2 = 0
        err2 = 0
        for i in xrange(len(breaks)-1):
            I2_, err2_ = integrate_fejer2(f, breaks[i], breaks[i+1])#, tol = tol)#, vec_func = False)
            I2 += I2_
            err2 += err2_
        ####
        #I3, err3 = quad(f, breaks[-1], Inf, limit = limit, epsabs = epsabs, epsrel = epsrel)
        I3, err3 = integrate_fejer2_pinf(f, breaks[-1])
        I = I1 + I2 + I3
        err = err1 + err2 + err3
    return I, err

# some ugly debugging
doplot__ = False
def set_plot__():
    global doplot__
    doplot__ = True

if __name__ == "__main__":
    from plotfun import *

    N1 = NormDistr(0,1)
    N2 = NormDistr(1,1)
    N3 = SumDistr(N1, N2)
    N4 = SumDistr(N1, NegDistr(N2))
    N5 = MulDistr(N1, N1)
    N6 = MulDistr(N2, N2)
    N7 = DivDistr(N1, N1)
    N8 = DivDistr(N2, N2)
    N9 = MulDistr(N2, SumDistr(N2, N2))
    N10 = SquareDistr(N1)
    N11 = SumDistr(N10, N10)
    N12 = FuncDistr(N1,
                    f = lambda x: x*x,
                    f_inv = lambda x: sqrt(abs(x)),
                    f_inv_deriv = lambda x: 1.0/(2*sqrt(abs(x))))
    N13 = MulDistr(N1, N1)
    N13prime = MulDistr(N1, NormDistr(0,1))
    N14 = ConstMulDistr(N2,0.5)
    N15 = ConstMulDistr(N2,-1)
    N16 = DivDistr(NormDistr(0,1), NormDistr(0,1))

    U1 = UniformDistr(1,3)
    UN1 = DivDistr(U1, N2)
    UN2 = SumDistr(N1, N2)
    UN3 = SumDistr(U1, NegDistr(UniformDistr(2,5)))
    UN4 = FuncDistr(MulDistr(U1, U1), f = lambda x: x/3.0, f_inv = lambda x: 3*x, f_inv_deriv = lambda x: 3)
    UN5 = DivDistr(UniformDistr(1,2), UniformDistr(3,4))
    UN6 = FuncDistr(UN5, f = arctan, f_inv = tan, f_inv_deriv = lambda x: 1+tan(x)**2)
    UN7 = MulDistr(N2, UniformDistr(9,11))
    UN8 = DivDistr(U1, UniformDistr(-2,1))
    UN9 = MulDistr(UniformDistr(9,11), N2)
    UN10 = FuncDistr(UniformDistr(3,5), f = arctan, f_inv = tan, f_inv_deriv = lambda x: 1+tan(x)**2)
    UN11 = DivDistr(UniformDistr(-2,1), UniformDistr(-2,1))

    ##! UI1 = InterpolatedDistr(UN2)
    ##! print N16.pdf(1e-17) # breakpoint problems occur here.  Should be about 0.3183
    #UI2 = InterpolatedDistr(N16)
    #print "UI2.err =", UI2.err

    #show_distr = histdistr
    show_distr = plotdistr

    #show_distr(N1)
    #show_distr(N2)
    #show_distr(UN2)
    #show_distr(N4)
    #show_distr(N5)
    #show_distr(N6)
    #show_distr(N7)
    #show_distr(N8)
    #show_distr(N9)
    #show_distr(N10);show_distr(N13) # Example: X**2 the same as X*X with our semantics
    #show_distr(N10);show_distr(N13prime) # Example: X**2 not the same as X*X.copy()
    #show_distr(N11)
    #show_distr(N14);show_distr(N15) # multiply by a constant
    #show_distr(U1)
    #show_distr(UN1)
    #show_distr(UN2)
    #show_distr(UN3);print UN3.breaks
    #show_distr(UN4)
    #show_distr(UN6)
    #show_distr(UN7)
    #show_distr(UN8, l=-6, u=6) # my favorite distr
    #show_distr(UN9)
    #show_distr(UN10);print UN10.breaks
    show_distr(UN11)
    #histdistr(UN11, l=4, u=4)
   # show_distr(UN1);print UN1.breaks;histdistr(UN1, l = -10, u = 10)
#    show_distr(UI1); print "U1.err =",UI1.err
    #show_distr(UI2);


    #xlim(-5,5)

    #print integrate(lambda x: UN1.PDF(x), limit=100)
    #print quad(lambda x: UN6.PDF(x), 0.2, 0.6, points=[], limit=100)
#    print integrate(lambda x: UN8.pdf(x), limit=100)
    #print quad(lambda x: UN8.PDF(x), -Inf, 0, limit=100)
    #print quad(lambda x: UN6.PDF(x), 0.52360, 0.6, points=[], limit=100)
    #print quad(lambda x: UN8.PDF(x), -10, 10, limit=100)
#    print integrate(lambda x: N16.pdf(x), limit=100)
#    print integrate(lambda x: UI2.pdf(x), limit=100)


    ## another example: http://www.physicsforums.com/showthread.php?t=75889
    ## probability density of two resistors in parallel XY/(X+Y)
    ##
    ## XY an X+Y are not independent: we are not yet ready to handle this
    #L = 40; U = 70
    #X = uniformDistr(100,120)
    #Y = uniformDistr(100,120)
    #N = mulDistr(X,Y)
    #D = sumDistr(X,Y)
    #Ni = interpolatedDistr(N)
    #Di = interpolatedDistr(D)
    #R = divDistr(Ni, Di)
    #Ri = interpolatedDistr(R)
    #print integrate(lambda x: Ri.PDF(x), limit=100)
    #show_distr(Ri, L, U);histdistr(R, l = L, u = U, n = 1000000)
    #xlim(L, U)
    

    
    show()
