from __future__ import print_function

from numpy import sqrt

from pacal.standard_distr import NormalDistr, ChiSquareDistr, ZeroDistr
from pacal.distr import Distr, SumDistr, DivDistr, InvDistr
from pacal.distr import sqrt as distr_sqrt

class NoncentralTDistr(DivDistr):
    def __init__(self, df = 2, mu = 0):
        d1 = NormalDistr(mu, 1)
        d2 = distr_sqrt(ChiSquareDistr(df) / df)
        super(NoncentralTDistr, self).__init__(d1, d2)
        self.df = df
        self.mu = mu
    def __str__(self):
        return "NoncentralTDistr(df={0},mu={1})#{2}".format(self.df, self.mu, self.id())
    def getName(self):
        return "NoncT({0},{1})".format(self.df, self.mu)

class NoncentralChiSquareDistr(SumDistr):
    def __init__(self, df, lmbda = 0):
        assert df >= 1
        d1 = NormalDistr(sqrt(lmbda))**2
        if df > 1:
            d2 = ChiSquareDistr(df - 1)
        else:
            d2 = ZeroDistr()
        super(NoncentralChiSquareDistr, self).__init__(d1, d2)
        self.df = df
        self.lmbda = lmbda
    def __str__(self):
        return "NoncentralChiSquare(df={0},lambda={1})#{2}".format(self.df, self.lmbda, self.id())
    def getName(self):
        return "NoncChi2({0},{1})".format(self.df, self.lmbda)

class NoncentralBetaDistr(InvDistr):
    def __init__(self, alpha = 1, beta = 1, lmbda = 0):
        d = 1 + ChiSquareDistr(2.0 * beta) / NoncentralChiSquareDistr(2 * alpha, lmbda)
        super(NoncentralBetaDistr, self).__init__(d)
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda
    def __str__(self):
        return "NoncentralBetaDistr(alpha={0},beta={1},lambda={2})#{3}".format(self.alpha, self.beta, self.lmbda, self.id())
    def getName(self):
        return "NoncBeta({0},{1},{2})".format(self.alpha, self.beta, self.lmbda)

class NoncentralFDistr(DivDistr):
    def __init__(self, df1 = 1, df2 = 1, lmbda = 0):
        d1 = NoncentralChiSquareDistr(df1, lmbda) / df1
        d2 = ChiSquareDistr(df2) / df2
        super(NoncentralFDistr, self).__init__(d1, d2)
        self.df1 = df1
        self.df2 = df2
        self.lmbda = lmbda
    def __str__(self):
        return "NoncentralFDistr(df1={0},df2={1},lambda={2})#{3}".format(self.df1, self.df2, self.lmbda, self.id())
    def getName(self):
        return "NoncF({0},{1},{2})".format(self.df1, self.df2, self.lmbda)
