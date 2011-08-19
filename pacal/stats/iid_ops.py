"""Operations on sequences of i.i.d. variables."""

import operator

import pacal.distr
from pacal.utils import binomial_coeff
from pacal.standard_distr import PDistr

def _int_exp(x, n, op = operator.mul):
    """Right to left exponentiation by squaring."""
    res = None
    for b in bin(n)[2:]:
        if res is not None:
            res = op(res, res)
        if b == "1":
            if res is not None:
                res = op(res, x)
            else:
                res = x
    return res
def _int_exp_all(x, n, op = operator.mul):
    """Exponentiation by squaring returning all powers."""
    res = [None] * (n+1)
    res[1] = x
    for i in xrange(2, n+1):
        res[i] = op(res[i // 2], res[i - i // 2])
    return res[1:]

def iid_op(X, n, op, all = False):
    if all:
        y = _int_exp_all(X, n, op)
    else:
        y = _int_exp(X, n, op)
    return y
def iid_sum(X, n, all = False):
    return iid_op(X, n, op = operator.add, all = all)
def iid_prod(X, n, all = False):
    return iid_op(X, n, op = operator.mul, all = all)
def iid_min(X, n, all = False):
    return iid_op(X, n, op = pacal.distr.min, all = all)
def iid_max(X, n, all = False):
    return iid_op(X, n, op = pacal.distr.max, all = all)

def iid_order_stat(X, n, k):
    pdf = X.get_piecewise_pdf()
    cdf = X.get_piecewise_cdf()
    ccdf = 1 - X.get_piecewise_cdf()
    fun = k * binomial_coeff(n, k) * pow(cdf, k-1) * pow(ccdf, n-k) * pdf
    return PDistr(fun.toInterpolated())
def iid_median(X, n):
    return iid_order_stat(X, n, n // 2)

# averages need special treatment
def _int_exp2(x, n, op = operator.mul):
    """Right to left exponentiation by squaring.

    Extended operator arguments."""
    res = None
    for i, b in enumerate(bin(n)[2:]):
        if res is not None:
            res = op(i, res, i, res)
        if b == "1":
            if res is not None:
                res = op(i, res, 1, x)
            else:
                res = x
    return res
def _int_exp_all2(x, n, op = operator.mul):
    """Exponentiation by squaring returning all powers.

    Extended operator arguments."""
    res = [None] * (n+1)
    res[1] = x
    for i in xrange(2, n+1):
        i1 = i // 2
        i2 = i - i // 2
        res[i] = op(i1, res[i1], i2, res[i2])
    return res[1:]
def iid_op2(X, n, op, all = False):
    if all:
        y = _int_exp_all2(X, n, op)
    else:
        y = _int_exp2(X, n, op)
    return y
def _lambda_average(n1, x1, n2, x2):
    n = float(n1 + n2)
    return (n1 * x1 + n2 * x2) / n
def iid_average(X, n, all = False):
    return iid_op2(X, n, op = _lambda_average, all = all)


if __name__ == "__main__":
    #print iid_sum(3, 10)
    #print iid_sum(3, 10, all = True)
    #print iid_max(3, 10, all = True)
    #print iid_average(3, 10)
    #print iid_average(3, 10, all = True)

    from pacal import *
    from pylab import *
    figure()
    for i in [1,51]:
        fun = iid_order_stat(BetaDistr(0.5,1.5), 51, i)
        fun.plot(xmin=0,xmax=1)
        fun.summary()
    show()
