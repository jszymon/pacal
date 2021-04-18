import numpy as np

from pacal.integration import *

def cauchy(x):
    return 1.0/(np.pi*(1.0+x*x))
def normpdf(x):
    return 1.0/np.sqrt(2*np.pi)*np.exp(-x*x/2)
def chisq1(x):
    return x**(-0.5)*np.exp(-x/2) / np.sqrt(2*np.pi)
def prodcauchy_uni(x):
    return 1.0/(2*np.pi)*np.log1p((1.0/(x*x)))
def beta(x):
    return x**-0.5*(1.0-x)**-0.5 / np.pi

def check_integ(method, f, a, b, exact, tol=1e-15):
    if a is not None and b is not None:
        i, e = method(f, a, b)
    elif a is not None:
        i, e = method(f, a)
    elif b is not None:
        i, e = method(f, b)
    else:
        i, e = method(f)
    assert abs(i - exact) <= tol

def test_clenshaw():
    check_integ(integrate_clenshaw, np.cos, 0, np.pi/2, 1)
    check_integ(integrate_clenshaw, np.cos, 0, 2, np.sin(2))
    check_integ(integrate_clenshaw, cauchy, 0, 350, 0.5, 1e-3)
def test_fejer2():
    check_integ(integrate_fejer2, np.cos, 0, np.pi/2, 1)
    check_integ(integrate_fejer2, np.cos, 0, 2, np.sin(2))
    check_integ(integrate_fejer2, np.cos, 0.1, 0.1, 0)
    check_integ(integrate_fejer2, np.cos, 0.1, 0.3/3, 0)
    check_integ(integrate_fejer2, np.cos, 2, 0, -np.sin(2))
def test_clenshaw_inf():
    check_integ(integrate_clenshaw_pminf, normpdf, None, None, 1)
    check_integ(integrate_clenshaw_pinf, normpdf, 0, None, 0.5)
    check_integ(integrate_clenshaw_minf, normpdf, None, 0, 0.5)
    # those don't work well with clenshaw because of endpoints:
    #check_integ(integrate_clenshaw_pminf, cauchy, None, None, 1)
    #check_integ(integrate_clenshaw_pinf, cauchy, 0, None, 0.5)
    #check_integ(integrate_clenshaw_minf, cauchy, None, 0, 0.5)
def test_fejer2_inf():
    check_integ(integrate_fejer2_pminf, normpdf, None, None, 1, 1e-5) # pminf not accurate
    check_integ(integrate_fejer2_pinf, normpdf, 0, None, 0.5)
    check_integ(integrate_fejer2_minf, normpdf, None, 0, 0.5)
    check_integ(integrate_fejer2_pminf, cauchy, None, None, 1.0, 1e-5) # pminf not accurate
    check_integ(integrate_fejer2_pinf, cauchy, 0, None, 0.5)
    check_integ(integrate_fejer2_minf, cauchy, None, 0, 0.5)
def test_integration_with_poles():
    check_integ(integrate_fejer2_Xn_transformP, np.log, 0, 1, -1)
    check_integ(integrate_fejer2_Xn_transformP, lambda x: np.log(x)/np.sqrt(x), 0, 1, -4)
    check_integ(integrate_fejer2_Xn_transformP, chisq1, 0, 100, 1)
    check_integ(integrate_fejer2_Xn_transform, prodcauchy_uni, 0, 200, 0.5, 1e-3)
    check_integ(integrate_fejer2_Xn_transformP, prodcauchy_uni, 0, 200, 0.5, 1e-3)
    check_integ(integrate_fejer2_Xn_transformN, prodcauchy_uni, -200, 0, 0.5, 1e-3)
    check_integ(integrate_fejer2_Xn_transformP, beta, 0, 0.5, 0.5)
    check_integ(integrate_fejer2_Xn_transformN, beta, 0.5, 1, 0.5, 1e-8) # pole at !=0

def test_vector_integration():
    a = np.linspace(0.1, 1, 100)
    def fv(x):
        return np.cos(np.outer(a, x))
    i, e = integrate_fejer2_vector(fv, 0, 1)
    exact = 1/a*np.sin(a*1)
    assert np.max(np.abs(i - exact)) < 1e-15
    
