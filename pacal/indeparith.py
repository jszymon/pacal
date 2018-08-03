"""Probabilistic arithmetic on independent random variables."""

from __future__ import print_function

import itertools
from functools import partial
import bisect
import operator

from numpy import asfarray
from numpy import multiply, add, divide
from numpy import unique, isnan, isscalar, size, flatnonzero
from numpy import sign, isinf, isfinite
from numpy import minimum, maximum, pi

from . import params

from .integration import *
from .interpolation import *
from .segments import DiracSegment
from .segments import Segment, SegmentWithPole, PInfSegment, MInfSegment
from .segments import PiecewiseDistribution, _segint
from .utils import epsunique, testPole
from .utils import get_parmap

import multiprocessing
import time

def _testConvPole(seg, L, U, pole_eps = None):
    if pole_eps is None:
        pole_eps = params.pole_detection.min_distance_from_pole
    poleL = False
    if seg.hasLeftPole() and abs(seg.a - L) <= pole_eps:
        poleL = True
    poleR = False
    if seg.hasRightPole() and abs(seg.b - U) <= pole_eps:
        poleR = True
    return poleL, poleR

def _unique_breakpoints(breaks, eps = 4 * finfo(float).eps):
    """Make breakpoints list unique"""
    ubreaks = []
    breaks.sort()
    i = 0
    while i < len(breaks):
        ubreaks.append(breaks[i])
        i += 1
        while i < len(breaks) and abs(breaks[i][0] - ubreaks[-1][0]) <= eps:
            ubreaks[-1][1] |= breaks[i][1]
            ubreaks[-1][2] |= breaks[i][2]
            ubreaks[-1][3] |= breaks[i][3]
            ubreaks[-1][4] |= breaks[i][4]
            i += 1
    return ubreaks

def conv(f, g):
    """Probabilistic sum (convolution) of f and g
    """
    # create the list of result breakpoints
    # each element is a tuple with breakpoint possible pole flags for left and right side
    fbreaks = f.getBreaksExtended()
    gbreaks = g.getBreaksExtended()
    breaks = []
    #print [str(b) for b in gbreaks]
    has_minf = False
    has_pinf = False
    #print "conv:", len(fbreaks),len(gbreaks)
    for fbrk in fbreaks:
        if isinf(fbrk.x) and fbrk.x < 0:
            has_minf = True
        elif isinf(fbrk.x) and fbrk.x > 0:
            has_pinf = True
        else:
            for gbrk in gbreaks:
                if isinf(gbrk.x) and gbrk.x < 0:
                    has_minf = True
                elif isinf(gbrk.x) and gbrk.x > 0:
                    has_pinf = True
                elif gbrk.dirac and not fbrk.dirac:
                    newbreak = [fbrk.x + gbrk.x, fbrk.negPole, fbrk.posPole, fbrk.Cont, fbrk.Cont]
                    breaks.append(newbreak)
                elif fbrk.dirac and not gbrk.dirac:
                    newbreak = [fbrk.x + gbrk.x, gbrk.negPole, gbrk.posPole, gbrk.Cont, gbrk.Cont]
                    breaks.append(newbreak)
                else:
                    newbreak = [fbrk.x + gbrk.x, False, False, False, False]
                    if fbrk.negPole and gbrk.negPole:
                        newbreak[1] = True
                    if fbrk.posPole and gbrk.posPole:
                        newbreak[2] = True
                    if fbrk.negPole and gbrk.posPole or fbrk.posPole and gbrk.negPole:
                        newbreak[1] = newbreak[2] = True
                    if not gbrk.Cont:
                        if fbrk.negPole and fbrk.posPole:
                            newbreak[3] = newbreak[4] = True
                        if fbrk.negPole:
                            newbreak[3] = True
                        if fbrk.posPole:
                            newbreak[4] = True
                    if not fbrk.Cont:
                        if gbrk.negPole and gbrk.posPole:
                            newbreak[3] = newbreak[4] = True
                        if gbrk.negPole:
                            newbreak[3] = True
                        if gbrk.posPole:
                            newbreak[4] = True
                    breaks.append(newbreak)
    if has_minf:
        breaks = [[-Inf, False, False]] + breaks
    if has_pinf:
        breaks.append([Inf, False, False])
    breaks = _unique_breakpoints(breaks)

    fg = PiecewiseDistribution([])
    if len(breaks)>1 and isinf(breaks[0][0]):
        # TODO: integration parameters for infinite and asymp should
        # differ
        segList = _findSegListAdd(f, g, breaks[1][0] - 1)
        seg = MInfSegment(breaks[1][0], Convxrunner(segList, params.integration_infinite).convx)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        breaks = breaks[1:]
    if len(breaks)>1 and isinf(breaks[-1][0]):
        segList = _findSegListAdd(f, g, breaks[-2][0] + 1)
        seg = PInfSegment(breaks[-2][0], Convxrunner(segList, params.integration_infinite).convx)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        breaks = breaks[:-1]
    for i in range(len(breaks)-1):
        segList = _findSegListAdd(f, g, (breaks[i][0] + breaks[i+1][0])/2)
        fun = Convxrunner(segList, params.integration_finite).convx
        seg = Segment(breaks[i][0], breaks[i+1][0], fun)
        left_pole = False; NoL = False; right_pole = False; NoR = False
        # potential singularities
        if breaks[i][2]: # potential singularity on the left
            NoL = True
            left_pole = testPole(fun, breaks[i][0])
            if left_pole and params.segments.debug_info:
                print("probably pole", estimateDegreeOfPole(fun, breaks[i][0]))
            elif params.segments.debug_info:
                print("probably no pole", estimateDegreeOfPole(fun, breaks[i][0]))
        if breaks[i][4]: # potential singularity on the left
            NoL = True
            left_pole = testPole(fun, breaks[i][0], deriv = True)
            if left_pole and params.segments.debug_info:
                print("probably deriv pole", estimateDegreeOfPole(fun, breaks[i][0], deriv = True))
            elif params.segments.debug_info:
                print("probably no deriv pole", estimateDegreeOfPole(fun, breaks[i][0], deriv = True))
        if breaks[i+1][1]: # potential singularity on the right
            NoR = True
            right_pole = testPole(fun, breaks[i+1][0], pos = False)
            if right_pole and params.segments.debug_info:
                print("probably pole", estimateDegreeOfPole(fun, breaks[i+1][0], pos = False))
            elif params.segments.debug_info:
                print("probably no pole", estimateDegreeOfPole(fun, breaks[i+1][0], pos = False))
        if breaks[i+1][3]: # potential singularity on the right
            NoR = True
            right_pole = testPole(fun, breaks[i+1][0], pos = False, deriv = True)
            if right_pole and params.segments.debug_info:
                print("probably deriv pole", estimateDegreeOfPole(fun, breaks[i+1][0], pos = False, deriv = True))
            elif params.segments.debug_info:
                print("probably no deriv pole", estimateDegreeOfPole(fun, breaks[i+1][0], pos = False, deriv = True))
        segint = seg.toInterpolatedSegment(left_pole, NoL, right_pole, NoR)
        fg.addSegment(segint)
    # Discrete parts of distributions
    fg_discr = convdiracs(f, g, fun = operator.add)
    for seg in fg_discr.getDiracs():
        fg.addSegment(seg)

    return fg

# The actual convolution:

# Integrand for convolution along X axis
def fun_c1(segi, segj, x, t):
    return segi.f(t)*segj.f(x-t)
# Integrand for convolution along Y axis
def fun_c2(segi, segj, x, t):
    return segi.f(x-t)*segj.f(t)
# top wrapper

# Integrand for maximum along X axis
def funi_maxx(segi, segj, x, t):
    return segi.f(t)*segj.f(x)
# Integrand for maximum along Y axis
def funj_maxx(segi, segj, x, t):
    return segi.f(x)*segj.f(t)

# Integrand for minimum along X axis
def funi_minx(segi, segj, x, t):
    return segi.f(t)*segj.f(x)
# Integrand for minimum along Y axis
def funj_minx(segi, segj, x, t):
    return segi.f(x)*segj.f(t)

class Convxrunner(object):
    def __init__(self, segList, integration_par):
        self.segList = segList
        self.integration_par = integration_par
    def convx(self, xx):#segList, integration_par, p_map, xx):
        """convolution of f and g
        """
        if isscalar(xx):
            xx=asfarray([xx])
        #import traceback
        #print traceback.print_stack()


        #import pickle
        #try:
        #    pickle.dumps(segList[0][0])
        #except:
        #    import pdb; pdb.set_trace()
        #    pickle.dumps(segList[0][0])
        #except:
        #    print "Q", segList[0][0]
        #    import pdb; pdb.set_trace()
        #    pickle.dumps(segList[0][0].f.vb)
        #    #raise
        #import cPickle as pickle
        #for segi, segj in segList:
        #    print "A",segj.f.args
        #    #print "B",pickle.loads(pickle.dumps(segj.f)).f.args
        #    print pickle.dumps(segj.f)

        p_map = get_parmap()
        res = p_map(self.conv_at_point, xx)
        res = array(res)
        return res
    def convprodx(self, xx):
        """convolution of f and g
        """
        if isscalar(xx):
            xx=asfarray([xx])
        p_map = get_parmap()
        res = p_map(self.convprod_at_point, xx)
        res = array(res)
        return res
    def convdivx(self, xx):
        """convolution of f and g
        """
        if isscalar(xx):
            xx=asfarray([xx])
        p_map = get_parmap()
        res = p_map(self.convdiv_at_point, xx)
        res = array(res)
        return res
    def convmaxx(self, xx):
        """convolution of f and g
        """
        if isscalar(xx):
            xx=asfarray([xx])
        p_map = get_parmap()
        res = p_map(self.convmax_at_point, xx)
        res = array(res)
        return res
    def convminx(self, xx):
        """Probabilistic minimum of f and g, integral at points xx.
        """
        if size(xx)==1:
            xx=asfarray([xx])
        p_map = get_parmap()
        res = p_map(self.convminx_at_point, xx)
        res = array(res)
        return res
    def conv_at_point(self, x):
        #print x
        #print x, multiprocessing.current_process().name
        segList = self.segList
        integration_par = self.integration_par
        I = 0.0
        err = 0
        for segi, segj in segList:
            fun1 = partial(fun_c1, segi, segj, x)
            fun2 = partial(fun_c2, segi, segj, x)
            if segi.isSegment() and segj.isSegment():
                Lx = max(segi.safe_a, x-segj.safe_b)
                Ux = min(segi.safe_b, x-segj.safe_a)
                Ly = max(segj.safe_a, x-segi.safe_b)
                Uy = min(segj.safe_b, x-segi.safe_a)

                if not isinf(segi.a) and not isinf(segj.a) and not isinf(segi.b) and not isinf(segj.b):
                    # both segments finite, check poles
                    poleLi, poleRi = _testConvPole(segi, Lx, Ux)
                    poleLj, poleRj = _testConvPole(segj, Ly, Uy)
                    poleL = poleLi or poleRj
                    poleU = poleRi or poleLj
                    if poleL and poleU:
                        Mx = (Lx + Ux)/2
                        My = (Ly + Uy)/2
                        if x > 0:
                            i1, e1 = _segint(fun1, Lx, Mx, force_poleL = True)
                            i2, e2 = _segint(fun2, Ly, My, force_poleL = True)
                        else:
                            i1, e1 = _segint(fun1, Mx, Ux, force_poleU = True)
                            i2, e2 = _segint(fun2, My, Uy, force_poleU = True)
                        i = i1 + i2
                        e = e1 + e2
                    elif poleLi or poleRi:
                        i, e = _segint(fun1, Lx, Ux, force_poleL = poleLi, force_poleU = poleRi)
                        #if isinf(i):
                        #    import pdb; pdb.set_trace()
                        #    i, e = _segint(fun1, Lx, Ux, force_poleL = poleLi, force_poleU = poleRi)
                    elif poleLj or poleRj:
                        i, e = _segint(fun2, Ly, Uy, force_poleL = poleLj, force_poleU = poleRj)
                    else:
                        # no poles just integrate over x
                        i, e = _segint(fun1, Lx, Ux)
                elif (isinf(segj.a) or isinf(segj.b)) and not isinf(segi.a) and not isinf(segi.b):
                    # integrate over x (the finite segment is segi)
                    poleL, poleR = _testConvPole(segi, Lx, Ux)
                    i, e = _segint(fun1, Lx, Ux, force_poleL = poleL, force_poleU = poleR)
                elif (isinf(segi.a) or isinf(segi.b)) and not isinf(segj.a) and not isinf(segj.b):
                    # integrate over y (the finite segment is segj)
                    poleL, poleR = _testConvPole(segj, Ly, Uy)
                    i, e = _segint(fun2, Ly, Uy, force_poleL = poleL, force_poleU = poleR)
                elif isinf(segi.a) and isinf(segj.b):
                    # infinite path: upper left quadrant
                    # integrate over the variable which starts at smaller value
                    if abs(Ux) < abs(Ly):
                        i, e = integrate_fejer2_minf(fun1, Ux)
                    else:
                        i, e = integrate_fejer2_pinf(fun2, Ly)
                elif isinf(segi.b) and isinf(segj.a):
                    # infinite path: lower right quadrant
                    # integrate over the variable which starts at smaller value
                    if abs(Lx) < abs(Uy):
                        i, e = integrate_fejer2_pinf(fun1, Lx)
                    else:
                        i, e = integrate_fejer2_minf(fun2, Uy)
                elif ((isinf(segi.a) and isinf(segj.a)) or (isinf(segi.b) and isinf(segj.b))):
                    # long but finite path across infinite segments
                    # need to split such that both integrations begin at small values
                    debug_plot = False
                    debug_info = False
                    #if 1e14 < x < 1e15:
                    #    debug_plot = True
                    #    debug_info = True
                    Mx = (Lx + Ux) / 2
                    My = (Ly + Uy) / 2
                    if isinf(segi.a) and isinf(segj.a):
                        ix, ex = integrate_with_pminf_guess(fun1, Mx, Ux, debug_plot = debug_plot, debug_info = debug_info)
                        iy, ey = integrate_with_pminf_guess(fun2, My, Uy, debug_plot = debug_plot, debug_info = debug_info)
                    else:
                        ix, ex = integrate_with_pminf_guess(fun1, Lx, Mx, debug_plot = debug_plot, debug_info = debug_info)
                        iy, ey = integrate_with_pminf_guess(fun2, Ly, My, debug_plot = debug_plot, debug_info = debug_info)
                    i = ix + iy
                    e = ex + ey
                    #if 1e14 < x < 1e15:
                    #    print x, segi, segj, i, log(i) - log(x**-1.5)
                    #    print repr(Lx), repr(Mx), repr(Ly), repr(My)
                    #    print fun1(array([Lx]))[0], fun1(array([Mx]))[0]
                    #    print fun2(array([Ly]))[0], fun2(array([My]))[0]
                    #    print ix, iy, log(ix) - log(x**-1.5), log(iy) - log(x**-1.5)
                    #    print
                else:
                    print("Should not be here!!!")
                    assert(False)
            #if 3.26 < log(-x+1) < 3.27:
            #    print "x", x, log(-x+1), segi, segj, repr(Lx), repr(Ux), ix
            #    print "y", x, log(-x+1), segi, segj, repr(Ly), repr(Uy), iy
            elif segi.isDirac() and segj.isSegment():
                i = segi.f*segj.f(x-segi.a)
                e=0
            elif segi.isSegment() and segj.isDirac():
                i = segj.f*segi.f(x-segj.a)
                e=0
            elif segi.isDirac() and segj.isDirac():
                i = 0
                e = 0
            I += i
            err += e
        return I

    def convminx_at_point(self, x):
        segList = self.segList
        integration_par = self.integration_par
        I = 0.0
        err = 0
        for segi, segj in segList:
            if segj.a <= x <= segj.b and segj.a != segj.b:
                if segi.isSegment() and segj.isSegment():
                    funi = partial(funi_minx, segi, segj, x)
                    L = max(segi.a,x)
                    U = segi.b
                    i, e = _segint(funi, L, U)
                elif segi.isDirac() and segj.isSegment():
                    i = segi.f*segj.f(x)   # TODO
                    e=0
                elif segi.isSegment() and segj.isDirac():
                    i = segj.f*segi.f(x)   # TODO
                    e=0
                elif segi.isDirac() and segj.isDirac():
                    pass #Dicrete part is done in convmin
                I += i
                err += e
            if segi.a <= x <= segi.b and segi.a != segi.b :
                if segi.isSegment() and segj.isSegment():
                    funj = partial(funj_minx, segi, segj, x)
                    L = max(segj.a, x)
                    U = segj.b
                    i, e = _segint(funj, L, U)
                elif segi.isDirac() and segj.isSegment():
                    i = segi.f*segj.f(x)
                    e=0
                elif segi.isSegment() and segj.isDirac():
                    i = segj.f*segi.f(x)
                    e=0
                elif segi.isDirac() and segj.isDirac():
                    pass #Dicrete part is done in convmin
                I += i
                err += e
        return I
    def convmax_at_point(self, x):
        """Probabilistic maximum of f and g, integral at point x
        """
        segList = self.segList
        integration_par = self.integration_par
        I = 0.0
        err = 0
        for segi, segj in segList:
            i, e = 0, 0
            U, L = 0, 0
            if segj.a <= x <= segj.b and segj.a != segj.b:
                if segi.isSegment() and segj.isSegment():
                    L = segi.a
                    U = min(segi.b,x)
                    funi = partial(funi_maxx, segi, segj, x)
                    i, e = _segint(funi, L, U)
                elif segi.isDirac() and segj.isSegment():
                    i = segi.f*segj.f(x)
                    e=0
                elif segi.isSegment() and segj.isDirac():
                    i = segj.f*segi.f(x)
                    e=0
                elif segi.isDirac() and segj.isDirac():
                    pass #Dicrete part is done in convmax
                I += i
                err += e
            if segi.a <= x <= segi.b and segi.a != segi.b:
                if segi.isSegment() and segj.isSegment():
                    L = segj.a
                    U = min(segj.b,x)
                    funj = partial(funj_maxx, segi, segj, x)
                    i, e = _segint(funj, L, U)
                elif segi.isDirac() and segj.isSegment():
                    i = segi.f*segj.f(x)
                    e=0
                elif segi.isSegment() and segj.isDirac():
                    i = segj.f*segi.f(x)
                    e=0
                elif segi.isDirac() and segj.isDirac():
                    pass #Dicrete part is done in convmax
                I += i
                err += e
        return I

    def convprod_at_point(self, x):
        """Probabilistic product (Melin's convolution), integral at points xx
        """
        segList = self.segList
        integration_par = self.integration_par
        # Integrand for division along X axis
        def funi(t):
            y = segi.f(t)*segj.f(x/t)*1.0/abs(t)
            y[t==0] = 0.0
            return y
        # Integrand for division along Y axis
        def funj(t):
            y = segi.f(x/t)*segj.f(t)*1.0/abs(t)
            y[t==0] = 0.0
            return y
        I = 0.0
        err = 0.0
        for segi, segj in segList:
            if segi.isSegment() and segj.isSegment():
                cwiartka = (segi.a + segi.b) * (segj.a + segj.b)
                i1=0
                forcepoleL = False
                forcepoleU = False
                if cwiartka >0 :
                    if x>0:
                        L = max(segi.a, sign(segi.a+segi.b)*x/abs(segj.b))
                        U = min(segi.b, sign(segi.a+segi.b)*x/abs(segj.a))
                    else:
                        L = segi.a
                        U = segi.b
                        i1, e1 = _segint(funj, segj.a, segj.b)
                else:
                    if x<0:
                        L = max(segi.a, -sign(segi.a+segi.b)*x/abs(segj.a))
                        U = min(segi.b, -sign(segi.a+segi.b)*x/abs(segj.b))
                    else: # TODO: fix needed when x=0 and second argument contains 0 - FIXED
                        L = segi.a
                        U = segi.b
                        i1, e1 = _segint(funj, segj.a, segj.b)
                i, e = _segint(funi, L, U, force_poleL = forcepoleL, force_poleU = forcepoleU )
                i=i+i1
            elif segi.isDirac() and segj.isSegment():
                i = segi.f*segj.f(x/segi.a)*1.0/abs(segi.a)   # TODO
                if x== 0.0 :
                    i = i + segi.f
                e = 0.0
            elif segi.isSegment() and segj.isDirac():
                i = segj.f*segi.f(x/segj.a)*1.0/abs(segj.a)
                if x== 0.0 :
                    i = i + segj.f
                e = 0.0
            I += i
            err += e
        return I

    def fun_div(self, x):
        segList = self.segList
        integration_par = self.integration_par
        if isscalar(x):
            if abs(x) < 1:
                fun = Convxrunner(segList, params.integration_finite).convprodx

                y = self.convdiv_at_point(x)
            else:
                y = self.convdiv_at_point(x)
        else:
            y = empty_like(x)
            mask  = (abs(x) < 1)
            p_map = get_parmap()
            y[mask]  = p_map(self.convdiv_at_point, x[mask]) # convdivx(segList, x[mask])
            y[~mask] = p_map(self.convdiv2_at_point, x[~mask]) # convdivx2(segList, x[~mask])
        return y
    def convdiv_at_point(self, x):
        """Probabilistic division of piecewise functions f and g,
        integral at points xx, along X axis
        """
        segList = self.segList
        integration_par = self.integration_par
        def fun(t):
            y = segi.f(x*t)*segj.f(t)*abs(t)
            return y
        I = 0.0
        err = 0.0
        for segi, segj in segList :
            if segi.isSegment() and segj.isSegment():
                if x == 0:
                    L = segj.a
                    U = segj.b
                else:
                    L = segi.a / x
                    U = segi.b / x
                    L, U = min(L, U), max(L, U)
                    L = max(segj.a, L)
                    U = min(segj.b, U)

                # integrate with variable transform even for finite intervals
                # if the integration domain is very wide
                force_minf = False
                force_pinf = False
                if not isinf(L) and L < 0 and (isinf(segi.a) or isinf(segj.a)):
                    force_minf = True
                if not isinf(U) and U > 0 and (isinf(segi.b) or isinf(segj.b)):
                    force_pinf = True
                i, e = _segint(fun, L, U, force_minf, force_pinf)

            elif segi.isDirac() and segj.isSegment():
                i = segi.f * segj.f(segi.a/x) * abs(segi.a)/x/x
                e = 0.0
                assert (segi.a != 0.0)
            elif segi.isSegment() and segj.isDirac():
                i = segj.f*segi.f(segj.a)*abs(segj.a)
                e = 0.0
                assert (segj.a != 0.0)
            I += i
            err += e
        return I

    def convdiv2_at_point(self, x):
        """Probabilistic division of piecewise functions f and g,
        integral at points xx, along Y axis
        """
        segList = self.segList
        integration_par = self.integration_par
        def fun(t):
            if x == 0:
                y = segi.f(zeros_like(t)) * segj.f(t) * abs(t)
            else:
                y = segi.f(t) * segj.f(t/x) * abs(t)/x/x
            return y
        I = 0.0
        err = 0.0
        for segi, segj in segList:
            if segi.isSegment() and segj.isSegment():
                if x == 0:
                    L = segj.a
                    U = segj.b
                else:
                    L = segj.a * x
                    U = segj.b * x
                    L, U = min(L, U), max(L, U)
                    L = max(segi.a, L)
                    U = min(segi.b, U)

                # integrate with variable transform even for finite intervals
                # if the integration domain is very wide
                force_minf = False
                force_pinf = False
                if not isinf(L) and L < 0 and (isinf(segi.a) or isinf(segj.a)):
                    force_minf = True
                if not isinf(U) and U > 0 and (isinf(segi.b) or isinf(segj.b)):
                    force_pinf = True
                i, e = _segint(fun, L, U, force_minf, force_pinf)
            elif segi.isDirac() and segj.isSegment():
                assert (segi.a != 0.0)
                i = segi.f * segj.f(segi.a/x) * abs(segi.a) / x / x
                e = 0

            elif segi.isSegment() and segj.isDirac():
                assert (segj.a != 0.0)
                i = segj.f * segi.f(segj.a) * abs(segj.a)
                e = 0
            I += i
            err += e
        return I

def epseq(a,b):
    if abs(a-b)<1e-10:
        return True
    else:
        return False
#def convmean(f, g, p = 0.5, q = 0.5):
#    """Probabilistic weighted mean of f and g
#    """
#    if  p + q != 1.0:
#        p1 = abs(p)/(abs(p) + abs(q))
#        q = abs(q)/(abs(p) + abs(q))
#        p=p1
#    if q == 0:
#        return f
#    bf = f.getBreaks()
#    bg = g.getBreaks()
#    b = add.outer(bf * p, bg * q)
#    fun = partial(convmeanx, segList, p, q)
#    ub = epsunique(b)
#    fg = PiecewiseDistribution([])
#    op = lambda x,y : p*x + q*y
#    if isinf(ub[0]):
#        segList = _findSegList(f, g, ub[1] -1, op)
#        seg = MInfSegment(ub[1], fun)
#        segint = seg.toInterpolatedSegment()
#        fg.addSegment(segint)
#        ub=ub[1:]
#    if isinf(ub[-1]):
#        segList = _findSegList(f, g, ub[-2] + 1, op)
#        seg = PInfSegment(ub[-2], fun)
#        segint = seg.toInterpolatedSegment()
#        fg.addSegment(segint)
#        ub=ub[0:-1]
#    for i in range(len(ub)-1) :
#        segList = _findSegList(f, g, (ub[i] + ub[i+1])/2, op)
#        seg = Segment(ub[i],ub[i+1], fun)
#        segint = seg.toInterpolatedSegment()
#        fg.addSegment(segint)
#
#    # Discrete parts of distributions
#    fg_discr = convdiracs(f, g, fun = lambda x,y : x * p + y * q)
#    for seg in fg_discr.getDiracs():
#        fg.addSegment(seg)
#    return fg

#def fun_cm(x, p, q, t):
#    return segi( t / p) * segj((x - t)/q)/p/q
#def convmeanx(segList, p, q, xx):
#    """Probabilistic weighted mean of f and g, integral at points xx
#    """
#    if size(xx)==1:
#        xx=asfarray([xx])
#    wyn = zeros_like(xx)
#    for j in range(len(xx)):
#        x = xx[j]
#        fun = partial(fun_cm, x, p, q)
#        I = 0
#        err = 0
#        for segi, segj in segList:
#            if segi.isSegment() and segj.isSegment():
#                L = max(segi.a*p, (x - segj.b * q))
#                U = min(segi.b*p, (x - segj.a * q))
#                i, e = _segint(fun, L, U)
#            elif segi.isDirac() and segj.isSegment():
#                i = segi.f*segj((x-segi.a)/q)/q   # TODO
#                e=0
#            elif segi.isSegment() and segj.isDirac():
#                i = segj.f*segi((x-segj.a)/p)/p   # TODO
#                e=0
#            elif segi.isDirac() and segj.isDirac():
#                pass
#                #i = segi(x-segj.a)/p/q          # TODO
#                #e=0;
#            I += i
#            err += e
#        wyn[j]=I
#    return wyn

def convmin(f, g): #TODO : NOW  segments of f and g should have the same breakpoint !!!
    """Probabilistic minimum of f and g.
    """
    bf = f.getBreaks()
    bg = g.getBreaks()
    f = f.splitByPoints(bg)
    g = g.splitByPoints(bf)
    b = minimum.outer(bf, bg)
    ub = unique(b)
    fg = PiecewiseDistribution([]);
    op = minimum
    if ub[0] == -Inf:
        segList = _findSegList(f, g, ub[1] -1, op)
        #fun = partial(convminx, segList)
        fun = Convxrunner(segList, params.integration_infinite).convminx
        seg = MInfSegment(ub[1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[1:]
    if ub[-1] == Inf :
        segList = _findSegList(f, g, ub[-2] + 1, op)
        #fun = partial(convminx, segList)
        fun = Convxrunner(segList, params.integration_infinite).convminx
        seg = PInfSegment(ub[-2], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[0:-1]
    for i in range(len(ub)-1) :
        segList = _findSegList(f, g, (ub[i] + ub[i+1])/2, op)
        #fun = partial(convminx, segList)
        fun = Convxrunner(segList, params.integration_finite).convminx
        seg = Segment(ub[i],ub[i+1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
    fg_discr = convdiracs(f, g, fun = op)

    # Discrete parts of distribution
    fg_discr  =_probDiracsInMin(f, g)
    fg.add_diracs(fg_discr)
    return fg
#def funi_minx(segi, segj, x, t):
#    return segi.f(t)*segj.f(x)
#def funj_minx(segi, segj, x, t):
#    return segi.f(x)*segj.f(t)
#def convminx(segList, xx):
#    """Probabilistic minimum of f and g, integral at points xx.
#    """
#    if size(xx)==1:
#        xx=asfarray([xx])
#    p_map = get_parmap()
#    res = p_map(partial(convminx_at_point, segList), xx)
#    res = array(res)
#    return res
#def convminx_at_point(segList, x):
#    I = 0
#    err = 0
#    for segi, segj in segList:
#        if segj.a <= x <= segj.b and segj.a != segj.b:
#            if segi.isSegment() and segj.isSegment():
#                funi = partial(funi_minx, segi, segj, x)
#                L = max(segi.a,x)
#                U = segi.b
#                i, e = _segint(funi, L, U)
#            elif segi.isDirac() and segj.isSegment():
#                i = segi.f*segj.f(x)   # TODO
#                e=0
#            elif segi.isSegment() and segj.isDirac():
#                i = segj.f*segi.f(x)   # TODO
#                e=0
#            elif segi.isDirac() and segj.isDirac():
#                pass #Dicrete part is done in convmin
#            I += i
#            err += e
#        if segi.a <= x <= segi.b and segi.a != segi.b :
#            if segi.isSegment() and segj.isSegment():
#                funj = partial(funj_minx, segi, segj, x)
#                L = max(segj.a, x)
#                U = segj.b
#                i, e = _segint(funj, L, U)
#            elif segi.isDirac() and segj.isSegment():
#                i = segi.f*segj.f(x)
#                e=0
#            elif segi.isSegment() and segj.isDirac():
#                i = segj.f*segi.f(x)
#                e=0
#            elif segi.isDirac() and segj.isDirac():
#                pass #Dicrete part is done in convmin
#            I += i
#            err += e
#    return I

def convmax(f, g):
    """Probabilistic maximum of f and g.
    """
    bf = f.getBreaks()
    bg = g.getBreaks()
    f= f.splitByPoints(bg)
    g= g.splitByPoints(bf)
    b = maximum.outer(bf, bg)
    ub = unique(b)
    fg = PiecewiseDistribution([])
    op = maximum
    if isinf(ub[0]):
        segList = _findSegList(f, g, ub[1] -1, op)
        #fun = partial(convmaxx, segList)
        fun = Convxrunner(segList, params.integration_infinite).convmaxx
        seg = MInfSegment(ub[1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[1:]
    if isinf(ub[-1]):
        segList = _findSegList(f, g, ub[-2] + 1.0, op)
        #fun = partial(convmaxx, segList)
        fun = Convxrunner(segList, params.integration_infinite).convmaxx
        seg = PInfSegment(ub[-2], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[0:-1]
    for i in range(len(ub)-1) :
        segList = _findSegList(f, g, (ub[i] + ub[i+1])/2.0, op)
        #fun = partial(convmaxx, segList)
        fun = Convxrunner(segList, params.integration_finite).convmaxx
        seg = Segment(ub[i],ub[i+1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
    # Discrete parts of distribution
    fg_discr  =_probDiracsInMax(f, g)
    fg.add_diracs(fg_discr)
    return fg


#def convmaxx(segList, xx):
#    """Probabilistic maximum of f and g, integral at points xx
#    """
#    if size(xx)==1:
#        xx=asfarray([xx])
#    wyn = zeros_like(xx)
#    funi = lambda t : segi.f(t)*segj.f(x)
#    funj = lambda t : segi.f(x)*segj.f(t)
#    for j in range(len(xx)) :
#        x = xx[j]
#        I = 0
#        err = 0
#        for segi, segj in segList:
#            i, e = 0, 0
#            U, L = 0, 0
#            if segj.a <= x <= segj.b and segj.a != segj.b:
#                if segi.isSegment() and segj.isSegment():
#                    L = segi.a
#                    U = min(segi.b,x)
#                    i, e = _segint(funi, L, U)
#                elif segi.isDirac() and segj.isSegment():
#                    i = segi.f*segj.f(x)
#                    e=0
#                elif segi.isSegment() and segj.isDirac():
#                    i = segj.f*segi.f(x)
#                    e=0
#                elif segi.isDirac() and segj.isDirac():
#                    pass #Dicrete part is done in convmax
#                I += i
#                err += e
#            if segi.a <= x <= segi.b and segi.a != segi.b:
#                if segi.isSegment() and segj.isSegment():
#                    L = segj.a
#                    U = min(segj.b,x)
#                    i, e = _segint(funj, L, U)
#                elif segi.isDirac() and segj.isSegment():
#                    i = segi.f*segj.f(x)
#                    e=0
#                elif segi.isSegment() and segj.isDirac():
#                    i = segj.f*segi.f(x)
#                    e=0
#                elif segi.isDirac() and segj.isDirac():
#                    pass #Dicrete part is done in convmax
#                I += i
#                err += e
#        wyn[j]=I
#    return wyn
def _split_for_prod_and_div(fun):
    breaks = fun.getBreaks()
    if (breaks[0]>0 or breaks[-1]<0):
        return fun
    breakslist = breaks.tolist()
    ind = flatnonzero(breaks == 0.0)
    if len(ind)==0:
        bisect.insort_left(breakslist, 0.0)
    ind = flatnonzero(array(breakslist) == 0.0)
    if len(ind)>0 and ind[0]==len(breakslist)-2:
        if isinf(breakslist[ind[0]+1]):
            bisect.insort_left(breakslist, -1.0)
        else:
            bisect.insort_left(breakslist, (breakslist[ind[0]+1])/2.0)
    if len(ind)>0 and ind[0]==1:
        if isinf(breakslist[ind[0]-1]):
            bisect.insort_left(breakslist, -1.0)
        else:
            bisect.insort_left(breakslist, (breakslist[ind[0]-1])/2.0)
    return fun.splitByPoints(unique(breakslist))

def convprod(f, g):
    """Probabilistic product (Melin's convolution) of piecewise
    functions f and g.
    """
    f = f.splitByPoints([-1, 0, 1])
    g = g.splitByPoints([-1, 0, 1])
    #f = _split_for_prod_and_div(f)
    #g = _split_for_prod_and_div(g)
    # For this case allway exists singularity at zero
    if f(0.0)>0.0 and g(0.0)>0.0:
        pole_at_zero = True
    else:
        pole_at_zero= False
    bf = f.getBreaks()
    bg = g.getBreaks()
    b = multiply.outer(bf, bg)
    op = operator.mul
    ub = epsunique(b)
    #ind = flatnonzero(ub == 0.0)
    #ublist = ub.tolist()
    #print ind[0], ub[ind[0]-1.0]
    #if len(ind)>0 and ind[0]<len(ub)-1:
    #    if isinf(ub[ind[0]+1.0]):
    #        bisect.insort_left(ublist, -1.0)
    #    else:
    #        bisect.insort_left(ublist, (ub[ind[0]+1.0])/2.0)
    #if len(ind)>0 and ind[0]>0:
    #    if isinf(ub[ind[0]-1.0]):
    #        bisect.insort_left(ublist, -1.0)
    #    else:
    #        bisect.insort_left(ublist, (ub[ind[0]-1.0])/2.0)
    #if len(ind) == 0 and ub[0]*ub[-1]<0:
    #    bisect.insort_left(ublist, 0)
    #ub = unique(array(ublist))
    #print "===", ub
    fg = PiecewiseDistribution([])
    if isinf(ub[0]):
        segList = _findSegList(f, g, ub[1] -1, op)
        #fun = partial(convprodx, segList)
        fun = Convxrunner(segList, params.integration_infinite).convprodx
        seg = MInfSegment(ub[1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[1:]
    if isinf(ub[-1]):
        segList = _findSegList(f, g, ub[-2] + 1, op)
        #fun = partial(convprodx, segList)
        fun = Convxrunner(segList, params.integration_infinite).convprodx
        seg = PInfSegment(ub[-2], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[0:-1]
    for i in range(len(ub)-1) :
        segList = _findSegList(f, g, (ub[i] + ub[i+1])/2, op)
        #fun = partial(convprodx, segList)
        fun = Convxrunner(segList, params.integration_finite).convprodx
        _NoL = False
        _NoR = False
        if (ub[i] == 0):
            if pole_at_zero or testPole(fun, ub[i], pos = True):
                #segint = InterpolatedSegmentWithPole(ub[i],ub[i+1], fun, left_pole = True)
                if params.segments.debug_info:
                    print("probably pole at 0 left prod", fun(0.0))
                seg = SegmentWithPole(ub[i],ub[i+1], fun, left_pole = True)
            else:
                if params.segments.debug_info:
                    print("probably no pole at 0 left prod", fun(0.0))
                seg = Segment(ub[i],ub[i+1], fun)
                _NoL = True
        elif (ub[i+1] == 0): # TODO add proper condition
            #segint = InterpolatedSegmentWithPole(ub[i],ub[i+1], fun, left_pole = False)
            if pole_at_zero or testPole(fun, ub[i+1], pos = False):
                if params.segments.debug_info:
                    print("probably pole at 0 right prod", fun(0.0))
                seg = SegmentWithPole(ub[i],ub[i+1], fun, left_pole = False)
            else:
                if params.segments.debug_info:
                    print("probably no pole at 0 right prod", fun(0.0))
                seg = Segment(ub[i],ub[i+1], fun)
                _NoR = True
        else:
            if params.segments.debug_info:
                print("probably no pole prod")
            seg = Segment(ub[i],ub[i+1], fun)
        segint = seg.toInterpolatedSegment(NoL = _NoL, NoR = _NoR)
        #segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)

    # Discrete parts of distributions
    fg_discr = convdiracs(f, g, fun = operator.mul)
    for seg in fg_discr.getDiracs():
        fg.addSegment(seg)
    seg0 = fg. findSegment(0.0)
    dirac_at_zero = _probDiracAtZeroInProd(f, g)
    if seg0 is not None and seg0.isDirac() and seg0.a == 0.0 and dirac_at_zero is not None:
        seg0.f = seg0.f + dirac_at_zero.f
    elif dirac_at_zero is not None:
        fg.addSegment(dirac_at_zero)
    return fg

#def convprodx(segList, xx):
#    """Probabilistic product (Melin's convolution), integral at points xx
#    """
#    if size(xx)==1:
#        xx=asfarray([xx])
#    wyn = zeros_like(xx)
#    for j in range(len(xx)) :
#        x = xx[j]
#        # Integrand for division along X axis
#        def funi(t):
#            y = segi.f(t)*segj.f(x/t)*1.0/abs(t)
#            y[t==0] = 0
#            return y
#        # Integrand for division along Y axis
#        def funj(t):
#            y = segi.f(x/t)*segj.f(t)*1.0/abs(t)
#            y[t==0] = 0
#            return y
#        I = 0
#        err = 0
#        for segi, segj in segList:
#            if segi.isSegment() and segj.isSegment():
#                cwiartka = (segi.a + segi.b) * (segj.a + segj.b)
#                i1=0
#                forcepoleL = False
#                forcepoleU = False
#                if cwiartka >0 :
#                    if x>0:
#                        L = max(segi.a, sign(segi.a+segi.b)*x/abs(segj.b))
#                        U = min(segi.b, sign(segi.a+segi.b)*x/abs(segj.a))
#                    else:
#                        L = segi.a
#                        U = segi.b
#                        i1, e1 = _segint(funj, segj.a, segj.b)
#                else:
#                    if x<0:
#                        L = max(segi.a, -sign(segi.a+segi.b)*x/abs(segj.a))
#                        U = min(segi.b, -sign(segi.a+segi.b)*x/abs(segj.b))
#                    else: # TODO: fix needed when x=0 and second argument contains 0 - FIXED
#                        L = segi.a
#                        U = segi.b
#                        i1, e1 = _segint(funj, segj.a, segj.b)
#                i, e = _segint(funi, L, U, force_poleL = forcepoleL, force_poleU = forcepoleU )
#                i=i+i1
#            elif segi.isDirac() and segj.isSegment():
#                i = segi.f*segj.f(x/segi.a)*1.0/abs(segi.a)   # TODO
#                if x== 0.0 :
#                    i = i + segi.f
#                e = 0
#            elif segi.isSegment() and segj.isDirac():
#                i = segj.f*segi.f(x/segj.a)*1.0/abs(segj.a)
#                if x== 0.0 :
#                    i = i + segj.f
#                e = 0
#            I += i
#            err += e
#        wyn[j]=I
#    return wyn

def _distr_signs(f):
    f_pos = f_neg = False
    for seg in f.segments:
        if seg.a < 0:
            f_neg = True
        if seg.b > 0:
            f_pos = True
    return f_pos, f_neg
def _distr_zero_signs(g):
    g_zero = g_zero_pos = g_zero_neg = False
    for seg in g.segments:
        if seg.a <= 0 <= seg.b:
            g_zero = True
            if seg.a < 0:
                g_zero_neg = True
            if seg.b > 0:
                g_zero_pos = True
    return g_zero, g_zero_pos, g_zero_neg

# Integrand for division
#def fun_div(segList, x):
#    if isscalar(x):
#        if abs(x) < 1:
#            fun = Convxrunner(segList, params.integration_finite).convprodx
#
#            y = convdivx(segList, x)
#        else:
#            y = convdivx2(segList, x)
#    else:
#        y = empty_like(x)
#        mask = (abs(x) < 1)
#        y[mask] = convdivx(segList, x[mask])
#        y[~mask] = convdivx2(segList, x[~mask])
#    return y
def convdiv(f, g):
    """Probabilistic division of piecewise functions f and g.
    """
    #f=f.splitByPoints([0])
    #g=g.splitByPoints([0])
    f = _split_for_prod_and_div(f)
    g = _split_for_prod_and_div(g)
    #f = f.splitByPoints([-1, 0, 1])
    #g = g.splitByPoints([-1, 0, 1])
    bf = f.getBreaks()
    bg = g.getBreaks()
    b = divide.outer(bf, bg).flatten()

    # check if result has infinite support
    f_pos, f_neg = _distr_signs(f)
    g_zero, g_zero_pos, g_zero_neg = _distr_zero_signs(g)
    if g_zero:
        if (f_pos and g_zero_neg) or (f_neg and g_zero_pos):
            b = hstack([b, [-Inf]])
        if (f_pos and g_zero_pos) or (f_neg and g_zero_neg):
            b = hstack([b, [Inf]])
    if min(b)*max(b)<0:
        b = hstack([b, [0]])
    ub = epsunique(b)

    ind = flatnonzero(ub == 0.0)
    ublist = ub.tolist()
    if len(ind)>0 and ind[0]<len(ub)-1:
        if isinf(ub[ind[0]+1]):
            bisect.insort_left(ublist, 1.0)
        else:
            pass
            #bisect.insort_left(ublist, (ub[ind[0]+1])/2)
    if len(ind)>0 and ind[0]>0:
        if isinf(ub[ind[0]-1]):
            bisect.insort_left(ublist, -1.0)
        else:
            pass
            #bisect.insort_left(ublist, (ub[ind[0]-1])/2)
    if len(ind) == 0 and ub[0]*ub[-1]<0:
        bisect.insort_left(ublist, 0)
    ub = unique(array(ublist))

    fg = PiecewiseDistribution([])
    if isinf(ub[0]):
        segList = _findSegListDiv(f, g, ub[1] - 1)
        #fun = partial(fun_div, segList)
        fun = Convxrunner(segList, params.integration_infinite).fun_div
        seg = MInfSegment(ub[1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[1:]
    if isinf(ub[-1]):
        segList = _findSegListDiv(f, g, ub[-2] + 1)
        #fun = partial(fun_div, segList)
        fun = Convxrunner(segList, params.integration_infinite).fun_div
        seg = PInfSegment(ub[-2], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub=ub[0:-1]
    for i in range(len(ub)-1) :
        segList = _findSegListDiv(f, g, (ub[i] + ub[i+1])/2)
        #fun = partial(fun_div, segList)
        fun = Convxrunner(segList, params.integration_infinite).fun_div
        _NoL = False
        _NoR = False
        if (ub[i] == 0):
            if testPole(fun, ub[i], pos = True):
                #segint = InterpolatedSegmentWithPole(ub[i],ub[i+1], fun, left_pole = True)
                if params.segments.debug_info:
                    print("probably pole at 0 left div", fun(0.0))
                seg = SegmentWithPole(ub[i], ub[i+1], fun, left_pole = True)
            else:
                if params.segments.debug_info:
                    print("probably no pole at 0 left div", fun(0.0))
                seg = Segment(ub[i], ub[i+1], fun)
                _NoL = True
        elif (ub[i+1] == 0): # TODO add proper condition
            #segint = InterpolatedSegmentWithPole(ub[i],ub[i+1], fun, left_pole = False)
            if testPole(fun, ub[i+1], pos = False):
                if params.segments.debug_info:
                    print("probably pole at 0 right div", fun(0.0))
                seg = SegmentWithPole(ub[i], ub[i+1], fun, left_pole = False)
            else:
                if params.segments.debug_info:
                    print("probably no pole at 0 right div", fun(0.0))
                seg = Segment(ub[i], ub[i+1], fun)
                _NoR = True
        else:
            if params.segments.debug_info:
                print("probably no pole div",  fun(0.0))
            seg = Segment(ub[i], ub[i+1], fun)
        segint = seg.toInterpolatedSegment(NoL = _NoL, NoR = _NoR)
        fg.addSegment(segint)
    # Discrete parts of distributions
    try:
        fg_discr = convdiracs(f, g, fun = operator.truediv)
    except:
        fg_discr = convdiracs(f, g, fun = operator.div)
    for seg in fg_discr.getDiracs():
        fg.addSegment(seg)
    return fg

#def convdivx(segList, xx):
#    """Probabilistic division of piecewise functions f and g,
#    integral at points xx, along X axis
#    """
#    if isscalar(xx):
#        xx=asfarray([xx])
#    res = zeros_like(xx)
#    for j in xrange(len(xx)) :
#        x = xx[j]
#        # Integrand for division
#        def fun(t):
#            y = segi.f(x*t)*segj.f(t)*abs(t)
#            return y
#        I = 0.0
#        err = 0.0
#        for segi, segj in segList :
#            if segi.isSegment() and segj.isSegment():
#                if x == 0:
#                    L = segj.a
#                    U = segj.b
#                else:
#                    L = segi.a / x
#                    U = segi.b / x
#                    L, U = min(L, U), max(L, U)
#                    L = max(segj.a, L)
#                    U = min(segj.b, U)
#
#                # integrate with variable transform even for finite intervals
#                # if the integration domain is very wide
#                force_minf = False
#                force_pinf = False
#                if not isinf(L) and L < 0 and (isinf(segi.a) or isinf(segj.a)):
#                    force_minf = True
#                if not isinf(U) and U > 0 and (isinf(segi.b) or isinf(segj.b)):
#                    force_pinf = True
#                i, e = _segint(fun, L, U, force_minf, force_pinf)
#
#            elif segi.isDirac() and segj.isSegment():
#                i = segi.f * segj.f(segi.a/x) * abs(segi.a)/x/x
#                e = 0
#                assert (segi.a != 0.0)
#            elif segi.isSegment() and segj.isDirac():
#                i = segj.f*segi.f(segj.a)*abs(segj.a)
#                e = 0
#                assert (segj.a != 0.0)
#            I += i
#            err += e
#        res[j] = I
#    return res
#
#def convdivx2(segList, xx):
#    """Probabilistic division of piecewise functions f and g,
#    integral at points xx, along Y axis
#    """
#    if size(xx)==1:
#        xx=asfarray([xx])
#    res = zeros_like(xx)
#    for j in xrange(len(xx)) :
#        x = xx[j]
#        # Integrand for division
#        def fun(t):
#            if x == 0:
#                y = segi.f(zeros_like(t)) * segj.f(t) * abs(t)
#            else:
#                y = segi.f(t) * segj.f(t/x) * abs(t)/x/x
#            return y
#        I = 0.0
#        err = 0.0
#        for segi, segj in segList:
#            if segi.isSegment() and segj.isSegment():
#                if x == 0:
#                    L = segj.a
#                    U = segj.b
#                else:
#                    L = segj.a * x
#                    U = segj.b * x
#                    L, U = min(L, U), max(L, U)
#                    L = max(segi.a, L)
#                    U = min(segi.b, U)
#
#                # integrate with variable transform even for finite intervals
#                # if the integration domain is very wide
#                force_minf = False
#                force_pinf = False
#                if not isinf(L) and L < 0 and (isinf(segi.a) or isinf(segj.a)):
#                    force_minf = True
#                if not isinf(U) and U > 0 and (isinf(segi.b) or isinf(segj.b)):
#                    force_pinf = True
#                i, e = _segint(fun, L, U, force_minf, force_pinf)
#            elif segi.isDirac() and segj.isSegment():
#                assert (segi.a != 0.0)
#                i = segi.f * segj.f(segi.a/x) * abs(segi.a) / x / x
#                e = 0
#
#            elif segi.isSegment() and segj.isDirac():
#                assert (segj.a != 0.0)
#                i = segj.f * segi.f(segj.a) * abs(segj.a)
#                e = 0
#            I += i
#            err += e
#        res[j] = I
#    return res

def _findSegListAdd(f, g, z):
    """It find list of segments for integration purposes, for given z
    input: f, g - picewise function, z = x + y
    output: list of segment products depends on z
    """
    seg_list = []
    for segi in f.segments:
        for segj in g.segments:
            if z - segi.a > segj.a and z - segi.b < segj.b:
                seg_list.append((segi, segj))
    return seg_list

def _findSegList(f, g, z, op):
    """It find list of segments for integration purposes, for given z
    input: f, g - picewise function, z = op(x,y), op - operation (+ - * /)
    output: list of segment products depends on z
    """
    list = []
    for segi in f.segments:
        for segj in g.segments:
            R1 = array([segi.a, segi.b, segi.a, segi.b])
            R2 = array([segj.a, segj.b, segj.b, segj.a])
            R = op(R1,R2)
            R = unique(R[isnan(R)==False])
            if min(R) < z < max(R):
                list.append((segi, segj))
    return list

def _findSegList2(f, g, z, op):
    """It find list of segments for integration purposes, for given z
    input: f, g - picewise function, z = op(x,y), op - operation (+ - * /)
    output: list of segment products depends on z. [UNUSED]
    """
    list = []
    for segi in f.segments:
        for segj in g.segments:
            R1 = array([segi.a, segi.b, segi.a, segi.b])
            R2 = array([segj.a, segj.b, segj.b, segj.a])
            if (segi.a < (segi.a+segi.b)/2.0 < segi.b):
                sig = sign((segi.a+segi.b)/2.0)
            elif isinf(segi.a):
                sig = sign(segi.b-1)
            elif isinf(segi.b):
                sig = sign(segi.a+1)
            R = sig*abs(op(R1,R2))
            R = unique(R[isnan(R)==False])
            if min(R) < z < max(R):
                list.append((segi, segj))
    return list

def _findSegListDiv(f, g, z):
    """It find list of segments for integration purposes, for given z
    input: f, g - picewise function, z = x/y,
    output: list of segment products depends on z.

    This function is used in division, taking -Infs into account.
    """
    seg_list = []
    for segi in f.segments:
        for segj in g.segments:
            # does x*z intersect the rectangular segment?
            yl = z * segj.a
            yr = z * segj.b
            if (yl > segi.a and yr < segi.b) or (yl < segi.b and yr > segi.a):
                seg_list.append((segi, segj))
    return seg_list


def _segint_(fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
            debug_info = False, debug_plot = False):
    """Common integration method for different kind of functions"""
    if L == U:
        return 0, 0
    if force_minf:
        i, e = integrate_fejer2_minf(fun, U, a = L, debug_info = debug_info, debug_plot = debug_plot)
    elif force_pinf:
        i, e = integrate_fejer2_pinf(fun, L, b = U, debug_info = debug_info, debug_plot = debug_plot)
    elif not isinf(L) and  not isinf(U):
        if force_poleL and force_poleU:
            i1, e1 = integrate_fejer2_Xn_transformP(fun, L, (L+U)*0.5, debug_info = debug_info, debug_plot = debug_plot)
            i2, e2 = integrate_fejer2_Xn_transformN(fun, (L+U)*0.5, U, debug_info = debug_info, debug_plot = debug_plot)
            i, e = i1+i2, e1+e2
        elif force_poleL:
            i, e = integrate_fejer2_Xn_transformP(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
        elif force_poleU:
            i, e = integrate_fejer2_Xn_transformN(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
        else:
            i, e = integrate_wide_interval(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
    elif isinf(L) and isfinite(U) :
        i, e = integrate_fejer2_minf(fun, U, debug_info = debug_info, debug_plot = debug_plot)
    elif isfinite(L) and isinf(U) :
        i, e = integrate_fejer2_pinf(fun, L, debug_info = debug_info, debug_plot = debug_plot)
    elif L<U:
        i, e = integrate_fejer2_pminf(fun, debug_info = debug_info, debug_plot = debug_plot)
    else:
        print("errors in convdiv: x, segi, segj, L, U =", L, U)
    return i,e


def convdiracs(f, g, fun = operator.add):
    """discrete convolution of f and g
    """
    fg = PiecewiseDistribution([])
    wyn = {}
    for fi in f.getDiracs():
        for gi in g.getDiracs():
            key = fun(fi.a, gi.a)
            if key in wyn:
                wyn[key] = wyn.get(key) + fi.f * gi.f
            else:
                wyn[key] = fi.f * gi.f
    for key in list(wyn.keys()):
        fg.addSegment(DiracSegment(key, wyn.get(key)))
    return fg
def _probDiracAtZeroInProd(f, g):
    """return None or DiracSegment at point 0 for product of piecewise function
    """
    intf, intg = 0, 0
    for fi in f.getDiracs():
        if fi.a == 0.0:
            for seg in g.getSegments():
                intf += seg.integrate() * fi.f
    for gi in g.getDiracs():
        if gi.a == 0.0:
            for seg in f.getSegments():
                intg += seg.integrate() * gi.f
    if intf + intg >0.0:
        return DiracSegment(0, intf + intg)
    else:
        return None

def _probDiracsInMin(f, g):
    """return discrete part of convmin
    """
    f_discr = PiecewiseDistribution([])
    for fi in f.getDiracs():
        f_discr.addSegment(DiracSegment(fi.a, fi.f * g.integrate(a = fi.a)))
    g_discr = PiecewiseDistribution([])
    for gi in g.getDiracs():
        g_discr.addSegment(DiracSegment(gi.a, gi.f * f.integrate(a = gi.a)))
    h_discr = PiecewiseDistribution([])
    for gi in g.getDiracs():
        dirac_f = f.getDirac(gi.a)
        if dirac_f is not None:
            h_discr.addSegment(DiracSegment(gi.a, gi.f * dirac_f.f))
    f_discr.add_diracs(g_discr)
    f_discr.add_diracs(h_discr)
    return f_discr

def _probDiracsInMax(f, g):
    """return discrete part of convmax
    """
    f_discr = PiecewiseDistribution([])
    for fi in f.getDiracs():
        f_discr.addSegment(DiracSegment(fi.a, fi.f * g.integrate(b = fi.a)))
    g_discr = PiecewiseDistribution([])
    for gi in g.getDiracs():
        g_discr.addSegment(DiracSegment(gi.a, gi.f * f.integrate(b = gi.a)))
    h_discr = PiecewiseDistribution([])
    for gi in g.getDiracs():
        dirac_f = f.getDirac(gi.a)
        if dirac_f is not None:
            h_discr.addSegment(DiracSegment(gi.a, gi.f * dirac_f.f))
    f_discr.add_diracs(g_discr)
    f_discr.add_diracs(h_discr)
    return f_discr
def dumpSegList(segList):
    """It dump segLis, for debug purposes only.
    """
    i=0
    for item in segList:
        i=i+1
        print(i, " ", item[0], ", ", item[1])



if __name__ == "__main__":
    def fun1(x):
        return 1-x**2

    from .segments import *
    from pacal import *
    import pickle
    params.general.parallel=False
    k = PiecewiseDistribution([])
    h = PiecewiseDistribution([])
    #k.addSegment(Segment(-1.0,1.0,partial(fun1)).toInterpolatedSegment())
    k.addSegment(ConstSegment(0.0, 1.0,1.0))
    h.addSegment(ConstSegment(1.0,1.5,2).toInterpolatedSegment())
    #k.addSegment(Segment(0.2,1.0,lambda x: 1.0+0.0*x))
    #k = k.toInterpolated()
    #h.addSegment(ConstSegment(-1,0,0.5))
    print(k, k.range())
    p = conv(k,h)
    print("=============================")
    #print pickle.dumps(p.segments[0].f.interp_at)
    #params.general.parallel=False
    p = conv(k,p)
    #p = conv(p,k)

    #print "======", p
    figure()
    p.plot()
    from pylab import show
    show()
