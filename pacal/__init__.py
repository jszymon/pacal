"""PaCal, the probabilistic calculator."""

from __future__ import print_function

from . import params

import numpy as _np
from pylab import show

_np.seterr(all="ignore")

from .utils import Inf

from .distr import DiscreteDistr
from .distr import exp, log, atan, min, max, sqrt, sign, sin, cos, tan, tanh, sigmoid

from .standard_distr import FunDistr
from .standard_distr import NormalDistr
from .standard_distr import UniformDistr
from .standard_distr import TrapezoidalDistr
from .standard_distr import CauchyDistr
from .standard_distr import ChiSquareDistr
from .standard_distr import ExponentialDistr
from .standard_distr import GammaDistr
from .standard_distr import BetaDistr
from .standard_distr import ParetoDistr
from .standard_distr import LevyDistr
from .standard_distr import LaplaceDistr
from .standard_distr import StudentTDistr
from .standard_distr import SemicircleDistr
from .standard_distr import FDistr
from .standard_distr import WeibullDistr
from .standard_distr import GumbelDistr
from .standard_distr import FrechetDistr
from .standard_distr import MollifierDistr

from .standard_distr import OneDistr
from .standard_distr import ZeroDistr
from .standard_distr import BinomialDistr
from .standard_distr import BernoulliDistr
from .standard_distr import PoissonDistr
from .standard_distr import MixDistr
from .distr import CondGtDistr
from .distr import CondLtDistr
from .distr import ConstDistr
from .distr import Gt
from .distr import Lt
from .distr import Between

from .stats.noncentral_distr import NoncentralTDistr
from .stats.noncentral_distr import NoncentralChiSquareDistr
from .stats.noncentral_distr import NoncentralBetaDistr
from .stats.noncentral_distr import NoncentralFDistr

from .stats.iid_ops import iid_sum, iid_prod, iid_max, iid_min, iid_average, iid_average_geom
from .stats.iid_ops import iid_order_stat, iid_median

from .stats.distr_est import LoglikelihoodEstimator

# dependent variables
from .depvars.copulas import PiCopula, FrankCopula, ClaytonCopula, GumbelCopula
from .depvars.nddistr import NDNormalDistr, IJthOrderStatsNDDistr
from .depvars.models import TwoVarsModel, Model




def _pickle_method(method):
    obj = method.__self__
    cls = method.__self__.__class__
    func_name = method.__func__.__name__
    #print "pickle>>>", func_name, obj, cls
    if func_name.startswith('__') and not func_name.endswith('__'):
        #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    #print "upickle>>>", func_name, obj, cls
    if obj and func_name in obj.__dict__:
        cls, obj = obj, None # if func_name is classmethod
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

if params.general.parallel:
    # make ufuncs picklable
    import types
    try:
        import copyreg
        #copy_reg.pickle(_np.ufunc, _pickle_ufunc, _unpickle_ufunc)
        copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    except:
        import copy_reg
        #copy_reg.pickle(_np.ufunc, _pickle_ufunc, _unpickle_ufunc)
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
