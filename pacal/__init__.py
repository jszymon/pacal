"""PaCal, the probabilistic calculator."""

import numpy as _np
_np.seterr(all="ignore")

from utils import Inf

from distr import DiscreteDistr, exp, log, atan, min, max, sqrt, sign, abs

from standard_distr import FunDistr
from standard_distr import NormalDistr
from standard_distr import UniformDistr
from standard_distr import CauchyDistr
from standard_distr import ChiSquareDistr
from standard_distr import ExponentialDistr
from standard_distr import GammaDistr
from standard_distr import BetaDistr
from standard_distr import ParetoDistr
from standard_distr import LevyDistr
from standard_distr import LaplaceDistr
from standard_distr import StudentTDistr
from standard_distr import SemicircleDistr
from standard_distr import FDistr
from standard_distr import WeibullDistr
from standard_distr import GumbelDistr
from standard_distr import FrechetDistr
from standard_distr import NoncentralTDistr
from standard_distr import NoncentralChiSquareDistr

from standard_distr import ConstDistr
from standard_distr import OneDistr
from standard_distr import ZeroDistr
from standard_distr import BinomialDistr
from standard_distr import MixDistr
from standard_distr import CondGtDistr
from standard_distr import CondLtDistr

from stats.iid_ops import iid_sum, iid_prod, iid_max, iid_min, iid_average
from stats.iid_ops import iid_order_stat, iid_median

from pylab import show
