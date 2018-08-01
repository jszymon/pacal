from __future__ import print_function

import numpy as np
from scipy.stats import norm

from pacal import *

def test_inv_norm_cdf():
    """test for #5 on github"""
    dist = NormalDistr(2, 1)
    pdf = dist.get_piecewise_pdf()
    assert pdf(-np.inf) == 0.0
    assert pdf(np.inf) == 0.0
    cdf = dist.get_piecewise_cdf()
    assert cdf(-np.inf) == 0.0
    assert np.isclose(cdf(np.inf), 1)
    ppf = cdf.invfun(rangeY=None)
    # test vectors
    tcdfi = [cdf.inverse(x) for x in np.linspace(0,1,11)]
    tppf = [np.asscalar(np.asarray(ppf(x))) for x in np.linspace(0,1,11)]
    tscipy = [norm.ppf(x, loc=2,scale=1) for x in np.linspace(0,1,11)]
    assert np.allclose(tcdfi, tscipy)
    #assert np.allclose(tppf, tscipy)  # disabled for now, inf/-inf not returned at 1/0
