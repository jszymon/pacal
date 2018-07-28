from __future__ import print_function

import numpy as np
from pacal import *

def test_inv_norm_cdf():
    """test for #5 on github"""
    dist = NormalDistr(2, 1)
    cdf = dist.get_piecewise_cdf()
    assert cdf(-np.inf) == 0.0
    assert np.isclose(cdf(np.inf), 1)
    ppf = cdf.invfun(rangeY=None)
    print([ppf(x/10) for x in np.linspace(0,1,11)])
