import numpy as np

from pacal import LogLogisticDistr


class TestLogLogistic:
    def setup(self):
        self.make_distrs()
    def make_distrs(self):
        test_ks = [0.2, 0.5, 0.99, 1, 1.5, 2, 2.5, 5, 10, 100]
        self.distrs = [LogLogisticDistr(k) for k in test_ks]
    def test_int_error(self):
        for d in self.distrs:
            assert np.abs(d.int_error() <= 1e-15)
    def theor_moment(self, d, i):
        k, s = d.k, d.s
        if i >= k:
            return np.inf
        return s**i / np.sinc(i/k)
    def test_moments(self):
        for d in self.distrs:
            for i in range(1, 4):
                m = d.moment(i, 0)
                mt = self.theor_moment(d, i)
                print(d.k, i, m, mt)
                if np.isfinite(mt):
                    assert np.abs(m - mt) < 1e-14
                else:
                    assert not np.isfinite(m), \
                      "number reported for nonexistant moment"
