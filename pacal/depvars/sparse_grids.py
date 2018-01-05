"""Multidimensional interp. on sparse grids."""

from __future__ import print_function

import itertools

from pacal.utils import cheb_nodes, binomial_coeff
from pacal.utils import convergence_monitor
import pacal.params as params

from numpy import array, ones, atleast_1d
from numpy import newaxis, squeeze, exp, subtract, where, zeros_like, sum, dot, floor, linspace

# import faster Cython versions if possible
try:
    #import pyximport; pyximport.install()
    from pacal.bary_interp import c_dense_grid_interp
    have_Cython = True
    print("Using compiled sparse grid routine")
except:
    print("Compiled sparse grid routine not available")
    have_Cython = False


def gen_partitions(nd, d):
    """Generate partitions of d into nd blocks."""
    if nd == 1:
        return [(d,)]
    part = []
    for x in range(1,d):
        new_part = [(x,)+p for p in gen_partitions(nd - 1, d-x)]
        part.extend(new_part)
    return part

def gen_Q(q, d):
    """Generate the set Q(q,d) of nested indices for nesting depth q
    and d dimensions.

    Q corresponds to P in the paper of Wasilkowski and Wozniakowski."""
    assert q >= d
    Q = []
    for i in range(q-d+1, q+1):
        Q.extend(gen_partitions(d, i))
    return Q

def cheb_weights(n):
    weights = ones(n)
    weights[::2] = -1
    weights[0] /= 2
    weights[-1] /= 2
    return weights



class AdaptiveSparseGridInterpolator(object):
    def __init__(self, f, d, a = None, b = None):
        if a is None:
            a = -ones(d)
        if b is None:
            b = ones(d)
        self.adaptive_init(f, d, a, b)
        self.adaptive_interp()

    def adaptive_init(self, f, d, a, b):
        self.f = f
        self.d = d
        self.a = a
        self.b = b
        self.q = self.d   # minimal nesting level
        self.init_ni = 11  # smallest degree along an axis
        ni = self.init_ni
        self.exps = [None]
        self.cheb_weights_cache = [None]
        self.xroot_cache = [list([None]) for j in range(self.d)] # cache for X coords of roots
        for i in range(self.q - self.d + 1 + 1):  # maximum degree is q - d + 1 by definition of Q
            self.exps.append(ni)
            self.cheb_weights_cache.append(cheb_weights(ni))
            for j in range(self.d):
                self.xroot_cache[j].append(cheb_nodes(ni, self.a[j], self.b[j]))
            if ni == 1:
                ni = 3
            else:
                ni = ni * 2 - 1
        self.ni = ni

        self.nodes, self.subgrid_map = self.get_nodes(self.q)
        Xs = [array([self.xroot_cache[j][self.q - self.d + 1][n[j]] for n in self.nodes]) for j in range(self.d)]
        self.Ys = atleast_1d(squeeze(self.f(*Xs)))

    def adaptive_interp(self, par = None):
        if par is None:
            par = params.interpolation_nd
        maxq = par.maxq
        q = self.q
        max_q = q-self.d+1
        ni = self.ni
        old_err = None
        cm = convergence_monitor(par = par.convergence)

        while q <= maxq:
            self.exps.append(ni)
            self.cheb_weights_cache.append(cheb_weights(ni))
            for j in range(self.d):
                self.xroot_cache[j].append(cheb_nodes(ni, self.a[j], self.b[j]))
            if ni == 1:
                ni = 3
            else:
                ni = ni * 2 - 1
                q = q + 1
            new_nodes, new_subgrid_map = self.get_nodes(q)
            # compute incremental nodes
            old_nodes = [tuple(2*i for i in n) for n in self.nodes]
            inc_nodes = list(set(new_nodes) - set(old_nodes))
            inc_nodes.sort()
            max_q = q-self.d+1
            inc_Xs = [array([self.xroot_cache[j][max_q][n[j]] for n in inc_nodes]) for j in range(self.d)]
            inc_Ys = atleast_1d(squeeze(self.f(*inc_Xs)))
            err = self.test_accuracy(inc_Xs, inc_Ys)
            maxy = max(abs(inc_Ys).max(), abs(self.Ys).max())
            if par.debug_info:
                print("interp. err", err, old_err, q)
            cm.add(err, maxy)
            if cm.test_convergence()[0]:
                break
            old_err = err
            # error too large, update the interpolator
            self.q = q
            self.ni = ni
            # now update nodes. TODO: cleanup!
            nodemap = dict((n, i) for i, n in enumerate(old_nodes))
            inc_nodemap = dict((n, i) for i, n in enumerate(inc_nodes))
            new_Xs = []
            new_Ys = []
            max_q = q - self.d + 1 # max degree
            for n in new_nodes:
                if n in nodemap:
                    Y = self.Ys[nodemap[n]]
                if n in inc_nodes:
                    Y = inc_Ys[inc_nodemap[n]]
                new_Ys.append(Y)
            self.nodes = new_nodes
            self.subgrid_map = new_subgrid_map
            self.Ys = array(new_Ys)

    def test_accuracy(self, new_Xs, new_Ys):
        """Test accuracy by comparing true and interpolated values at
        given points."""
        N_samp = -1
        if N_samp == -1 or N_samp >= len(new_Xs[0]):
            errs = abs(self.interp_at(*new_Xs) - new_Ys)
        else:
            ns = [int(i) for i in floor(linspace(0, len(new_Xs[0])-1, N_samp))]
            test_Xs = [X[ns] for X in new_Xs]
            test_Ys = new_Ys[ns]
            errs = abs(self.interp_at(*test_Xs) - test_Ys)
        err = errs.max()
        return err

    def get_nodes(self, q):
        """Compute nodes for current ni."""
        nodes = []
        node_inds_map = {}
        subgrid_map = {} # maps subgrids to nodes

        ps = gen_Q(q, self.d)  # compute partitions.  Each partition corresponds to a subgrid
        for p in ps:
            # each new node is described with a set of indices.  Each
            # index is the number of interpolation root for given node
            # and dimension
            max_q = q-self.d+1
            max_d = self.exps[max_q]
            if self.init_ni == 1:
                new_nodes = itertools.product(*[list(range(0, max_d, 1 << (max_q - e))) if e > 1 else [max_d/2] for e in p])
            else:
                new_nodes = itertools.product(*[list(range(0, max_d, 1 << (max_q - e))) for e in p])

            # node indices for this subgrid
            subgrid_indices = []
            for ni in new_nodes:
                if ni in node_inds_map:
                    idx = node_inds_map[ni]
                else:
                    idx = len(nodes)
                    node_inds_map[ni] = idx
                    nodes.append(ni)
                subgrid_indices.append(idx)
            # subgrid's coefficient
            si = sum(p)
            c = (-1)**(q-si) * binomial_coeff(self.d - 1, q-si)
            subgrid_map[p] = (c, array(subgrid_indices))
        if params.interpolation_nd.debug_info:
            print(len(subgrid_map), "rect. subgrids,", sum(len(subgrid_map[k]) for k in subgrid_map), "ops per X,", len(nodes), "nodes,", end=' ')
            print("full grid size", self.exps[q-self.d+1]**self.d, "dim", self.d)
        return nodes, subgrid_map

    def interp_at(self, *X):
        if not hasattr(X[0], '__iter__'):
            scalar_x = True
            X = [array([x]) for x in X]
        else:
            scalar_x = False
            shape = X[0].shape
            X = [x.ravel() for x in X]
            if X[0].shape[0] == 0:
                return array([])
        y = 0
        full_grid_cache = {}
        for p, subgrid in self.subgrid_map.items():
            c, ni = subgrid
            #yi = self.full_grid_interp(p, self.Ys[ni], X, full_grid_cache)
            yi = self.full_grid_interp_new(p, self.Ys[ni], X, full_grid_cache)
            y += c * yi
        if scalar_x:
            y = squeeze(y)
        else:
            y.shape = shape
        return y

    def full_grid_interp(self, p, fs, X, fg_cache):
        # accept X as a list of arrays
        den = 1
        if have_Cython:
            one_over_x_m_xi_list = [] # C
        else:
            one_over_x_m_xi_grid = None
        for i, e in enumerate(p):
            if (i, e) in fg_cache:
                one_over_x_m_xi, den_factor = fg_cache[(i, e)]
            else:
                Xs = self.xroot_cache[i][e]
                Ws = self.cheb_weights_cache[e]
                diff = subtract.outer(X[i], Xs)
                mask = (diff == 0).any(axis=-1)
                if any(mask):
                    w = where(diff == 0)
                    one_over_x_m_xi = zeros_like(diff)
                    one_over_x_m_xi[w] = 1
                    one_over_x_m_xi[~mask] = Ws/(diff[~mask])
                else:
                    one_over_x_m_xi = Ws/diff
                den_factor = sum(one_over_x_m_xi, axis=1)
                fg_cache[(i, e)] = one_over_x_m_xi, den_factor

            den *= den_factor
            if have_Cython:
                one_over_x_m_xi_list.append(one_over_x_m_xi) # C
            else:
                if one_over_x_m_xi_grid is None:
                    one_over_x_m_xi_grid = one_over_x_m_xi
                else:
                    newshape = (one_over_x_m_xi.shape[0],) + (1,) + (one_over_x_m_xi.shape[-1],)
                    one_over_x_m_xi_grid = one_over_x_m_xi_grid[...,newaxis] * one_over_x_m_xi.reshape(newshape)
                one_over_x_m_xi_grid = one_over_x_m_xi_grid.reshape((X[0].shape[0], -1))

        if have_Cython:
            num = c_dense_grid_interp(one_over_x_m_xi_list, fs)
        else:
            num = dot(one_over_x_m_xi_grid, fs)
        return num / den

    def full_grid_interp_new(self, p, fs, X, fg_cache):
        # accept X as a list of arrays
        first = True
        for i, e in reversed(list(enumerate(p))):
            if (i, e) in fg_cache:
                one_over_x_m_xi, den_factor = fg_cache[(i, e)]
            else:
                Xs = self.xroot_cache[i][e]
                Ws = self.cheb_weights_cache[e]
                diff = subtract.outer(X[i], Xs)
                mask = (diff == 0).any(axis=-1)
                if any(mask):
                    w = where(diff == 0)
                    one_over_x_m_xi = zeros_like(diff)
                    one_over_x_m_xi[w] = 1
                    one_over_x_m_xi[~mask] = Ws/(diff[~mask])
                else:
                    one_over_x_m_xi = Ws/diff
                den_factor = sum(one_over_x_m_xi, axis=1)
                fg_cache[(i, e)] = one_over_x_m_xi, den_factor

            if first:
                fs_r = fs.reshape((-1,one_over_x_m_xi.shape[1]))
                fs = dot(one_over_x_m_xi, fs_r.T)
                first = False
            else:
                fs_r = fs.reshape((X[0].shape[0], fs.shape[1] // one_over_x_m_xi.shape[1], one_over_x_m_xi.shape[1]))
                fs = (fs_r * one_over_x_m_xi[:,newaxis,:]).sum(axis=-1)
            fs /= den_factor[:,newaxis]

        return fs[:,0]

if __name__ == "__main__":

    def f(*X):
        #return 1
        #return sum(X)
        #return sum(sin(x) for x in X)
        return exp(-sum(x*x for x in X))


    d = 3
    X = ([0.1, 0.2] * 20)[:d]

    #X[1] = -1
    #a = -ones(d)
    #a[1] = 0 # X is now out of range, error should be large



    global debug
    debug = False

    #import pstats, cProfile
    #cProfile.runctx("AdaptiveSparseGridInterpolator(f, d)", globals(), locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()


    ai = AdaptiveSparseGridInterpolator(f, d)#, a = a)
    yi = ai.interp_at(*X)
    yt = f(*X)
    print(yi, yt, yi - yt)
    debug = True
    X = [array([0.1, 0.2, 1.0]), array([0.2, 0.1, 0.5]), array([0.2, 0.1, 0.1])][0:d]
    X = [array([0.1, 0.2, 1.0]), array([0.2, 0.1, 0.5]), array([0.2, 0.1, 0.1])][0:d]
    yi = ai.interp_at(*X)
    yt = f(*X)
    print(yi, yt, yi - yt)
