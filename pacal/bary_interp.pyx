import numpy as np

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
#from stdlib cimport malloc, free

cimport numpy as np
cimport cython



DTYPE = np.double
ctypedef np.double_t DTYPE_t



#@cython.boundscheck(False)
#@cython.cdivision(False)
#def bary_interp_old(np.ndarray[DTYPE_t, ndim=1] Xs not None, np.ndarray[DTYPE_t, ndim=1] Ys not None,
#                np.ndarray[DTYPE_t, ndim=1] weights not None, np.ndarray[DTYPE_t, ndim=1] X not None):
#    cdef unsigned int i, j
#    cdef unsigned int m = Xs.shape[0], n = X.shape[0]
#    cdef DTYPE_t temp, num, den, x, nx
#
#    cdef np.ndarray[DTYPE_t, ndim=1] y
#
#    y = np.empty([X.shape[0]], dtype = DTYPE)
#
#    for i in range(n):
#        x = X[i]
#        num = 0.0
#        den = 0.0
#        for j in range(m):
#            nx = Xs[j]
#            if x == nx:
#                y[i] = Ys[j]
#                break
#            temp = weights[j] / (x - nx)
#            num += temp * Ys[j]
#            den += temp
#        else:
#            # Tricky case which can occur when ends of the interval are
#            # almost equal.  xdiff can be close to but nonzero, but the sum
#            # in the denominator can be exactly zero.
#            if den != 0:
#                y[i] = num / den
#            else:
#                # find smallest xdiff
#                y[i] = Ys[np.argmin([abs(x - xn) for xn in Xs])]
#    return y


@cython.boundscheck(False)
@cython.cdivision(False)
def bary_interp(np.ndarray[DTYPE_t, ndim=1] Xs not None, np.ndarray[DTYPE_t, ndim=1] Ys not None,
                       np.ndarray[DTYPE_t, ndim=1] weights not None, np.ndarray[DTYPE_t, ndim=1] X not None):
    cdef unsigned int i, j
    cdef unsigned int m = Xs.shape[0], n = X.shape[0]
    cdef DTYPE_t temp, num, den, x, nx

    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty([X.shape[0]], dtype = DTYPE)

    # copy input numpy arrays to standard C arrays
    cdef DTYPE_t *Xs_copy = <DTYPE_t *>malloc(m * sizeof(DTYPE_t))
    cdef DTYPE_t *Ys_copy = <DTYPE_t *>malloc(m * sizeof(DTYPE_t))
    cdef DTYPE_t *weights_copy = <DTYPE_t *>malloc(m * sizeof(DTYPE_t))
    for j in range(m):
        Xs_copy[j] = Xs[j]
        Ys_copy[j] = Ys[j]
        weights_copy[j] = weights[j]

    for i in range(n):
        x = X[i]
        num = 0.0
        den = 0.0
        for j in range(m):
            nx = Xs_copy[j]
            if x == nx:
                y[i] = Ys_copy[j]
                break
            temp = weights_copy[j] / (x - nx)
            num += temp * Ys_copy[j]
            den += temp
        else:
            # Tricky case which can occur when ends of the interval are
            # almost equal.  xdiff can be close to but nonzero, but the sum
            # in the denominator can be exactly zero.
            if den != 0:
                y[i] = num / den
            else:
                # find smallest xdiff
                y[i] = Ys[np.argmin([abs(x - xn) for xn in Xs])]

    free(weights_copy)
    free(Ys_copy)
    free(Xs_copy)
    return y

@cython.boundscheck(False)
@cython.cdivision(False)
def c_dense_grid_interp(one_over_x_m_xi_list not None, np.ndarray[DTYPE_t, ndim=1] fs not None):
    cdef unsigned int i, j
    cdef int k # need int for reversed loop
    cdef unsigned int d = len(one_over_x_m_xi_list)
    cdef unsigned int n = one_over_x_m_xi_list[0].shape[0]
    cdef unsigned int M = fs.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty([n], dtype = DTYPE)


    # copy input numpy arrays to standard C arrays

    cdef DTYPE_t *fs_copy = <DTYPE_t *>malloc(M * sizeof(DTYPE_t))
    for j in range(M):
        fs_copy[j] = fs[j]
    cdef unsigned int *ms = <unsigned int *>malloc(d * sizeof(unsigned int))
    cdef DTYPE_t **one_over_x_m_xi_list_copy = <DTYPE_t **>malloc(M * sizeof(DTYPE_t *))
    cdef np.ndarray[DTYPE_t, ndim=2] one_over_x_m_xi

    for i in range(d):
        one_over_x_m_xi = one_over_x_m_xi_list[i]
        ms[i] = one_over_x_m_xi.shape[1]
        one_over_x_m_xi_list_copy[i] = <DTYPE_t *>malloc(ms[i] * n * sizeof(DTYPE_t))
        for j from 0 <= j < n:
             for k from 0 <= k < ms[i]:
                 one_over_x_m_xi_list_copy[i][k*n+j] = one_over_x_m_xi[j,k]


    # compute the cross product

    cdef unsigned int idx = 0
    cdef unsigned int *nd_idx = <unsigned int *>malloc(d * sizeof(unsigned int))
    cdef DTYPE_t *temp = <DTYPE_t *>malloc(n * sizeof(DTYPE_t))
    cdef DTYPE_t *num = <DTYPE_t *>malloc(n * sizeof(DTYPE_t))

    for j in range(d):
        nd_idx[j] = 0
    for i in range(n):
        num[i] = 0
    # loop over grid points
    for idx in range(M):
        for i from 0 <= i < n:
            temp[i] = fs_copy[idx] * one_over_x_m_xi_list_copy[0][n * nd_idx[0] + i]
        for j in range(1, d):
            for i from 0 <= i < n:
                temp[i] *= one_over_x_m_xi_list_copy[j][n * nd_idx[j] + i]
        for i from 0 <= i < n:
            num[i] += temp[i]
        # advance to next index
        for k from d > k >= 0:
            nd_idx[k] += 1
            if nd_idx[k] < ms[k]:
                break
            else:
                nd_idx[k] = 0
        #print idx, ",".join([str(nd_idx[k]) for k in range(d)])
    for i in range(n):
        y[i] = num[i]

    free(num)
    free(temp)
    free(nd_idx)
    for j in range(d):
        free(one_over_x_m_xi_list_copy[j])
    free(one_over_x_m_xi_list_copy)
    free(ms)
    free(fs_copy)
    return y

