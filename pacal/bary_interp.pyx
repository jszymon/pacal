import numpy as np

from stdlib cimport malloc, free

cimport numpy as np
cimport cython



DTYPE = np.double
ctypedef np.double_t DTYPE_t



@cython.boundscheck(False)
@cython.cdivision(False)
def bary_interp_old(np.ndarray[DTYPE_t, ndim=1] Xs not None, np.ndarray[DTYPE_t, ndim=1] Ys not None,
                np.ndarray[DTYPE_t, ndim=1] weights not None, np.ndarray[DTYPE_t, ndim=1] X not None):
    cdef unsigned int i, j
    cdef unsigned int m = Xs.shape[0], n = X.shape[0]
    cdef DTYPE_t temp, num, den, x, nx

    cdef np.ndarray[DTYPE_t, ndim=1] y

    y = np.empty([X.shape[0]], dtype = DTYPE)

    for i in range(n):
        x = X[i]
        num = 0.0
        den = 0.0
        for j in range(m):
            nx = Xs[j]
            if x == nx:
                y[i] = Ys[j]
                break
            temp = weights[j] / (x - nx)
            num += temp * Ys[j]
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
    return y


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



