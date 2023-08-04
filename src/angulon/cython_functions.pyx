#distutils: language = c++
#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:19:40 2022
"""

from libc.stdio cimport printf
from libc.math cimport sqrt
import scipy
import warnings
from scipy.linalg.cython_lapack cimport zheevr, zgeev
import numpy as np
import config
from print_msg import print_status
# Cython internally would handle this ambiguity so that the user would not need to use different names.
cimport cython 
cimport openmp
cimport numpy as cnp # Importing parts of NumPY C-API
cnp.import_array()  # So numpy's C API won't segfault
include "array_defs.pxi"
from libc.stdlib cimport malloc, free

def cython_eigsh(float[:,:] H):
    
    cdef int N_block = len(H)
    cdef complex[::1,:] H_fortran = np.asfortranarray(np.complex128(H))
    cdef double[::1] eigval = np.zeros((N_block,), dtype=np.float64, order='F')
    cdef complex[::1,:] eigvec = np.zeros((N_block,N_block), dtype=np.complex128, order='F')
    cdef int[:] idx_argsort = np.zeros((N_block), dtype=np.int32)
    eigvec = getEigVecs(H_fortran, eigval, eigvec, N_block, idx_argsort)
    eigenvectors = np.asarray(eigvec.base.real[:,:-1], dtype = np.float32, order='C')
    eigenvalues = np.asarray(eigval.base[:-1], dtype = np.float32, order='C')
    
    return eigenvalues, eigenvectors

cdef double[::1] getEigVals(complex[::1,:] H_fortran, double[::1] eigval_tmp, complex[::1,:] eigvec_tmp, int N_block, int[:] idx) nogil:
    eigval_tmp[:] = 0.0
    eigvec_tmp[:,:] = 0.0
    ZGEEVNoGIL(H_fortran[:,:], eigval_tmp, eigvec_tmp, idx, N_block)
    # The eigenvalues are returned in ascending order, but not repeated according to their multiplicity.
    # However, it is essentially impossible to get the same eigenvalues twice.   
    return eigval_tmp

cdef complex[::1,:] getEigVecs(complex[::1,:] H_fortran, double[::1] eigval_tmp, complex[::1,:] eigvec_tmp, int N_block, int[:] idx) nogil:
    eigval_tmp[:] = 0.0
    eigvec_tmp[:,:] = 0.0
    ZGEEVNoGIL(H_fortran[:,:], eigval_tmp, eigvec_tmp, idx, N_block)
    # The eigenvalues are returned in ascending order, but not repeated according to their multiplicity.
    # However, it is essentially impossible to get the same eigenvalues twice.   
    return eigvec_tmp
    
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
        int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    long index
    np.float64_t value

cdef int _compare(const_void *a, const_void *b):
    cdef np.float64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1

cdef int[:] argsort(double[:] data, int[:] order) nogil:
    cdef long i
    cdef long n = data.shape[0]
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)
    
    return order

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         real(double complex x)
    double cabs   "abs" (double complex x)
    double complex sqrt(double complex x)
    
@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void ZGEEVNoGIL(complex[::1,:] H, double[:] eigvals,
                complex[::1,:] Z, int[:] idx, int nrows) nogil:
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.
    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigenvalues.
    Z : array_like
        Output array of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    cdef char jobvl = b'N'
    cdef char jobvr = b'V'
    cdef int i, j, k, lwork = -1
    cdef int same_eigv = 0
    cdef complex dot
    cdef complex wkopt
    cdef int info=0
    cdef complex * work

    #These need to be freed at end
    cdef complex * eival = <complex *>PyDataMem_NEW(nrows * sizeof(complex))
    cdef complex * vl = <complex *>PyDataMem_NEW(nrows * nrows *
                                                 sizeof(complex))
    cdef complex * vr = <complex *>PyDataMem_NEW(nrows * nrows *
                                                 sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((2*nrows) * sizeof(double))

    # First call to get lwork
    zgeev(&jobvl, &jobvr, &nrows, &H[0,0], &nrows,
          eival, vl, &nrows, vr, &nrows,
          &wkopt, &lwork, rwork, &info)
    lwork = int(real(wkopt))
    work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    # Solving
    zgeev(&jobvl, &jobvr, &nrows, &H[0,0], &nrows,
          eival, vl, &nrows, vr, &nrows, #&Z[0,0],
          work, &lwork, rwork, &info)
    for i in range(nrows):
        eigvals[i] = real(eival[i])
    # After using lapack, numpy...
    # lapack does not seems to have sorting function
    # zheevr sort but not zgeev
    idx = argsort(eigvals, idx)
    for i in range(nrows):
        eigvals[i] = real(eival[idx[i]])
        for j in range(nrows):
            Z[j,i] = vr[j + idx[i]*nrows]

    for i in range(1, nrows):
        if cabs(eigvals[i] - eigvals[i-1]) < 1e-12:
            same_eigv += 1
            for j in range(same_eigv):
                dot = 0.
                for k in range(nrows):
                    dot += conj(Z[k,i-j-1]) * Z[k,i]
                for k in range(nrows):
                    Z[k,i] -= Z[k,i-j-1] * dot
                dot = 0.
                for k in range(nrows):
                    dot += conj(Z[k,i]) * Z[k,i]
                dot = sqrt(dot)
                for k in range(nrows):
                    Z[k,i] /= dot
        else:
            same_eigv = 0

    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(vl)
    PyDataMem_FREE(vr)
    PyDataMem_FREE(eival)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")