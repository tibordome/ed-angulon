#distutils: language = c++
#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:40:32 2021

@author: tibor
"""

from cython.parallel import parallel, prange, threadid
cimport openmp
from copy import deepcopy
from functools import partial
import numpy as np
cimport cython
from libc.math cimport round as cyround
from libc.stdio cimport printf 
from libc.math cimport sqrt, pi, pow
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement, predecrement
import config
import scipy.special as sp
import scipy.integrate

def getH(int L):
    """
    Calculate Hamiltonian matrix in block L
    Parameters:
    --------------
    L: int, block
    Returns:
    --------------
    data_list: float list containing non-zero entries of matrix H
    data_row_list: int list for rows associated to non-zero entries of matrix H
    data_col_list: int list for columns associated to non-zero entries of matrix H
    states_L.size(): size of H-block (number of rows)"""
    
    if L > config.L_max:
        raise ValueError('You want to diagonalize an L-block which does not exist!')
    
    # Loading states_L_tmp into the map states_L
    states_L_tmp = config.states_all[L] # List
    cdef map[int,vector[int]] states_L
    for i in range(len(states_L_tmp)):
        states_L[i].push_back(states_L_tmp[i][0])
        states_L[i].push_back(states_L_tmp[i][1])
        states_L[i].push_back(states_L_tmp[i][2])
        for j in range((len(states_L_tmp[i])-3)//2):
            states_L[i].push_back(states_L_tmp[i][3+2*j])
            states_L[i].push_back(states_L_tmp[i][3+2*j+1])
            
    # Defining memoryviews
    cdef float[:,:,:,:,:,:] CG = config.CG
    cdef float[:,:,:,:,:] W_1 = config.W_1
    cdef float[:,:,:,:,:] W_2 = config.W_2
    cdef float[:,:] U = config.U
    cdef float[:] w = config.w
    cdef int l_max = config.l_max
    cdef float delta_k = config.delta_k
    cdef int[:,:] bos_qindices = config.bos_qindices
    cdef int[:,:] pos = config.states_pos
    cdef int[:,:] section = config.states_section
    cdef int nlen = <int>((len(config.states_all[L])/openmp.omp_get_max_threads()))
    cdef int nb_blocks = openmp.omp_get_max_threads()+1
    cdef bint with_W_1 = config.with_W_1
    cdef bint with_W_2 = config.with_W_2  
    cdef bint with_mol = config.with_mol
    cdef bint with_Froehlich = config.with_Froehlich
    cdef Py_ssize_t r_idx
    cdef float[:,:] matrix_data = np.zeros((nb_blocks, <int>states_L.size()**2//(nb_blocks-2)*10), dtype=np.float32) # *10 has proven sufficiently large in all cases.
    for r_idx in prange(nb_blocks, schedule = "dynamic", nogil=True):
        matrix_data[r_idx] = H(L, matrix_data[r_idx], states_L, CG, W_1, W_2, U, w, bos_qindices, pos, section, with_W_1, with_W_2, with_mol, with_Froehlich, l_max, delta_k, r_idx, nlen, openmp.omp_get_max_threads())
    data_list = []
    data_row_list = []
    data_col_list = []
    for i in range(nb_blocks):
        run = 0
        while matrix_data[i][run+2] != 0.0:
            data_row_list.append(matrix_data[i][run])
            data_col_list.append(matrix_data[i][run+1])
            data_list.append(matrix_data[i][run+2])
            run += 3
    return data_list, data_row_list, data_col_list, states_L.size()

cdef map[int, vector[int]] diff(vector[int] row, vector[int] col, int r_H, int c_H) nogil:
    """
    Deals with differences btw. 2 states row and col
    Assumption 1: Row has more phonons than col, i.e. row[2] < col[2]
    Assumption 2: Number of phonons in row and col are less or equal to 2, i.e. abs(row[2]-col[2]) <= 2
    Assumption 3: Row has at least 1 phonon, i.e. row[2] > 0
    Parameters:
    --------------
    row: row state vector
    col: column state vector
    r_H: position of row state vector in block (L)
    c_H: position of column state vector in block (L)
    Returns:
    --------------
    diff: a map containing all occupational differences between row and col
    The format of diff is [diff_val, [# of box, 0 (for r) or 1 (for c), (larger) occ in box], [...], ...]"""
    
    cdef int diff_val = 0 # This counts the number of occupational differences, considering all boxes and counting more than 1 if occupation differs by more than 1
    cdef map[int, vector[int]] diff
    cdef int where_to_append = 1
    cdef int r_run = 0
    cdef int c_run = 0
    
    while r_run < cyround((row.size()-3)/2)+1 and c_run < cyround((col.size()-3)/2)+1:
        if r_run < cyround((row.size()-3)/2) and c_run >= cyround((col.size()-3)/2): # E.g. in case there are no phonons in col (recall: this would not be allowed for row)
            diff[where_to_append].push_back(row[2*r_run+3])
            diff[where_to_append].push_back(0)
            diff[where_to_append].push_back(r_H)
            diff[where_to_append].push_back(c_H)
            diff[where_to_append].push_back(row[2*r_run+4])
            where_to_append += 1
            diff_val += row[2*r_run+4]
            r_run += 1
            
        elif r_run >= cyround((row.size()-3)/2) and c_run < cyround((col.size()-3)/2):
            diff[where_to_append].push_back(col[2*c_run+3])
            diff[where_to_append].push_back(1)
            diff[where_to_append].push_back(r_H)
            diff[where_to_append].push_back(c_H)
            diff[where_to_append].push_back(col[2*c_run+4])
            where_to_append += 1
            diff_val += col[2*c_run+4]
            c_run += 1
            
        else:
            if r_run == cyround((row.size()-3)/2) and c_run == cyround((col.size()-3)/2): # Both indices worked through vectors, we're done!
                break
            if row[2*r_run+3] < col[2*c_run+3]: # Col does not occupy box number row[2*l+3]
                diff[where_to_append].push_back(row[2*r_run+3])
                diff[where_to_append].push_back(0)
                diff[where_to_append].push_back(r_H)
                diff[where_to_append].push_back(c_H)
                diff[where_to_append].push_back(row[2*r_run+4])
                where_to_append += 1
                diff_val += row[2*r_run+4]
                r_run += 1
            elif row[2*r_run+3] > col[2*c_run+3]: # Row does not occupy box number col[2*l+3]
                diff[where_to_append].push_back(col[2*c_run+3])
                diff[where_to_append].push_back(1)
                diff[where_to_append].push_back(r_H)
                diff[where_to_append].push_back(c_H)
                diff[where_to_append].push_back(col[2*c_run+4])
                where_to_append += 1
                diff_val += col[2*c_run+4]
                c_run += 1
            else: # This is the case row[2*r_run+3] == col[2*c_run+3], i.e. both col and row occupy box number row[2*l+3], but with different numbers !perhaps!
                if row[2*r_run+4] > col[2*c_run+4]:
                    diff[where_to_append].push_back(row[2*r_run+3])
                    diff[where_to_append].push_back(0)
                    diff[where_to_append].push_back(r_H)
                    diff[where_to_append].push_back(c_H)
                    diff[where_to_append].push_back(row[2*r_run+4])
                    where_to_append += 1
                    diff_val += row[2*r_run+4] - col[2*c_run+4]
                if row[2*r_run+4] < col[2*c_run+4]:
                    diff[where_to_append].push_back(col[2*c_run+3])
                    diff[where_to_append].push_back(1)
                    diff[where_to_append].push_back(r_H)
                    diff[where_to_append].push_back(c_H)
                    diff[where_to_append].push_back(col[2*c_run+4])
                    where_to_append += 1
                    diff_val += col[2*c_run+4] - row[2*r_run+4]
                r_run += 1
                c_run += 1
    diff[0].push_back(diff_val) # The index (first argument) specifies before which list position next argument shall be inserted (not very fast for lists)
    return diff
    
cdef float[:] H(int L, float[:] matrix_data, map[int,vector[int]] states_L, float[:,:,:,:,:,:] CG, float[:,:,:,:,:] W_1, float[:,:,:,:,:] W_2, float[:,:] U, float[:] w, int[:,:] bos_qindices, int[:,:] pos, int[:,:] section, bint with_W_1, bint with_W_2, bint with_mol, bint with_Froehlich, int l_max, float delta_k, int r_idx, int nlen, int ndiv) nogil:
    """
    Calculates the entries of H in one a part of block L
    Parameters:
    -------------
    L: block of interest
    matrix_data: will store our row, column, and entry info
    states_L: contains all states in block L
    CG: contains all relevant Clebsch-Gordan coefficients
    W_1: contains all relevant W_1 coefficients
    W_2: contains all relevant W_2 coefficients
    U: contains all relevant U coefficients
    w: contains all relevant w coefficients
    bos_qindices, pos, section: helps in navigating between states
    with_W_1, with_W_2, l_max, delta_k: physical parameters
    r_idx, nlen, ndiv: numerical parameters
    Returns:
    --------------
    matrix_data: now containing the entries of H, with row and column info"""
    
    # Thread-local definitions
    cdef int r_min = r_idx*nlen
    cdef int r_max
    if r_idx < ndiv:
        r_max = (r_idx+1)*nlen-1
    else:
        r_max = <int>states_L.size()-1
    cdef int r
    cdef int run
    cdef int run_1
    cdef int run_2
    cdef int c
    cdef int c_run_1
    cdef int c_run_2
    cdef float add = 0.0
    cdef map[int, vector[int]] diff_entry
    cdef map[int,vector[int]] r_rel
    cdef map[int,vector[int]] c_rel
    cdef int where_to_push = 0
    cdef int pos_run = 0
    cdef int section_run = 0
    cdef int box_number_dagger = 0
    cdef int box_number_no_dagger = 0
    cdef int mu = 0
    cdef int mu_row = 0
    cdef int mu_col = 0
    cdef int mu_1 = 0
    cdef int mu_2 = 0
    cdef int mu_1_prime = 0
    cdef int mu_2_prime = 0
    cdef int mu_dagger = 0
    cdef int mu_no_dagger = 0
    cdef int mu_dagger_first = 0
    cdef int mu_dagger_second = 0
    cdef int k_index = 0
    cdef int k_index_row = 0
    cdef int k_index_col = 0
    cdef int k_index_dagger = 0
    cdef int k_index_no_dagger = 0
    cdef int k_index_dagger_first = 0
    cdef int k_index_dagger_second = 0
    cdef int lamb = 0
    cdef int lambd = 0
    cdef int lambd_row = 0
    cdef int lambd_col = 0
    cdef int lambd_1_prime = 0
    cdef int lambd_2_prime = 0
    cdef int lambd_dagger = 0
    cdef int lambd_no_dagger = 0
    cdef int lambd_dagger_first = 0
    cdef int lambd_dagger_second = 0
    cdef int r_count = 0
    cdef int c_count = 0
    cdef int i
    cdef int j
    cdef int matrix_append = 0
    
    # Angulon Term: \sum_{k \lambda] V_{\lambda}(k)[b_{k\lambda 0}^+ + h.c.]
    if with_Froehlich:
        for r in range(r_min, r_max+1): 
            if states_L[r][2] > 0:
                pos_run = pos[L, states_L[r][2]-1]
                section_run = section[L, states_L[r][2]-1]
                for c in range(section_run): # This will be our col index
                    diff_entry = diff(states_L[r], states_L[c+pos_run], r, c+pos_run)
                    if states_L[r][0] == states_L[c+pos_run][0] and states_L[r][1] == states_L[c+pos_run][1] and diff_entry[0][0] == 1: # Last condition is redundant
                        box_number_dagger = diff_entry[1][0]
                        mu = bos_qindices[box_number_dagger][2]
                        k_index = bos_qindices[box_number_dagger][0]
                        lambd = bos_qindices[box_number_dagger][1]
                        add = 0.0
                        if mu == 0 and lambd <= l_max: # Only row mu should be zero for the b^+ term
                            add = add + <float>(sqrt((2*lambd+1)/(4*pi))*sqrt(delta_k)*U[k_index, lambd])*sqrt(diff_entry[1][4])
                        if add != 0.0:
                            matrix_data[matrix_append] = r # This is the b+ term
                            matrix_data[matrix_append+1] = c+pos_run
                            matrix_data[matrix_append+2] = add
                            matrix_data[matrix_append+3] = c+pos_run # This is the b term
                            matrix_data[matrix_append+4] = r
                            matrix_data[matrix_append+5] = add
                            matrix_append += 6
                            
    # Diagonal entries
    for r in range(r_min, r_max+1): 
        add = 0.0
        if with_mol:
            add += states_L[r][0]*(states_L[r][0]+1) # 0th order term
        for run in range(<int>cyround((states_L[r].size()-3)//2)):
            k_index = bos_qindices[states_L[r][2*run+3]][0]
            lambd = bos_qindices[states_L[r][2*run+3]][1]
            add += w[k_index]*states_L[r][2*run+4] # First order correction
        for run in range(<int>cyround((states_L[r].size()-3)//2)):
            lambd = bos_qindices[states_L[r][2*run+3]][1]
            if with_mol:
                add += lambd*(lambd+1)*states_L[r][2*run+4] # The simplest (b+b) contribution from B(\Lambda^2) (and thus does not change the dispersion qualitatively)
        for run in range(<int>cyround((states_L[r].size()-3)//2)):
            if states_L[r][2] != 0: # 0 spherical component of - 2* J'*\Lambda : -2* J'_{0}*\Lambda_{0}
                if with_mol:
                    add += -2*states_L[r][2*run+4]*states_L[r][1]*bos_qindices[states_L[r][2*run+3]][2] # boson_algebra*(-2*mol_term*bos_term)
        for run_1 in range(<int>cyround((states_L[r].size()-3)//2)): 
            for run_2 in range(<int>cyround((states_L[r].size()-3)//2)): # note that run_1 as the lower bound avoids double counts
                mu_1 = bos_qindices[states_L[r][2*run_1+3]][2]
                mu_2 = bos_qindices[states_L[r][2*run_2+3]][2]
                if run_1 == run_2 and states_L[r][2*run_1+4] >= 2: # here we deal with b+b+bb contribution from B(\Lambda^2), part 1
                    if with_mol:
                        add += mu_1*mu_2*states_L[r][2*run_1+4]*(states_L[r][2*run_1+4]-1) 
        for run_1 in range(<int>cyround((states_L[r].size()-3)//2)): 
            for run_2 in range(<int>cyround((states_L[r].size()-3)//2)):
                if run_1 != run_2: # Here we deal with b+b+bb contribution from B(\Lambda^2), part 1
                    mu_1 = bos_qindices[states_L[r][2*run_1+3]][2]
                    mu_2 = bos_qindices[states_L[r][2*run_2+3]][2]
                    if with_mol:
                        add += mu_1*mu_2*states_L[r][2*run_1+4]*states_L[r][2*run_2+4]  
                if run_1 != run_2 and bos_qindices[states_L[r][2*run_1+3]][0] == bos_qindices[states_L[r][2*run_2+3]][0] and bos_qindices[states_L[r][2*run_1+3]][1] == bos_qindices[states_L[r][2*run_2+3]][1] and bos_qindices[states_L[r][2*run_1+3]][2] == bos_qindices[states_L[r][2*run_2+3]][2]+1:  # here we deal with b+b+bb contribution from B(\Lambda^2), part 2/2
                    lambd_1_prime = bos_qindices[states_L[r][2*run_2+3]][1]
                    lambd_2_prime = bos_qindices[states_L[r][2*run_1+3]][1]
                    mu_1_prime = bos_qindices[states_L[r][2*run_2+3]][2]
                    mu_2_prime = bos_qindices[states_L[r][2*run_1+3]][2]
                    if with_mol:
                        add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime+1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime-1))*states_L[r][2*run_1+4]*states_L[r][2*run_2+4]
                if run_1 != run_2 and bos_qindices[states_L[r][2*run_1+3]][0] == bos_qindices[states_L[r][2*run_2+3]][0] and bos_qindices[states_L[r][2*run_1+3]][1] == bos_qindices[states_L[r][2*run_2+3]][1] and bos_qindices[states_L[r][2*run_1+3]][2] == bos_qindices[states_L[r][2*run_2+3]][2]-1:  # here we deal with b+b+bb contribution from B(\Lambda^2), part 2/2
                    lambd_1_prime = bos_qindices[states_L[r][2*run_1+3]][1]
                    lambd_2_prime = bos_qindices[states_L[r][2*run_2+3]][1]
                    mu_1_prime = bos_qindices[states_L[r][2*run_1+3]][2]
                    mu_2_prime = bos_qindices[states_L[r][2*run_2+3]][2]
                    if with_mol:
                        add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime+1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime-1))*states_L[r][2*run_1+4]*states_L[r][2*run_2+4]
        
        if with_W_1 == True:
            for run_1 in range(<int>cyround((states_L[r].size()-3)//2)): 
                for run_2 in range(<int>cyround((states_L[r].size()-3)//2)):
                    if run_1 == run_2 and states_L[r][2] != 0:   # Here we take care of W_1 term in the case of r = c
                        k_index = bos_qindices[states_L[r][2*run_1+3]][0]
                        lambd = bos_qindices[states_L[r][2*run_1+3]][1]
                        mu_1 = bos_qindices[states_L[r][2*run_1+3]][2]
                        for lamb in range(l_max+1):
                            add += W_1[k_index, k_index, lambd, lamb, lambd]*CG[lambd, mu_1 + l_max, lamb, l_max, lambd, mu_1 + l_max]*states_L[r][2*run_1+4]*delta_k

        if add != 0.0:
            matrix_data[matrix_append] = r 
            matrix_data[matrix_append+1] = r
            matrix_data[matrix_append+2] = add
            matrix_append += 3
    
    # Terms b+b etc.: \sum_{k,q,\lambda, l, l', \nu}1_W_{l'\lambda}^l(k,q)C^{l\nu}_{l'\nu\lambda 0}b^+_{kl\nu}b_{ql'\nu} + B(J-Lambda)**2 + \sum_{k \lambda \mu} w_k b^+_{k \lambda \mu} b_{k \lambda \mu}                            
    for r in range(r_min, r_max+1): 
        if states_L[r][2] > 0:
            pos_run = pos[L, states_L[r][2]]
            section_run = section[L, states_L[r][2]]
            for c in range(pos_run, pos_run+section_run):
                add = 0.0
                diff_entry = diff(states_L[r], states_L[c], r, c)
                if diff_entry[0][0] == 2: # Aleady excludes diagonal matrix elements    
                    if states_L[r][1] == states_L[c][1]: 
                        if with_W_1 == True:
                            if diff_entry[1][1] == 0: # This condition or the next is automatically true if diff_val == 2, but still without this part, you would not know which box corresponds to dagger and which one to no dagger
                                box_number_dagger = diff_entry[1][0] # You have two boxes: in your head, you label, the left one, box_of_larger_occ_1 = k l nu (more heavily occupied by i=row), and, the right one, box_of_larger_occ_2 = q l' nu (more heavily occupied by p=col)
                                box_number_no_dagger = diff_entry[2][0]
                                mu_dagger = bos_qindices[box_number_dagger][2]
                                mu_no_dagger = bos_qindices[box_number_no_dagger][2]
                                if mu_dagger == mu_no_dagger:
                                    k_index_dagger = bos_qindices[box_number_dagger][0]
                                    k_index_no_dagger = bos_qindices[box_number_no_dagger][0]
                                    lambd_dagger = bos_qindices[box_number_dagger][1]
                                    lambd_no_dagger = bos_qindices[box_number_no_dagger][1]
                                    for lamb in range(0, l_max+1):
                                        add += W_1[k_index_dagger, k_index_no_dagger, lambd_no_dagger, lamb, lambd_dagger]*CG[lambd_no_dagger, mu_dagger + l_max, lamb, l_max, lambd_dagger, mu_dagger + l_max]*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])*delta_k
                                        
                            if diff_entry[1][1] == 1:
                                box_number_dagger = diff_entry[2][0] # You have two boxes: in your head, you label, the left one, box_of_larger_occ_1 = q l' nu (more heavily occupied by p=col), and, the right one, box_of_larger_occ_2 = k l nu (more heavily occupied by i=row)
                                box_number_no_dagger = diff_entry[1][0]
                                mu_dagger = bos_qindices[box_number_dagger][2]
                                mu_no_dagger = bos_qindices[box_number_no_dagger][2]
                                if mu_dagger == mu_no_dagger:
                                    k_index_dagger = bos_qindices[box_number_dagger][0]
                                    k_index_no_dagger = bos_qindices[box_number_no_dagger][0]
                                    lambd_dagger = bos_qindices[box_number_dagger][1]
                                    lambd_no_dagger = bos_qindices[box_number_no_dagger][1]
                                    for lamb in range(l_max+1):
                                        add += W_1[k_index_dagger, k_index_no_dagger, lambd_no_dagger, lamb, lambd_dagger]*CG[lambd_no_dagger, mu_dagger + l_max, lamb, l_max, lambd_dagger, mu_dagger + l_max]*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])*delta_k
                        else:
                            pass
                                        
                    elif states_L[r][1] == states_L[c][1]+1: # -1+1 spherical component of - 2* J'*\Lambda: -2* J'_{-1}*\Lambda_{1}; it is essentially a b^+b term
                        if diff_entry[1][1] == 0:
                            lambd_row = bos_qindices[diff_entry[1][0]][1]
                            lambd_col = bos_qindices[diff_entry[2][0]][1]
                            k_index_row = bos_qindices[diff_entry[1][0]][0]
                            k_index_col = bos_qindices[diff_entry[2][0]][0]
                            mu_row = bos_qindices[diff_entry[1][0]][2]
                            mu_col = bos_qindices[diff_entry[2][0]][2]
                            if lambd_row == lambd_col and k_index_row == k_index_col and mu_row == mu_col+1:
                                if with_mol:
                                    add += -2*1/2*sqrt(lambd_row*(lambd_row+1)-mu_col*(mu_col+1))*sqrt(states_L[r][0]*(states_L[r][0]+1)-states_L[c][1]*(states_L[c][1]+1))*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])
                        if diff_entry[1][1] == 1:
                            lambd_row = bos_qindices[diff_entry[2][0]][1]
                            lambd_col = bos_qindices[diff_entry[1][0]][1]
                            k_index_row = bos_qindices[diff_entry[2][0]][0]
                            k_index_col = bos_qindices[diff_entry[1][0]][0]
                            mu_row = bos_qindices[diff_entry[2][0]][2]
                            mu_col = bos_qindices[diff_entry[1][0]][2]
                            if lambd_row == lambd_col and k_index_row == k_index_col and mu_row == mu_col+1:
                                if with_mol:
                                    add += -2*1/2*sqrt(lambd_row*(lambd_row+1)-mu_col*(mu_col+1))*sqrt(states_L[r][0]*(states_L[r][0]+1)-states_L[c][1]*(states_L[c][1]+1))*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])
                        
                    else:  # +1-1 spherical component of - 2* J'*\Lambda: -2* J'_{1}*\Lambda_{-1}; it is essentially a b^+b term
                        if diff_entry[1][1] == 0: # This condition or the next is automatically true if diff_val == 2, but still without this part, you would not know which box corresponds to dagger and which one to no dagger
                            lambd_row = bos_qindices[diff_entry[1][0]][1]
                            lambd_col = bos_qindices[diff_entry[2][0]][1]
                            k_index_row = bos_qindices[diff_entry[1][0]][0]
                            k_index_col = bos_qindices[diff_entry[2][0]][0]
                            mu_row = bos_qindices[diff_entry[1][0]][2]
                            mu_col = bos_qindices[diff_entry[2][0]][2]
                            if lambd_row == lambd_col and k_index_row == k_index_col and mu_row == mu_col-1:
                                if with_mol:
                                    add += -2*1/2*sqrt(lambd_row*(lambd_row+1)-mu_col*(mu_col-1))*sqrt(states_L[r][0]*(states_L[r][0]+1)-states_L[c][1]*(states_L[c][1]-1))*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])
                        if diff_entry[1][1] == 1:
                            lambd_row = bos_qindices[diff_entry[2][0]][1]
                            lambd_col = bos_qindices[diff_entry[1][0]][1]
                            k_index_row = bos_qindices[diff_entry[2][0]][0]
                            k_index_col = bos_qindices[diff_entry[1][0]][0]
                            mu_row = bos_qindices[diff_entry[2][0]][2]
                            mu_col = bos_qindices[diff_entry[1][0]][2]
                            if lambd_row == lambd_col and k_index_row == k_index_col and mu_row == mu_col-1:
                                if with_mol:
                                    add += -2*1/2*sqrt(lambd_row*(lambd_row+1)-mu_col*(mu_col-1))*sqrt(states_L[r][0]*(states_L[r][0]+1)-states_L[c][1]*(states_L[c][1]-1))*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])
                
                if diff_entry[0][0] == 4 and with_mol: # Here we deal with the complicated (b+b+bb) part of B(\Lambda^2)
                    r_rel.clear()
                    c_rel.clear()
                    r_count = 0
                    c_count = 0
                    for i in range(<int>(diff_entry.size()-1)):
                        if diff_entry[i+1][1] == 0:
                            for j in range(<int>(diff_entry[i+1].size())):
                                r_rel[r_count].push_back(diff_entry[i+1][j])
                            r_count += 1
                        else:
                            for j in range(<int>(diff_entry[i+1].size())):
                                c_rel[c_count].push_back(diff_entry[i+1][j])
                            c_count += 1
                    for r_run_1 in range(<int>(r_rel.size())): # Using list comprehension to extract those boxes that are striclty more heavily occupied by r than c
                        for r_run_2 in range(<int>(r_rel.size())): 
                            for c_run_1 in range(<int>(c_rel.size())):
                                for c_run_2 in range(<int>(c_rel.size())):
                                    lambd_1_prime = bos_qindices[c_rel[c_run_1][0]][1]
                                    lambd_2_prime = bos_qindices[c_rel[c_run_2][0]][1]
                                    mu_1_prime = bos_qindices[c_rel[c_run_1][0]][2]
                                    mu_2_prime = bos_qindices[c_rel[c_run_2][0]][2]
                                    
                                    if r_run_1 == r_run_2 and c_run_1 != c_run_2 and r_rel.size() == 1: 
                                        if bos_qindices[r_rel[r_run_1][0]][0] == bos_qindices[c_rel[c_run_1][0]][0] and bos_qindices[r_rel[r_run_1][0]][1] == bos_qindices[c_rel[c_run_1][0]][1] and bos_qindices[r_rel[r_run_1][0]][2] == bos_qindices[c_rel[c_run_1][0]][2]+1 and bos_qindices[r_rel[r_run_2][0]][0] == bos_qindices[c_rel[c_run_2][0]][0] and bos_qindices[r_rel[r_run_2][0]][1] == bos_qindices[c_rel[c_run_2][0]][1] and bos_qindices[r_rel[r_run_2][0]][2] == bos_qindices[c_rel[c_run_2][0]][2]-1:
                                            add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime+1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime-1))*sqrt(r_rel[r_run_1][4])*sqrt(r_rel[r_run_1][4]-1)*sqrt(c_rel[c_run_1][4])*sqrt(c_rel[c_run_2][4])
                                        if bos_qindices[r_rel[r_run_1][0]][0] == bos_qindices[c_rel[c_run_1][0]][0] and bos_qindices[r_rel[r_run_1][0]][1] == bos_qindices[c_rel[c_run_1][0]][1] and bos_qindices[r_rel[r_run_1][0]][2] == bos_qindices[c_rel[c_run_1][0]][2]-1 and bos_qindices[r_rel[r_run_2][0]][0] == bos_qindices[c_rel[c_run_2][0]][0] and bos_qindices[r_rel[r_run_2][0]][1] == bos_qindices[c_rel[c_run_2][0]][1] and bos_qindices[r_rel[r_run_2][0]][2] == bos_qindices[c_rel[c_run_2][0]][2]+1:
                                            add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime-1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime+1))*sqrt(r_rel[r_run_1][4])*sqrt(r_rel[r_run_1][4]-1)*sqrt(c_rel[c_run_1][4])*sqrt(c_rel[c_run_2][4])
                                    if r_run_1 != r_run_2 and c_run_1 == c_run_2 and c_rel.size() == 1:
                                        if bos_qindices[r_rel[r_run_1][0]][0] == bos_qindices[c_rel[c_run_1][0]][0] and bos_qindices[r_rel[r_run_1][0]][1] == bos_qindices[c_rel[c_run_1][0]][1] and bos_qindices[r_rel[r_run_1][0]][2] == bos_qindices[c_rel[c_run_1][0]][2]+1 and bos_qindices[r_rel[r_run_2][0]][0] == bos_qindices[c_rel[c_run_2][0]][0] and bos_qindices[r_rel[r_run_2][0]][1] == bos_qindices[c_rel[c_run_2][0]][1] and bos_qindices[r_rel[r_run_2][0]][2] == bos_qindices[c_rel[c_run_2][0]][2]-1:
                                            add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime+1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime-1))*sqrt(c_rel[c_run_1][4])*sqrt(c_rel[c_run_1][4]-1)*sqrt(r_rel[r_run_1][4])*sqrt(r_rel[r_run_2][4])
                                        if bos_qindices[r_rel[r_run_1][0]][0] == bos_qindices[c_rel[c_run_1][0]][0] and bos_qindices[r_rel[r_run_1][0]][1] == bos_qindices[c_rel[c_run_1][0]][1] and bos_qindices[r_rel[r_run_1][0]][2] == bos_qindices[c_rel[c_run_1][0]][2]-1 and bos_qindices[r_rel[r_run_2][0]][0] == bos_qindices[c_rel[c_run_2][0]][0] and bos_qindices[r_rel[r_run_2][0]][1] == bos_qindices[c_rel[c_run_2][0]][1] and bos_qindices[r_rel[r_run_2][0]][2] == bos_qindices[c_rel[c_run_2][0]][2]+1:
                                            add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime-1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime+1))*sqrt(c_rel[c_run_1][4])*sqrt(c_rel[c_run_1][4]-1)*sqrt(r_rel[r_run_1][4])*sqrt(r_rel[r_run_2][4])
                                    if r_run_1 != r_run_2 and c_run_1 != c_run_2:
                                        if bos_qindices[r_rel[r_run_1][0]][0] == bos_qindices[c_rel[c_run_1][0]][0] and bos_qindices[r_rel[r_run_1][0]][1] == bos_qindices[c_rel[c_run_1][0]][1] and bos_qindices[r_rel[r_run_1][0]][2] == bos_qindices[c_rel[c_run_1][0]][2]+1 and bos_qindices[r_rel[r_run_2][0]][0] == bos_qindices[c_rel[c_run_2][0]][0] and bos_qindices[r_rel[r_run_2][0]][1] == bos_qindices[c_rel[c_run_2][0]][1] and bos_qindices[r_rel[r_run_2][0]][2] == bos_qindices[c_rel[c_run_2][0]][2]-1:
                                            add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime+1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime-1))*sqrt(c_rel[c_run_1][4])*sqrt(c_rel[c_run_2][4])*sqrt(r_rel[r_run_1][4])*sqrt(r_rel[r_run_2][4])
                                        if bos_qindices[r_rel[r_run_1][0]][0] == bos_qindices[c_rel[c_run_1][0]][0] and bos_qindices[r_rel[r_run_1][0]][1] == bos_qindices[c_rel[c_run_1][0]][1] and bos_qindices[r_rel[r_run_1][0]][2] == bos_qindices[c_rel[c_run_1][0]][2]-1 and bos_qindices[r_rel[r_run_2][0]][0] == bos_qindices[c_rel[c_run_2][0]][0] and bos_qindices[r_rel[r_run_2][0]][1] == bos_qindices[c_rel[c_run_2][0]][1] and bos_qindices[r_rel[r_run_2][0]][2] == bos_qindices[c_rel[c_run_2][0]][2]+1:
                                            add += 1/2*sqrt(lambd_1_prime*(lambd_1_prime+1)-mu_1_prime*(mu_1_prime-1))*sqrt(lambd_2_prime*(lambd_2_prime+1)-mu_2_prime*(mu_2_prime+1))*sqrt(c_rel[c_run_1][4])*sqrt(c_rel[c_run_2][4])*sqrt(r_rel[r_run_1][4])*sqrt(r_rel[r_run_2][4])
                
                if add != 0.0:
                    matrix_data[matrix_append] = r
                    matrix_data[matrix_append+1] = c
                    matrix_data[matrix_append+2] = add
                    matrix_append += 3

    # Term b+b+: \sum_{k,q,\lambda, l, l', \nu} 2W_{l'\lambda}^l(q,k)(-1)^{\nu}C^{l\nu}_{l'\nu\lambda 0} [b^+_{ql\nu}b^+_{kl' -\nu} + h.c.]
    if with_W_2 == True:
        for r in range(r_min, r_max+1): 
            if states_L[r][2] > 1:
                pos_run = pos[L, states_L[r][2]-2]
                section_run = section[L, states_L[r][2]-2]
                for c in range(pos_run, pos_run+section_run): # This will be the column now
                    diff_entry = diff(states_L[r], states_L[c], r, c)
                    if states_L[r][0] == states_L[c][0] and states_L[r][1] == states_L[c][1] and diff_entry[0][0] == 2:
                        box_number_dagger_first = diff_entry[1][0] # One can also do this the other way cyround, consistency is important though
                        if (diff_entry.size() - 1) == 1:
                            box_number_dagger_second = diff_entry[1][0]
                        else:
                            box_number_dagger_second = diff_entry[2][0]
                        mu_dagger_first = bos_qindices[box_number_dagger_first][2]
                        mu_dagger_second = bos_qindices[box_number_dagger_second][2]
                        k_index_dagger_first = bos_qindices[box_number_dagger_first][0]
                        k_index_dagger_second = bos_qindices[box_number_dagger_second][0]
                        lambd_dagger_first = bos_qindices[box_number_dagger_first][1]
                        lambd_dagger_second = bos_qindices[box_number_dagger_second][1]
                        add = 0.0
                        if mu_dagger_first == -mu_dagger_second:
                            for lamb in range(l_max+1):
                                if box_number_dagger_first != box_number_dagger_second:
                                    add += W_2[k_index_dagger_first, k_index_dagger_second, lambd_dagger_second, lamb, lambd_dagger_first]*CG[lambd_dagger_second, mu_dagger_first + l_max, lamb, l_max, lambd_dagger_first, mu_dagger_first + l_max]*sqrt(diff_entry[1][4])*sqrt(diff_entry[2][4])*pow(-1, mu_dagger_first)*delta_k
                                else:
                                    add += W_2[k_index_dagger_first, k_index_dagger_second, lambd_dagger_second, lamb, lambd_dagger_first]*CG[lambd_dagger_second, mu_dagger_first + l_max, lamb, l_max, lambd_dagger_first, mu_dagger_first + l_max]*sqrt(diff_entry[1][4]-1)*sqrt(diff_entry[1][4])*pow(-1, mu_dagger_first)*delta_k
                            if add != 0.0:
                                matrix_data[matrix_append] = r
                                matrix_data[matrix_append+1] = c
                                matrix_data[matrix_append+2] = add
                                matrix_data[matrix_append+3] = c # This accounts for the h.c. part
                                matrix_data[matrix_append+4] = r
                                matrix_data[matrix_append+5] = add
                                matrix_append += 6

    return matrix_data

def corr_k_tile(float[:] state, int L):
        
        # input:  state: eigenstate of block L of which the corr_function shall be calculated 
        #         k, lambd, mu, k_prime, lambd_prime, mu_prime: the latter shall be calculated in the single-particle state |k, lambd, mu> and |k_prime, lambd_prime, mu_prime> (k, k_prime are not the real momenta, just index)
        #         L: the block from which state is drawn
        # output: real corr_function of state calculated in single-particle state |k, lambd, mu> and |k_prime, lambd_prime, mu_prime>
        
        # Loading states_L_tmp into the map states_L
        states_L_tmp = config.states_all[L] # List
        cdef map[int,vector[int]] states_L
        for i in range(len(states_L_tmp)):
            states_L[i].push_back(states_L_tmp[i][0])
            states_L[i].push_back(states_L_tmp[i][1])
            states_L[i].push_back(states_L_tmp[i][2])
            for j in range((len(states_L_tmp[i])-3)//2):
                states_L[i].push_back(states_L_tmp[i][3+2*j])
                states_L[i].push_back(states_L_tmp[i][3+2*j+1])
        cdef float k_max = config.k_max[L]
        cdef float k_min = config.k_min
        cdef float delta_k = config.delta_k
        cdef int l_max = config.l_max
        cdef float[:,:,:,:,:,:] corr_k_arr = np.zeros((<int>(cyround((k_max-k_min)/delta_k))+1, <int>(cyround((k_max-k_min)/delta_k))+1, l_max+1, 2*l_max+1, l_max+1, 2*l_max+1), dtype = np.float32)
        cdef int k_maxindex = <int>(cyround((k_max-k_min)/delta_k))+1
        cdef int k
        for k in prange(k_maxindex, schedule = "dynamic", nogil = True):
            corr_k_arr[k] = corr_k_parallel(corr_k_arr[k], state, states_L, L, k_max, k_min, delta_k, l_max, k)
        return corr_k_arr.base

cdef float[:,:,:,:,:] corr_k_parallel(float[:,:,:,:,:] corr_k_arr, float[:] state, map[int,vector[int]] states_L, int L, float k_max, float k_min, float delta_k, int l_max, int k) nogil:
    cdef int N_boxes = 0
    cdef int N_lmuboxes = 0
    cdef int lambd
    cdef int mu
    cdef int k_prime
    cdef int lambd_prime
    cdef int mu_prime
    cdef int l
    cdef int run_row
    cdef int run_col
    cdef int row_bos_box
    cdef int col_bos_box
    for l in range(l_max+1):
        N_boxes += <int>(cyround((k_max-k_min)/delta_k)+1)*(2*l+1)
    for l in range(l_max+1):
        N_lmuboxes += 2*l+1
    for run_row in range(states_L.size()):
        for run_col in range(states_L.size()):
            for row_bos_box in range(N_boxes):
                if row_bos_box // N_lmuboxes == k: # Only if momentum == k
                    for col_bos_box in range(N_boxes):
                        if search_in_state(states_L[run_row], row_bos_box, L, k_max, k_min, delta_k, l_max) and search_in_state(states_L[run_col], col_bos_box, L, k_max, k_min, delta_k, l_max):
                            lambd = lindex(row_bos_box, l_max)
                            mu = muindex(row_bos_box, lambd, l_max)
                            k_prime = kindex(col_bos_box, l_max)
                            lambd_prime = lindex(col_bos_box, l_max)
                            mu_prime = muindex(col_bos_box, lambd_prime, l_max)
                            corr_k_arr[k_prime, lambd, mu + lambd, lambd_prime, mu_prime+lambd_prime] += state[run_row]*state[run_col]*sqrt(nb_bos_in_box(states_L[run_row], row_bos_box, L, k_max, k_min, delta_k, l_max)*nb_bos_in_box(states_L[run_col], col_bos_box, L, k_max, k_min, delta_k, l_max))
    return corr_k_arr
           

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint search_in_state(vector[int] state, int bos_box, int L, float k_max, float k_min, float delta_k, int l_max) nogil:
    cdef int N_boxes = 0
    cdef int box_run
    cdef int l
    if state.size() > 2: # We have > 0 bosons in "state"
        for l in range(l_max+1):
            N_boxes += <int>(cyround((k_max-k_min)/delta_k)+1)*(2*l+1)
        for box_run in range(<int>(cyround((state.size()-3)/2))):
            if state[2*box_run+3] == bos_box:
                return True
        return False
    else:
        return False
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int nb_bos_in_box(vector[int] state, int bos_box, int L, float k_max, float k_min, float delta_k, int l_max) nogil:
    cdef int N_boxes = 0
    cdef int box_run = 0
    cdef int l = 0
    if state.size() > 2: # We have > 0 bosons in "state"
        N_boxes = 0
        for l in range(l_max+1):
            N_boxes += <int>(cyround((k_max-k_min)/delta_k)+1)*(2*l+1)
        for box_run in range(<int>(cyround((state.size()-3)/2))):
            if state[2*box_run+3] == bos_box:
                return state[2*box_run+4]
        return 0
    else: # Else part should not be invoked
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int kindex(int bos_box, int l_max) nogil:
    cdef int N_lmuboxes = 0
    cdef int l
    for l in range(l_max+1):
        N_lmuboxes += 2*l+1
    return bos_box // N_lmuboxes

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int lindex(int bos_box, int l_max) nogil:
    cdef int N_lmuboxes = 0
    cdef int l
    for l in range(l_max+1):
        N_lmuboxes += 2*l+1
    cdef int k_rest = bos_box % N_lmuboxes
    cdef int l_found
    for l in range(l_max+1):
        k_rest -= 2*l+1
        if k_rest < 0:
            l_found = l
            break
    return l_found

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int muindex(int bos_box, int l_found, int l_max) nogil:
    cdef int N_lmuboxes = 0
    cdef int l
    for l in range(l_max+1):
        N_lmuboxes += 2*l+1
    cdef int k_rest = bos_box % N_lmuboxes
    cdef int l_rest
    if l_found > 0:
        l_rest = k_rest - (2*(l_found-1)+1)
    else:
        l_rest = k_rest
    return l_rest-l_found
    


def corr_r_tile(state, L, r):
    
    # parallelizing corr_function_r calculations
    # input: r is an index in order to assign it to corr_r in the main file
      
    cdef float real = 0.0
    cdef float imag = 0.0
    cdef Py_ssize_t lambd
    cdef Py_ssize_t mu
    cdef Py_ssize_t lambd_prime
    cdef Py_ssize_t mu_prime
    cdef Py_ssize_t k
    cdef Py_ssize_t k_prime
    corr_r_arr = np.zeros((config.l_max+1, 2*config.l_max+1, config.l_max+1, 2*config.l_max+1), dtype=complex)
    for lambd in range(0, config.l_max+1):
        for mu in range(-lambd, lambd+1):
            for lambd_prime in range(0, config.l_max+1):
                for mu_prime in range(-lambd_prime, lambd_prime+1):
                    corr_function = complex(real, imag)
                    for k in range(0, <int>(cyround((config.k_max[L]-config.k_min)/config.delta_k))+1):
                        for k_prime in range(0, <int>(cyround((config.k_max[L]-config.k_min)/config.delta_k))+1):
                            corr_function += complex(0.0, 1.0)**(lambd-lambd_prime)*(2/np.pi)*(r*config.delta_r+config.delta_r)**2*config.delta_k*(k*config.delta_k+config.k_min)*(k_prime*config.delta_k+config.k_min)*sp.spherical_jn(lambd, (k*config.delta_k+config.k_min)*(r*config.delta_r+config.delta_r))*sp.spherical_jn(lambd_prime, (k_prime*config.delta_k+config.k_min)*(r*config.delta_r+config.delta_r))*config.corr_k[k, k_prime, lambd, mu+lambd, lambd_prime, mu_prime+lambd_prime] 
                            # Note that have config.delta_k, not config.delta_k**2, since the b operators loose their dimensions upon discretization
                    corr_r_arr[lambd, mu + lambd, lambd_prime, mu_prime+lambd_prime] = corr_function
                    
    return (corr_r_arr, r)

                    
def dens_tile(state, L, r):
    
    # r is index
    
    def phonon_density(state, L, r, theta_r, phi_r):
        
        # r is index, theta_r and phi_r not
        # input:  state: eigenstate of block L of which the phonon density shall be calculated 
        #         r, theta_r, phi_r: the latter shall be calculated in the single-particle state |r, theta_r, phi_r> (k is not the real momentum, just index)
        #         L: the block from which state is drawn
        # output: real (!check!) phonon density of state calculated in single-particle state |r, theta_r, phi_r>, i.e. at position \mathbf(r)
        
        cdef float real = 0.0
        cdef float imag = 0.0
        phononic_density = complex(real, imag)
        for lambd in range(0, config.l_max+1):
            for mu in range(-lambd, lambd+1):
                for lambd_prime in range(0, config.l_max+1):
                    for mu_prime in range(-lambd_prime, lambd_prime+1):
                        phononic_density += (1/((r*config.delta_r+config.delta_r)**2))*complex(0.0, 1.0)**(-lambd+lambd_prime)*sp.sph_harm(mu, lambd, phi_r, theta_r)*np.conj(sp.sph_harm(mu_prime, lambd_prime, phi_r, theta_r))*config.corr_r[r, lambd, mu + lambd, lambd_prime, mu_prime + lambd_prime]  # the access values result from its initialization
        return phononic_density
      
        
    theta_vec = np.arange(0, <int>(cyround((config.theta_max)/config.delta_theta))+1)
    phon_dens_vec = np.zeros((len(theta_vec),), dtype=complex)
    for theta in theta_vec:
        phon_dens_vec[theta] = phonon_density(state, L, r, theta*config.delta_theta, 0.0)
    return (phon_dens_vec, r)