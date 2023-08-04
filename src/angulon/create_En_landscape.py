#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:03:00 2019

@author: tibor
"""

import numpy as np
from class_ham import ham
import os
from scipy import sparse
from print_msg import print_status
from accept_reject import acceptEig
import config
from utilities import getH
from cython_functions import cython_eigsh

def getnShareState(C, L):
    """ Returns array of fractions of n-contributions, n in [-L_block, L_block]"""
    return np.array([C[n, :].sum()/C.sum() for n in range(2*L+1)])

def traceG(E, overlaps, eigenvalues):
    tr = 0.0+0.0j
    for run in range(len(overlaps)):
        tr += np.dot(overlaps[run], overlaps[run])/(E-eigenvalues[run]-config.eps) # Only overlap with non-interacting state.
    return tr

def createEnLandscape(start_time):
    """
    Creates the spectrum/spectral function in the energy vs density landscape
    Parameters are set in the config.py file
    Effect:
    --------------
    plot_matrix.txt: Contains spectral function
    spectrum_x.txt: Contains density values for spectrum plot
    spectrum_y.txt: Contains spectrum values for spectrum plot"""
    
    log_n_c_tilde_vector = np.linspace(config.n_min, config.n_max, (abs(config.n_max-config.n_min)+1)+(config.n_c_div-1)*abs(config.n_max-config.n_min))
    n_c_vector = np.exp(log_n_c_tilde_vector)*(config.m_b*config.B)**(-3/2)
    E_vector = np.linspace(config.E_max, config.E_min, (abs(config.E_max-config.E_min)+1)+(config.E_div-1)*abs(config.E_max-config.E_min))  # such that [::-1] will not be necessary
    step_E = abs(config.E_max-config.E_min)/(abs(config.E_max-config.E_min)*config.E_div)
    
    N_block_list = []
    discard_eigval = [[[] for L in range(config.L_max+1)] for i in range(len(n_c_vector))]
    
    # Check whether arrays exist already
    for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
        if not os.path.exists('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False")):
            # Saving to disc
            sp_func = np.zeros((config.L_max+1-config.L_max*config.L_max_only, (abs(config.E_max-config.E_min)+1)+(config.E_div-1)*abs(config.E_max-config.E_min),(abs(config.n_max-config.n_min)+1)+(config.n_c_div-1)*abs(config.n_max-config.n_min)))
            np.savetxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), sp_func[L_ind], fmt='%1.7e')
            y = [[] for _ in range(config.L_max+1-config.L_max*config.L_max_only)]
            x = [[] for _ in range(config.L_max+1-config.L_max*config.L_max_only)]
            np.savetxt('{}/spectrum_x_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), x[L_ind], fmt='%1.7e')
            np.savetxt('{}/spectrum_y_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), y[L_ind], fmt='%1.7e')
        
    for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
        
        sp_func_L = np.loadtxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
        x_L = list(np.loadtxt('{}/spectrum_x_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False")))
        y_L = list(np.loadtxt('{}/spectrum_y_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False")))
        # Check whether L-block has finished already
        if np.any(sp_func_L) == False: # whether any of them are non-zero
            nb_L_done = False
        else:
            nb_L_done = True
            
        if nb_L_done == True:
            continue
        else:
        
            print_status(start_time, "L = {0}. Precalculate U, W, state space...".format(L))
            config.makeGlobalDeltaK(config.delta_k)
            config.makeGlobalDensity(n_c_vector[0])
            ED = ham(config.u_0, config.u_1, config.r_0, config.r_1, config.m_b, n_c_vector[0], config.k_max[L], config.k_min, config.delta_k, config.l_max, config.L_max, config.n_ph_max, config.B, config.a_bb, config.mod_what)
            ED.getStatesAndFunctions()
            
            ED.createBosStateList()
            ED.createHashTable()
            config.makeGlobalStates(ED.states_all)
            config.makeGlobalPos(ED.states_pos)
            config.makeGlobalSection(ED.states_section)
            config.makeGlobalCG(ED.CG)
            config.makeGlobalBosQIndices(ED.bos_qindices)
            config.makeGlobalU(ED.U)
            config.makeGlobalW1(ED.W_1)
            config.makeGlobalW2(ED.W_2)
            config.makeGlobalw(ED.w)
        
            for idx, n_c in enumerate(n_c_vector[:]):
                print_status(start_time, "L = {0}, bath density is {1}".format(L, np.log(n_c)))
                # Update parameters
                ED.changeDensity(n_c) # To avoid recalculating w etc.  
                config.makeGlobalDensity(n_c)
                config.makeGlobalU(ED.U)
                config.makeGlobalW1(ED.W_1)
                config.makeGlobalW2(ED.W_2)
                config.makeGlobalw(ED.w)
            
                # Pooled multithreaded job submission
                data_tot, data_row_tot, data_col_tot, N_block = getH(L)
                print_status(start_time, "Filled sparse matrix of size ({}, {})".format(N_block, N_block))
                # Create sparse matrix
                H = sparse.csr_matrix((data_tot, (data_row_tot, data_col_tot)), shape=(N_block, N_block))
                if n_c == n_c_vector[0]:
                    N_block_list.append(N_block)
                
                # Solve eigenvalue problem
                eigenvalues, eigenvectors = cython_eigsh(np.float32(H.todense()))
                print_status(start_time, "Lowest eigenvalues read {}".format(eigenvalues[:4]))
                if config.selected_eigs[L] != None:
                    for eig in config.selected_eigs[L]:
                        if acceptEig(eigenvalues[eig]/config.B, eig, L, discard_eigval, np.log(n_c), idx):
                            y_L.append(eigenvalues[eig]/config.B)
                else:
                    for eig in range(len(eigenvalues)):
                        y_L.append(eigenvalues[eig]/config.B)
                    
                # Spectral function calculations
                if config.selected_eigs[L] != None:
                    for E in E_vector:
                        trace = traceG(config.E_min+abs(E-config.E_max), eigenvectors[0,:], eigenvalues) # eigenvectors[0,:] gives overlap entries
                        sp_func_L[int(round(abs(E-config.E_min)/step_E)), idx] += trace.imag
                else:
                    
                    for E in E_vector:
                        trace = traceG(config.E_min+abs(E-config.E_max), eigenvectors[0,:], eigenvalues) # eigenvectors[0,:] gives overlap entries
                        sp_func_L[int(round(abs(E-config.E_min)/step_E)), idx] += trace.imag
            # Assemble spectrum
            for idx, n_c in enumerate(n_c_vector[:]):
                if config.selected_eigs[L] == None:
                    x_L.extend(np.repeat(log_n_c_tilde_vector[idx],N_block_list[0]-1))
                else:
                    x_L.extend(np.repeat(log_n_c_tilde_vector[idx],len(config.selected_eigs[L])-len(discard_eigval[idx][L])))
        
            # Saving
            np.savetxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), sp_func_L, fmt='%1.7e')
            np.savetxt('{}/spectrum_x_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), x_L, fmt='%1.7e')
            np.savetxt('{}/spectrum_y_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), y_L, fmt='%1.7e')
    
    print_status(start_time, "Finished")