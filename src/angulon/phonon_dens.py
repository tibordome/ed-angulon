#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:58:44 2021

@author: tibor
"""

import numpy as np
from class_ham import ham
from scipy import sparse
from print_msg import print_status
import config
from utilities import getH, corr_k_tile, corr_r_tile, dens_tile
from multiprocessing import Pool
from plotting_utilities import plotDensMultFigs
from functools import partial

def getPhononDensities(start_time):
    """
    Fetches the phonon densities in real space
    Parameters for n and L are set in the config.py file
    Note that we calculate the phonon density for the lowest-eigenvalue-state in the L-block.
    Note that we calculate the phonon density at phi-angle = 0.0 and vary r and theta (r, theta, phi), the latter btw. 0 and np.pi.
    Effect:
    --------------
    phonon_density.txt: Contains phonon density information"""
    print_status(start_time,'Starting getPhononDensities()')
    
    log_n_c_tilde_vector = np.linspace(config.n_min, config.n_max, (abs(config.n_max-config.n_min)+1)+(config.n_c_div-1)*abs(config.n_max-config.n_min))
    n_c_vector = np.exp(log_n_c_tilde_vector)
    for L in range(config.L_max*config.L_max_only, config.L_max+1):
        for n_c_idx, n_c in enumerate(n_c_vector):
            if n_c == n_c_vector.min():
                print_status(start_time,'Setting the stage')
                # Setting the stage
                ED = ham(config.u_0, config.u_1, config.r_0, config.r_1, config.m_b, n_c, config.k_max[L], config.k_min, config.delta_k, config.l_max, L, config.n_ph_max, config.B, config.a_bb, config.mod_what)
                ED.getStatesAndFunctions()
                 
                config.makeGlobalStates(ED.states_all)
                config.makeGlobalPos(ED.states_pos)
                config.makeGlobalSection(ED.states_section)
                config.makeGlobalCG(ED.CG)
                config.makeGlobalBosQIndices(ED.bos_qindices.base)
                config.makeGlobalU(ED.U)
                config.makeGlobalW1(ED.W_1)
                config.makeGlobalW2(ED.W_2)
                config.makeGlobalw(ED.w)
            else:
                # Update parameters
                ED.changeDensity(n_c) # To avoid recalculating w etc.  
                config.makeGlobalDensity(n_c)
                config.makeGlobalU(ED.U)
                config.makeGlobalW1(ED.W_1)
                config.makeGlobalW2(ED.W_2)
                config.makeGlobalw(ED.w)
        
            # Getting H
            data_tot, data_row_tot, data_col_tot, N_block = getH(L)
            H = sparse.csr_matrix((data_tot, (data_row_tot, data_col_tot)), shape=(N_block, N_block))
            eigenvalues, eigenvectors = sparse.linalg.eigsh(H, k = N_block-1, which = 'SA', tol=1E-5, maxiter=20000)
            print_status(start_time,'Finished calculating the matrix elements. Smallest eigenvalue for n = {:.2} and delta_k = {:.2} and N = {} in block L = {} is {:.2}'.format(np.log(n_c), config.delta_k, config.n_ph_max, L, eigenvalues[0]))
        
            # Getting correlation functions
            corr_k = corr_k_tile(np.float32(eigenvectors[:,0]), L)
            print_status(start_time,'Finished calculating corr_k array.')
            config.makeGlobalCorrK(corr_k)
            
            paramlist = [(i,) for i in np.arange(0, int(round((config.r_space_max-config.delta_r)/config.delta_r))+1)]
        
            corr_r = np.zeros((int(round((config.r_space_max-config.delta_r)/config.delta_r))+1, config.l_max+1, 2*config.l_max+1, config.l_max+1, 2*config.l_max+1), dtype=complex)
            with Pool() as pool:
                result = pool.starmap(partial(corr_r_tile, eigenvectors[:,0], L), paramlist) # started all the jobs
                for present in result:
                    corr_r_arr, r = tuple(present)
                    corr_r[r] = corr_r_arr
            print_status(start_time,'Finished calculating corr_r array.')
            config.makeGlobalCorrR(corr_r)
            
            # Calculating phonon density profile
            phon_dens = np.zeros((len(np.arange(0, int(round((config.r_space_max-config.delta_r)/config.delta_r))+1)), len(np.arange(0, int(round((config.theta_max)/config.delta_theta))+1))), dtype=complex)
            with Pool() as pool:
                result = pool.starmap(partial(dens_tile, eigenvectors[:,0], L), paramlist) # started all the jobs
                for present in result:
                    phon_dens_vec, r = tuple(present)
                    phon_dens[r] = phon_dens_vec
            print_status(start_time,'Finished calculating phonon densities, saving to phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.txt'.format(L, n_c_idx, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
            np.savetxt('{}/phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, n_c_idx, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), phon_dens.real, fmt='%1.7e')   # use exponential notation
    np.savetxt('{}/nc_values_N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), n_c_vector, fmt='%1.7e')   # use exponential notation
    
    # Plot density profiles in separate figures
    plotDensMultFigs(start_time)
    
    print_status(start_time,'Finished getPhononDensities()')