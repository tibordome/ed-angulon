#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:14:05 2021

@author: tibor
"""

import numpy as np
import subprocess
import sys
import os
import inspect
from multiprocessing import Pool
from functools import partial
import time
start_time = time.time()
import config
config.initialize()
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['bash', 'create_dir.bash'], cwd=currentdir)
subprocess.call(['rm', 'utilities.so'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
subprocess.call(['rm', 'utilities.cpp'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
subprocess.call(['rm', 'class_ham.so'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
subprocess.call(['rm', 'class_ham.cpp'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
subprocess.call(['python3', 'setup.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
subprocess.call(['chmod', '777', 'utilities.so'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
subprocess.call(['chmod', '777', 'class_ham.so'], cwd=os.path.join(currentdir, '..', 'src/angulon'))
sys.path.append(os.path.join(currentdir, '..', 'src/angulon'))
from class_ham import ham
from scipy import sparse
from print_msg import print_status
from utilities import getH, corr_k_tile, corr_r_tile, dens_tile

def test_density_realness():
    
    print_status(start_time,'Starting test_density_realness()')
    
    # Setting the stage
    ED = ham(config.u_0, config.u_1, config.r_0, config.r_1, config.m_b, config.n_c, config.k_max[config.L_max], config.k_min, config.delta_k, config.l_max, config.L_max, config.n_ph_max, config.B, config.a_bb, config.mod_what)
    ED.getStatesAndFunctions()
     
    config.makeGlobalStates(ED.states_all)
    config.makeGlobalPos(ED.states_pos)
    config.makeGlobalSection(ED.states_section)
    config.makeGlobalCG(ED.CG)
    config.makeGlobalBosQIndices(ED.bos_qindices)
    config.makeGlobalU(ED.U)
    config.makeGlobalW1(ED.W_1)
    config.makeGlobalW2(ED.W_2)
    config.makeGlobalw(ED.w)

    # Getting H
    data_tot, data_row_tot, data_col_tot, N_block = getH(config.L_max)
    H = sparse.csr_matrix((data_tot, (data_row_tot, data_col_tot)), shape=(N_block, N_block))
    eigenvalues, eigenvectors = sparse.linalg.eigsh(H, k = N_block-1, which = 'SA', tol=1E-5, maxiter=20000)
    print_status(start_time,'Finished calculating the matrix elements. Eigenvalue for n = {:.2} and delta_k = {:.2} and N = {} in block L = {} is'.format(config.n_c, config.delta_k, config.n_ph_max, config.L_max))

    # Getting correlation functions
    corr_k = corr_k_tile(np.float32(eigenvectors[:,0]), config.L_max)
    print_status(start_time,'Finished calculating corr_k array.')
    config.makeGlobalCorrK(corr_k)
    
    paramlist = [(i,) for i in np.arange(0, int(round((config.r_space_max-config.delta_r)/config.delta_r))+1)]

    corr_r = np.zeros((int(round((config.r_space_max-config.delta_r)/config.delta_r))+1, config.l_max+1, 2*config.l_max+1, config.l_max+1, 2*config.l_max+1), dtype=complex)
    with Pool() as pool:
        result = pool.starmap(partial(corr_r_tile, eigenvectors[:,0], config.L_max), paramlist) # started all the jobs
        for present in result:
            corr_r_arr, r = tuple(present)
            corr_r[r] = corr_r_arr
    print_status(start_time,'Finished calculating corr_r array.')
    config.makeGlobalCorrR(corr_r)
    
    # Calculating phonon density profile
    phon_dens_temp = np.zeros((len(np.arange(0, int(round((config.r_space_max-config.delta_r)/config.delta_r))+1)), len(np.arange(0, int(round((config.theta_max)/config.delta_theta))+1))), dtype=complex)
    with Pool() as pool:
        result = pool.starmap(partial(dens_tile, eigenvectors[:,0], config.L_max), paramlist) # started all the jobs
        for present in result:
            phon_dens_vec, r = tuple(present)
            phon_dens_temp[r] = phon_dens_vec
    
    eps = 10**(-6) # Tolerance
    assert np.any(np.absolute(phon_dens_temp.imag) > eps) == False, "The phonon densities have a non-negligible imaginary contribution"