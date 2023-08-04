#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:48:03 2019

@author: tibor
"""

import numpy as np
import inspect
import os

def getEigsList(purpose="all"):
    if purpose == "custom":
        return [list(range(1)), list(range(int(round((k_max[1]-k_min)/delta_k))-39)), list(range(int(round((k_max[2]-k_min)/delta_k))-20))]
    else:
        assert purpose == "all"
        return [None, None, None]
        
def initialize():   
    
    global u_0 # Molecule-boson interaction potential amplitude. The values here should be experimentally motivated
    global u_1 # Same as previous for l = 1.
    global r_0 # The real-space shape of the molecule-boson interaction potential is determined by the collection of parameters r_i and u_i
    global r_1 # Same as previous for l = 1.
    global m_b # The dispersion relation behavior depends both on btw. m_b and n_c, but for simplicity m_b is often set to 1
    global n_c # Momentary density in code (gets overwritten constantly)
    global k_max # The higher, the more precise.
    global l_max # Boson lambda quantum number cutoff; we often assume l_max is <= 1, since the mol_bos_potential is only implemented for these, but W_2 term in principle allows for l_max anything
    global L_max # Molecular angular momentum (which is same as total angular momentum in bath picture) quantum number cutoff
    global n_ph_max # My kernels can't really take n_ph_max = 4 due to memory restrictions
    global B # Magnetic field, conventionally set to 1.
    global a_bb # Boson-boson interaction strength.
    global states_pos # Will store state information
    global states_section # Will store state information
    global states_all # Will store state information
    global diff_all # Will store the occupation number differences between the various mol-bos states
    global bos_qindices # Will store state information
    global delta_k # Not everything is permitted for delta_k, since it must divide k_max with 0 rest
    global CG # Will store CG coefficients
    global U # Will store Fröhlich term coefficients
    global w # Will store dispersion relation info
    global W_1 # Will store b+b term coefficients
    global W_2 # Will store b+b+ term coefficients
    global eigvectors # Will store all eigenvectors
    global n_min # Log, n_max - n_min shall be even for viz purposes
    global n_max # Again, log
    global E_max # Max number on E-axis, E_max-E_min shall be even
    global E_min # Min number on E-axis
    global E_div # The inverse of this is the step size in the n-discretization
    global n_c_div # The inverse of this is the step size in the n-discretization
    global eps # The complex off-set in the green function calculations
    global k_min # Minimal k
    global with_W_1 # If True, W_1 is taken into account
    global with_W_2 # If True, W_2 is taken into account
    global with_mol
    global with_Froehlich
    global mod_what # The hashing-relying spectral function calculation needs number to take the modulo with
    global L_max_only # Whether all L-blocks up to config.L_max shall be included or just config.L_max
    global selected_eigs # Either getEigsList("all") or getEigsList("custom"). In the latter case, go to getEigsList and specify your custom eigenvalues that you need. In the former, you will get N_block - 1 eigenstates (-1 to run ARPACK only once)
    global corr_r # Will store the real-space correlation functions
    global corr_k # Will store the momentum-space correlation functions
    global raw_data_dest # Destination for raw output data, .txt files
    global viz_dest # Destination for plots
    global r_space_max # Maximum radius out to which we calculate phonon densities
    global delta_r # Radial discretization in phonon density calculations
    global theta_max # Maximum theta (r, phi, theta) in phonon density calculations
    global delta_theta # Angular (theta) discretization in phonon density calculations
    global phi_max # Maximum phi (r, phi, theta) in phonon density calculations
    global delta_phi # Angular (phi) discretization in phonon density calculations
    
    # Units and Interaction Strengths
    r_0 = 1.5 
    r_1 = 1.5 
    u_0 = 0
    u_1 = 50 # usually u_0/1.75
    m_b = 1 
    B = 1
    a_bb = 0.0 # 0.0 for non-interacting bosons, 3.3 otherwise
    
    # Numerical-physical parameters
    k_max = [4., 4., 4.]
    n_c = np.exp(0)
    delta_k = 0.05
    k_min = delta_k
    with_W_1 = True
    with_W_2 = False
    with_mol = True
    with_Froehlich = True
    l_max = 1
    L_max = 2
    L_max_only = False
    n_ph_max = 1
    eps = 0.05j  
    
    # Spectral function only specifications
    n_min = -8
    n_max = 4
    n_c_div = 32
    E_div = 32
    E_max = 7
    E_min = -4
    
    # Phonon Density specifications
    r_space_max = 50
    delta_r = 0.1
    theta_max = np.pi
    delta_theta = theta_max/40
    phi_max = 2*np.pi
    delta_phi = phi_max/100
    
    # Truly numerical parameters
    mod_what = 20   
    selected_eigs = getEigsList("all")  
    
    # Auxilliary
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    raw_data_dest = os.path.join(currentdir, '..', 'output', 'raw_data')
    viz_dest = os.path.join(currentdir, '..', 'output', 'viz')
    

def makeGlobalStates(states_all_): # the global variable has to have a different name than the argument passed to this function
    global states_all
    states_all = states_all_
    
def makeGlobalPos(states_pos_):
    global states_pos
    states_pos = states_pos_

def makeGlobalSection(states_section_):
    global states_section
    states_section = states_section_
    
def makeGlobalDiffAll(diff_all_):
    global diff_all
    diff_all = diff_all_
    
def makeGlobalBosQIndices(bos_qindices_):
    global bos_qindices
    bos_qindices = bos_qindices_

def makeGlobalDeltaK(delta_k_):
    global delta_k
    global k_min
    global k_max
    global selected_eigs
    delta_k = delta_k_
    k_min = delta_k_
    selected_eigs = getEigsList("all")
    
def makeGlobalU(U_):
    global U
    U = U_
    
def makeGlobalW1(W_1_):
    global W_1
    W_1 = W_1_
    
def makeGlobalW2(W_2_):
    global W_2
    W_2 = W_2_
    
def makeGlobalCG(CG_):
    global CG
    CG = CG_
    
def makeGlobalDensity(n_c_):
    global n_c
    n_c = n_c_
    
def makeGlobalKMax(k_max_):
    global k_max
    k_max = k_max_
    
def makeGlobalNphMax(n_ph_max_):
    global n_ph_max
    n_ph_max = n_ph_max_

def makeGlobaluVal(u_0_):
    global u_0
    global u_1
    u_0 = u_0_   
    u_1 = u_0_/1.75
    
def makeGlobalEigvectors(eigvectors_):
    global eigvectors
    eigvectors = eigvectors_
    
def makeGlobalCorrR(corr_r_):
    global corr_r
    corr_r = corr_r_
    
def makeGlobalCorrK(corr_k_):
    global corr_k
    corr_k = corr_k_
    
def makeGlobalw(w_):
    global w
    w = w_
    
def makeGlobalWithW1(with_W_1_):
    global with_W_1
    with_W_1 = with_W_1_

def makeGlobalWithW2(with_W_2_):
    global with_W_2
    with_W_2 = with_W_2_
