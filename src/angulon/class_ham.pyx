#distutils: language = c++
#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:37:12 2021

@author: tibor
"""

import numpy as np
cimport numpy as cnp
from numpy cimport PyArray_ZEROS
from cython_gsl cimport *
from scipy.special cimport cython_special
from functools import partial
from cython.parallel import prange
from sympy.physics.quantum.cg import CG
import scipy.integrate
from libc.math cimport round, sqrt, pi, exp
import scipy.special as sp
cimport openmp
import math # For factorial
import copy # For deepcopy (to not change mutable object like list, np.array, in local function)

cdef float V(float r, int lambd, float r_0, float r_1, float u_0, float u_1) nogil: # Molecule-boson interaction potential. Two versions exist. Original plus the one used to derive a_ib <-> u_0 mapping
    if lambd == 0:
        return -exp(-r**2/(r_0**2))*u_0/(2*r_0**2)
    elif lambd == 1:
        return -exp(-r**2/(r_1**2))*u_1/(2*r_1**2)
    else:
        return 0.0 # Theoretically, lambd can be larger than 1, but in our case, contribution is zero
    
cdef double f1(double r, void * params) nogil:
    cdef double lambd=(<double *> params)[0]
    cdef double k =   (<double *> params)[1]
    cdef double r_0 = (<double *> params)[2]
    cdef double r_1 = (<double *> params)[3]
    cdef double u_0 = (<double *> params)[4]
    cdef double u_1 = (<double *> params)[5]
    return r**2*V(<float>r, <int>lambd, <float>r_0, <float>r_1, <float>u_0, <float>u_1)*cython_special.spherical_jn(<int>lambd, <float>(k*r))
        
cdef double f2(double r, void * params) nogil:
    cdef double lambd = (<double *> params)[0]
    cdef double l=      (<double *> params)[1]
    cdef double lprime =(<double *> params)[2]
    cdef double k =     (<double *> params)[3]
    cdef double q =     (<double *> params)[4]
    cdef double r_0 =   (<double *> params)[5]
    cdef double r_1 =   (<double *> params)[6]
    cdef double u_0 =   (<double *> params)[7]
    cdef double u_1 =   (<double *> params)[8]
    return r**2*V(<float>r, <int>lambd, <float>r_0, <float>r_1, <float>u_0, <float>u_1)*cython_special.spherical_jn(<int>l,<float>(k*r))*cython_special.spherical_jn(<int>lprime,<float>(q*r))

cdef class ham:
    
    cdef float u_0
    cdef float u_1
    cdef float r_0
    cdef float r_1
    cdef float m_b
    cdef float n_c
    cdef float k_max
    cdef float k_min
    cdef float delta_k
    cdef int l_max
    cdef int L_max
    cdef int n_ph_max
    cdef float a_bb
    cdef int mod_what
    cdef float B
    cdef int N
    cdef public float[:,:] U
    cdef float[:,:] U_no_dens
    cdef float[:,:,:,:,:] W
    cdef public float[:,:,:,:,:] W_1
    cdef public float[:,:,:,:,:] W_2
    cdef public float[:,:,:,:,:,:] CG
    cdef public int[:,:] states_pos
    cdef public int[:,:] states_section
    cdef public float[:] w
    cdef public int[:,:] bos_qindices
    cdef bint bos_dilute
    cdef public list states_all
    cdef list bos_state_list
    cdef list hash_table_list
    
    def __cinit__(self, float u_0, float u_1, float r_0, float r_1, float m_b, float n_c, float k_max, float k_min, float delta_k, int l_max, int L_max, int n_ph_max, float B, float a_bb, int mod_what, bint bos_dilute = True):
        
        # The following are instance variables (which have different values for each object instance), not class variables, which would be defined above all methods
        self.u_0 = u_0
        self.u_1 = u_1
        self.r_0 = r_0
        self.r_1 = r_1
        self.m_b = m_b
        self.n_c = n_c
        self.k_max = k_max
        self.k_min = k_min
        self.delta_k = delta_k
        self.l_max = l_max
        self.L_max = L_max
        self.n_ph_max = n_ph_max
        self.a_bb = a_bb
        self.mod_what = mod_what
        self.B = B
        cdef int N_tmp = 0 # Number of phononic states (boxes that can be filled by phonons)
        cdef int l
        for l in range(0, l_max+1):
            N_tmp += <int>(round((k_max-k_min)/delta_k)+1)*(2*l+1) # Note that we exclude k = 0 hereby, since that is the analytical result
                                                          # To avoid ValueError: factorial() only accepts integral values, we use round (int() just truncates)
        self.N = N_tmp
        self.U = np.zeros((<int>(round((k_max-k_min)/delta_k))+1, self.l_max+1), dtype = np.float32)  # not always 2
        self.W_1 = np.zeros((<int>(round((k_max-k_min)/delta_k))+1, <int>(round((k_max-k_min)/delta_k))+1, l_max+1, l_max+1, l_max+1), dtype = np.float32)
        self.W_2 = np.zeros((<int>(round((k_max-k_min)/delta_k))+1, <int>(round((k_max-k_min)/delta_k))+1, l_max+1, l_max+1, l_max+1), dtype = np.float32)
        self.U_no_dens = np.zeros((<int>(round((k_max-k_min)/delta_k))+1, self.l_max+1), dtype = np.float32)  # not always 2
        self.W = np.zeros((<int>(round((k_max-k_min)/delta_k))+1, <int>(round((k_max-k_min)/delta_k))+1, l_max+1, l_max+1, l_max+1), dtype = np.float32)
        self.CG = np.zeros((l_max+1, 2*l_max+1, l_max+1, 2*l_max+1, l_max+1, 2*l_max+1), dtype = np.float32)
        self.states_pos = np.zeros((L_max+1, n_ph_max+1), dtype = np.int32)
        self.states_section = np.zeros((L_max+1, n_ph_max+1), dtype = np.int32)
        self.states_all = []
        self.w = np.zeros(<int>(round((k_max-k_min)/delta_k))+1, dtype = np.float32)
        self.bos_qindices = np.zeros((<int>(round(N_tmp)), 3), dtype = np.int32)
        self.bos_state_list = []
        
    cdef float V_bb(self): # Boson-boson interaction quantifier
        return 4*pi*self.a_bb/self.m_b

    cdef float e(self, float k): # Dispersion of free bosons
        return k**2/(2*self.m_b)

    cdef float dispersion(self, float k): # Dispersion of weakly interacting bosons
        return sqrt(self.e(k)*(self.e(k)+2*self.V_bb()*self.n_c))
    
    cdef float u(self, float k): # Boguliubov coefficient
        return sqrt((self.e(k)+self.V_bb()*self.n_c)/(2*self.dispersion(k))+1/2)

    cdef float v(self, float k): # Boguliubov coefficient
        return -sqrt((self.e(k)+self.V_bb()*self.n_c)/(2*self.dispersion(k))-1/2)
    
    cdef float U_no_dens_f(self, float k, int lambd): # Relevant function in angulon Hamiltonian, laboratory frame
        
        cdef gsl_integration_workspace *w
        w = gsl_integration_workspace_alloc (1000)
        cdef gsl_function F
        cdef double params[6]
        params[0] = <double>lambd
        params[1] = <double>k
        params[2] = <double>self.r_0
        params[3] = <double>self.r_1
        params[4] = <double>self.u_0
        params[5] = <double>self.u_1        
        F.function = &f1
        F.params = params
        
        cdef double result
        cdef double error
        gsl_integration_qags (&F, 0.0, 1000, 1e-6, 1e-6, 1000, w, &result, &error) # Do not use 1e+4 or 10**4 or similar notation for upper integration boundary.
        gsl_integration_workspace_free(w)
        return result
    
    cdef float W_f(self, float k, float q, int lprime, int lambd, int l) nogil: # Relevant function in angulon Hamiltonian, laboratory frame
        cdef float cg = self.CG[lprime, 0+self.l_max, lambd, 0+self.l_max, l, 0+self.l_max]
        cdef gsl_integration_workspace *w
        w = gsl_integration_workspace_alloc (1000)
        cdef gsl_function F
        cdef double params[9]
        params[0] = <double>lambd
        params[1] = <double>l
        params[2] = <double>lprime
        params[3] = <double>k
        params[4] = <double>q
        params[5] = <double>self.r_0
        params[6] = <double>self.r_1
        params[7] = <double>self.u_0
        params[8] = <double>self.u_1
        F.function = &f2
        F.params = params
        
        cdef double result
        cdef double error
        gsl_integration_qags (&F, 0.0, 1000, 1e-6, 1e-6, 1000, w, &result, &error) # Do not use 1e+4 or 10**4 or similar notation for upper integration boundary.
        gsl_integration_workspace_free(w)
        return 2/(pi)*k*q*result*cg*sqrt((2*lprime+1)/(2*l+1))
    
    cdef int getBosStat(self, int n_ph): # Counts the number of ways to arrange n_ph bosons in self.N boxes
        return math.factorial(self.N-1+n_ph)/(math.factorial(n_ph)*math.factorial(self.N-1))
    
    def getBosStates(self, n_ph):
        
        # Output: b_states, list of [n_ph, box #, number of bosons, box #, number of bosons], i.e. if n_ph == 0 just [0]
        
        def right_left(a): # Ethymology: in general, you pull rightmost !=0 phonon (except at index len(a)-1) from i to the right by one step to i+1; however, if there is an entry != 0 right to i+1, the position to where the one has just been pulled, at i+l,then you pull phonon at i+l all the way to i+1; writing a[:] or copy.deepcopy(a) here as an argument is illegal
    
            a = copy.deepcopy(a) # leave exterior np.array a unchanged
            if isinstance(a, np.ndarray) == False:
                raise ValueError('This is an illegal input')   
            for i in range(len(a[0])-1, -1, -1):
                if a[0][i] != 0:
                    if i == len(a[0])-1:  # if last one is != 0, do not move that one
                        pass 
                    else: 
                        # 1. drag 1 phonon to the right
                        a[0][i] -= 1
                        a[0][i+1] += 1
                        # 2. make sure that to the right of the one that has just been dragged there, just before, there is no != 0 entry: drag them to position i+1
                        if i+2 > len(a[0])-1:
                            break
                        for l in range(i+2, len(a[0]),+1):
                            if a[0][l] != 0:
                                a[0][i+1] += a[0][l]
                                a[0][l] = 0
                        break
            return a
    

        def transform(a):
    
            # In python, a dynamic array is an 'array' from the array module, but it is easier to work with dynamic lists and convert them to arrays in the end.
            lst = []
            size_of_x = 0
    
            # 1. add two list elements when you find a != 0 entry in a
            for i in range(len(a[0])):
                if a[0][i] != 0:
                    lst.append(i)
                    lst.append(a[0][i])
                    size_of_x += 2
            
            # 2. transform list into array
            vec = np.asarray(lst).reshape((size_of_x,))
            
            return vec
        
        # 1. create the "box-#-ascending-based" enumeration of all the possible combinations of distributing n_ph bosons among N boxes (problem: a lot of zero occupied boxes)
        x = np.zeros((<int>(round(self.getBosStat(n_ph))), 1, <int>(round(self.N))), dtype = np.int32) # hence each row out of the bose_stat(n_ph) many has shape (1, self.N), so far at least (before transform fct.). Also, we want integer values for easy access later, not float (default).
        b_states = []
        x[0][0][0] = n_ph # initializing first realization
        for l in range(1, <int>(round(self.getBosStat(n_ph)))):
            x[l] = right_left(x[l-1]) # the argument of right_left needs to be passed as a copy to avoid changes made to the data to which where x[l-1] is pointing, a priori, but actually, this does not work, since python works by call-by-object whereby the mutable object np.array is changed itself, hence the previous pointer/label/variable/name x[l-1] would, after leaving the function body, point to an object with different data; solution: deepcopy
        # 2. transform into more efficient representation with lists (array module is too constrictive)
        for l in range(0, <int>(round(self.getBosStat(n_ph)))):
            vec = transform(x[l])
            lst = vec.tolist()
            lst = [n_ph] + lst
            b_states.append(lst)
            
        return b_states
      
    
    def getStates(self, z_proj_constraint): 
        
        # We restrict to states with a valid bosonic sum of projections onto z'axis if z_proj_constraint = True. Otherwise, we will have more states, including those that violate the z-projection constraint.
        # Output: 'states' list will be a list (np.array is not even possible due to different shapes of the rows) of all the possible states, entries having the form [L, n, # bosons, # bosonic state S_i, # bosons in S_i, ...]
        
        def isStateValid(row): # Return whether projection condition is satisfied or not
            state_valid = False
            boson_projs = 0
            for run in range(0, <int>(round((len(row)-3)/2))):
                boson_projs += row[2*run+4]*self.bos_qindices[row[2*run+3]][2]
            if boson_projs == row[1]:
                state_valid = True
            return state_valid
        
        def getMolQNumbers(L): # This will yield an ordered list of all possible combinations of L, M and n
            x = []
            for n in range(-L, L+1):
                x.append([L, n])
            return x
    
        states = []
        
        if z_proj_constraint == True:
            for L in range(0, self.L_max+1):
                mol_qnumbers = getMolQNumbers(L)
                helper = []
                for n_ph in range(0, self.n_ph_max+1):
                    b_states = self.getBosStates(n_ph)
                    for I in range(0, len(mol_qnumbers)): 
                        for b_index in range(0, len(b_states)):
                            add = mol_qnumbers[I]+b_states[b_index]
                            if isStateValid(add) == True: # The state shall only be appended if it is permissible
                                helper.append(add)
                if len(helper) != 0:
                    states.append(helper)
            return states
        
        else:
            for L in range(0, self.L_max+1):
                mol_qnumbers = getMolQNumbers(L)
                helper = []
                for n_ph in range(0, self.n_ph_max+1):
                    b_states = self.getBosStates(n_ph)
                    for I in range(0, len(mol_qnumbers)): 
                        for b_index in range(0, len(b_states)):
                            add = mol_qnumbers[I]+b_states[b_index]
                            helper.append(add)
                if len(helper) != 0:
                    states.append(helper)
            return states
        
    def createBosStateList(self):
        
        bos_state_list = []
        for n_ph in range(0, self.n_ph_max+1):
            b_states = self.getBosStates(n_ph)  # A list of lists
            for b_index in range(0, len(b_states)):
                bos_state_list.append(b_states[b_index])
        self.bos_state_list = bos_state_list
        
    def getBosStatesQIndices(self): 
        
        # Output: returns an ordered numpy array with all possible combinations of k, \lambda and \mu
        
        x = np.zeros((<int>(round(self.N)), 3), dtype = np.int32)
        pos = 0
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1): # note that using a k-index instead of k is more efficient, since box remains integer valued (valid for all quantum numbers)
            for lambd in range(0, self.l_max+1):
                for mu in range(-lambd, lambd+1):
                    x[pos][0] = k
                    x[pos][1] = lambd
                    x[pos][2] = mu
                    pos += 1              # This is the index position
                                          # In general, floating point numbers cannot be represented exactly. One should therefore be careful of round-off errors --> use round()  
        return x
    
    def calculateUNoDens(self):
        
        # Output: self-explanatory
        
        cdef Py_ssize_t k
        cdef Py_ssize_t lambd
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            for lambd in range(0, self.l_max+1):
               self.U_no_dens[k, lambd] = self.U_no_dens_f(k*self.delta_k+self.k_min, lambd)
    
    cdef float[:,:,:,:] calculateWFixK(self, int k, float[:,:,:,:] arr) nogil:
            cdef Py_ssize_t q
            cdef Py_ssize_t lprime
            cdef Py_ssize_t lambd
            cdef Py_ssize_t l
            for q in range(<int>(round((self.k_max-self.k_min)/self.delta_k))+1):
                for lprime in range(self.l_max+1):
                    for lambd in range(self.l_max+1):
                        for l in range(self.l_max+1):
                            arr[q, lprime, lambd, l] = self.W_f(k*self.delta_k+self.k_min, q*self.delta_k+self.k_min, lprime, lambd, l)
            return arr
    
    def calculateW(self):
        cdef float[:,:,:,:,:] arr = np.zeros((<int>(round((self.k_max-self.k_min)/self.delta_k))+1, <int>(round((self.k_max-self.k_min)/self.delta_k))+1, self.l_max+1, self.l_max+1, self.l_max+1), dtype = np.float32)
        cdef Py_ssize_t k
        for k in prange(<int>(round((self.k_max-self.k_min)/self.delta_k))+1, schedule = "dynamic", nogil=True):
            self.W[k] = self.calculateWFixK(k, arr[k]) # started all the jobs
                
    def calculateDispersion(self):
         
        # Output: self-explanatory
        
        cdef Py_ssize_t k_index
        for k_index in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            self.w[k_index] = self.dispersion(k_index*self.delta_k+self.k_min)
        
    def getStatesAndRelated(self):
        
        # Output: self-explanatory
        
        self.states_all = self.getStates(True)
        self.states_pos = np.zeros((self.L_max+1, self.n_ph_max+1), dtype = np.int32)
        self.states_section = np.zeros((self.L_max+1, self.n_ph_max+1), dtype = np.int32)
        cdef Py_ssize_t L
        cdef Py_ssize_t n_ph
        for L in range(0, self.L_max+1):
            for n_ph in range(0, self.n_ph_max+1):
                self.states_pos[L, n_ph] = self.getStatesIndex(L, n_ph)
                self.states_section[L, n_ph] = self.getStatesSectionSize(L, n_ph)
        
    def getStatesAndFunctions(self):  
        
        # Output: sets class variables U, w and CG as np.arrays; further, sets class variable states_all as a list and class variables bos_qindices, states_pos and states_pos_section as a np.arrays and sets W_1 and W_2
        
        self.calculateDispersion()
        self.calculateUNoDens()
        cdef Py_ssize_t k
        cdef Py_ssize_t q
        cdef Py_ssize_t lambd
        cdef Py_ssize_t lprime
        cdef Py_ssize_t l
        cdef Py_ssize_t l_1
        cdef Py_ssize_t l_2
        cdef Py_ssize_t m_1
        cdef Py_ssize_t m_2
        cdef Py_ssize_t m
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            for lambd in range(0, self.l_max+1):  # note that self.l_max, which I had here previously, is utterly wrong!
                self.U[k, lambd] = sqrt(8*(k*self.delta_k+self.k_min)**2*self.n_c*self.e(k*self.delta_k+self.k_min)/(self.dispersion(k*self.delta_k+self.k_min)*(2*lambd+1)))*self.U_no_dens[k, lambd] 
        for l_1 in range(0, self.l_max+1):
            for m_1 in range(0, 2*self.l_max+1):
                for l_2 in range(0, self.l_max+1):
                    for m_2 in range(0, 2*self.l_max+1):
                        for l in range(0, self.l_max+1):
                            for m in range(0, 2*self.l_max+1):
                                self.CG[l_1, m_1, l_2, m_2, l, m] = float(CG(l_1, m_1 - self.l_max, l_2, m_2 - self.l_max, l, m - self.l_max).doit().evalf())
        self.calculateW()
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            for q in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
                for lprime in range(0, self.l_max+1):
                    for lambd in range(0, self.l_max+1):
                        for l in range(0, self.l_max+1):
                            self.W_1[k, q, lprime, lambd, l] = (self.u(q*self.delta_k+self.k_min)*self.u(k*self.delta_k+self.k_min)+self.v(q*self.delta_k+self.k_min)*self.v(k*self.delta_k+self.k_min))*sqrt((2*lambd+1)/(4*pi))*self.W[k, q, lprime, lambd, l]
                            self.W_2[k, q, lprime, lambd, l] = self.u(k*self.delta_k+self.k_min)*self.v(q*self.delta_k+self.k_min)*sqrt((2*lambd+1)/(4*pi))*self.W[k, q, lprime, lambd, l]
        
        self.bos_qindices = self.getBosStatesQIndices()
        self.getStatesAndRelated()
    
    def changeDensity(self, n_c):
        
        # Output: new n_c affects U, w, W_1, W_2, but not W
        
        self.n_c = n_c
        cdef Py_ssize_t k
        cdef Py_ssize_t q
        cdef Py_ssize_t lambd
        cdef Py_ssize_t lprime
        cdef Py_ssize_t l
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            for lambd in range(0, self.l_max+1):  # note that self.l_max, which I had here previously, is utterly wrong!
                self.U[k, lambd] = sqrt(8*(k*self.delta_k+self.k_min)**2*self.n_c*self.e(k*self.delta_k+self.k_min)/(self.dispersion(k*self.delta_k+self.k_min)*(2*lambd+1)))*self.U_no_dens[k, lambd] 
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            for q in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
                for lprime in range(0, self.l_max+1):
                    for lambd in range(0, self.l_max+1):
                        for l in range(0, self.l_max+1):
                            self.W_1[k, q, lprime, lambd, l] = (self.u(q*self.delta_k+self.k_min)*self.u(k*self.delta_k+self.k_min)+self.v(q*self.delta_k+self.k_min)*self.v(k*self.delta_k+self.k_min))*sqrt((2*lambd+1)/(4*pi))*self.W[k, q, lprime, lambd, l]
                            self.W_2[k, q, lprime, lambd, l] = self.u(k*self.delta_k+self.k_min)*self.v(q*self.delta_k+self.k_min)*sqrt((2*lambd+1)/(4*pi))*self.W[k, q, lprime, lambd, l]
        self.calculateDispersion()
    
    def changeN(self, N_new):
        
        # Output: affects states_all and related only
        
        self.n_ph_max = N_new
        self.getStatesAndRelated()
        
    def changeMomentumDiscretization(self, delta_k_new):
        
        # Output: affects everything essentially
        
        self.__init__(self.u_0, self.u_1, self.r_0, self.r_1, self.m_b, self.n_c, self.k_max, self.k_min, delta_k_new, self.l_max, self.L_max, self.n_ph_max, self.B, self.a_bb, self.mod_what)
        self.getStatesAndFunctions()
        
    def changeMaxMomentum(self, k_max_new):
         
        # Output: affects everything essentially
       
        self.__init__(self.u_0, self.u_1, self.r_0, self.r_1, self.m_b, self.n_c, k_max_new, self.k_min, self.delta_k, self.l_max, self.L_max, self.n_ph_max, self.B, self.a_bb, self.mod_what)
        self.getStatesAndFunctions()
        
        
    def changeU(self, u_0):
        
        # Output: new u_0 only affects U; not W (calculated on spot)
        
        self.u_0 = u_0
        self.u_1 = u_0/1.75
        self.calculateUNoDens()
        cdef Py_ssize_t k
        cdef Py_ssize_t lambd
        for k in range(0, <int>(round((self.k_max-self.k_min)/self.delta_k))+1):
            for lambd in range(0, self.l_max+1):  # note that self.l_max, which I had here previously, is utterly wrong!
                self.U[k, lambd] = sqrt(8*(k*self.delta_k+self.k_min)**2*self.n_c*self.e(k*self.delta_k+self.k_min)/(self.dispersion(k*self.delta_k+self.k_min)*(2*lambd+1)))*self.U_no_dens[k, lambd] 
        
    def getStatesIndex(self, L, n_ph): 
        
        # Output: returns the index position within states_all
        
        for i in range(0, len(self.states_all[L])):
            if self.states_all[L][i][2] == n_ph:
                return i
            
    def getStatesSectionSize(self, L, n_ph): 
        
        # Output: returns the n_ph-section size within states_all
        count = 0
        for i in range(0, len(self.states_all[L])):
            if self.states_all[L][i][2] == n_ph:
                count += 1
            if self.states_all[L][i][2] > n_ph: # To make it faster (check with timeit...)
                break
        return count
        
    def hashFunction(self, bos_state):
        
        # Input: list, which describes bosonic state; zeroth index: number of bosons, first index: bos-box number b_1, second index: occ. in b_1, etc.
        # Output: hash value of bos_state
        
        hash_val = 0
        bos_numb = 0
        for run in range(0, <int>(round((len(bos_state)-1)/2))):
            for run_box in range(0, bos_state[2*run+2]): # Each phonon in the phonon box
                hash_val += (bos_state[2*run+1]+1)*(len(self.bos_qindices)**bos_numb)
                bos_numb += 1
        return hash_val
    
    def createHashTable(self):
        
        # Output: a hash_table, i.e. a list, with blocks (fixed mod value, i.e. collisions) consisting of tuples (S_i (bosonic state), f(S_i))
        
        hash_table = [0]*self.mod_what
        hash_table[0] = [(0, 0)] # For state [L,0,0]
        for state_run in range(len(self.bos_state_list)):
            hash_val = self.hashFunction(self.bos_state_list[state_run]) 
            mod_val = hash_val % self.mod_what
            if hash_table[mod_val] == 0:
                hash_table[mod_val] = [(state_run+1, hash_val)]
            else:   # It is already occupied with something
                hash_table[mod_val].append((state_run+1, hash_val))
        
        self.hash_table_list = hash_table
            

    def getC(self, state, L):
        
        # Input: state, an array with different contributions from each states_all[L] component, and L
        # Output: C matrix of state, in which the column index runs over each possible self.bos_state_list entry, and in which row index gives the impurity state (only those that are allowed in self.states_all[L], i.e. n index from -L to L)
        
        C_matrix = np.zeros((2*L+1, len(self.bos_state_list)+1)) # Why +1?: No bosons is also valid!
        for state_run in range(len(state)):
            hash_val = self.hashFunction(self.states_all[L][state_run][2:])
            mod_val = hash_val % self.mod_what   
            for block_run in range(len(self.hash_table_list[mod_val])):
                if hash_val == self.hash_table_list[mod_val][block_run][1]: # Makes sense since hash function is injective!
                    row_index = self.states_all[L][state_run][1]+L # Projection along L vector could in theory take on 3 values for L=1 (for instance)
                    C_matrix[row_index, self.hash_table_list[mod_val][block_run][0]] = state[state_run]
        return C_matrix