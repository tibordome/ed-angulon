#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:08:36 2021

@author: tibor
"""

import config

def acceptEig(eig, eig_run, L, discard_eigval, n_c, idx):
    if L == 1:
        if eig >= 2 + config.w_val[0] - 0.05:
            discard_eigval[idx][L].append(eig_run) # In place
            return False
    return True
