#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:14:05 2021

@author: tibor
"""

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', 'config'))
import config
config.initialize()

def test_config():
    
    assert (config.k_max[0]/config.delta_k).is_integer()
    assert config.m_b == 1 
    assert config.B == 1
    assert len(config.k_max) == config.L_max+1
