#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:17:24 2021

@author: tibor
"""

import os
import sys
import inspect
import subprocess
import time
start_time = time.time()
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['rm', 'out.txt'], cwd=os.path.join(currentdir, '..', '..', 'output', 'raw_data'))
subprocess.call(['python3', 'setup.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir))
subprocess.call(['chmod', '777', 'utilities.so'], cwd=os.path.join(currentdir))
subprocess.call(['chmod', '777', 'class_ham.so'], cwd=os.path.join(currentdir))
subprocess.call(['chmod', '777', 'cython_functions.so'], cwd=os.path.join(currentdir))
sys.path.append(os.path.join(currentdir, '..', '..', 'config'))
import config
config.initialize()
sys.path.append(os.path.join(currentdir, '..', 'src/angulon'))
from phonon_dens import getPhononDensities
from create_En_landscape import createEnLandscape

# Spectrum & Spectral Function
createEnLandscape(start_time)

# Phonon Densities
getPhononDensities(start_time)