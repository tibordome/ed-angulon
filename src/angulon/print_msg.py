#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:26:34 2021

@author: tibor
"""

import time
import config

# Routine to print script status to command line, with elapsed time
def print_status(start_time,message):
    elapsed_time = time.time() - start_time
    file_object = open('{}/out.txt'.format(config.raw_data_dest), 'a+')
    print('%d\ts: %s' % (elapsed_time,message))
    file_object.write('%d\ts: %s \n' % (elapsed_time,message))
    file_object.close()