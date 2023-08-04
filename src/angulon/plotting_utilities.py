#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:50:42 2021

@author: tibor
"""

import matplotlib.pyplot as plt
colormap = plt.cm.gist_ncar
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib import colors
from pylab import text
import config
from print_msg import print_status
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

def isleft(row_idx, col_idx, rows, cols):
    if col_idx == 0:
        return True
    else:
        return False
    
def isbottom(row_idx, col_idx, rows, cols):
    if row_idx == len(rows)-1:
        return True
    else:
        return False

def plotSlctedDens(n_c_idx_l, text_label_l, L, desired_r_max, start_time):
    """
    Plot phonon density profile at various bath densities and L quantum numbers in one figure, as specified in config.py
    
    E.g. 
    # What to plot
    n_c_idx_l = [0, 10, 20]
    text_label_l = ['(a)', '(b)', '(c)']
    L = 0
    desired_r_max = 7
    """
    print_status(start_time,'Starting plotSlctedDens()')
        
    assert desired_r_max <= config.r_space_max, "Note that desired_r_max cannot be larger than config.r_space_max"
    log_n_c_tilde_vector = np.linspace(config.n_min, config.n_max, (abs(config.n_max-config.n_min)+1)+(config.n_c_div-1)*abs(config.n_max-config.n_min))
    n_c_vector = np.exp(log_n_c_tilde_vector)
    
    fig, ax = plt.subplots(nrows=len(n_c_idx_l), ncols=2, figsize=(18, 18))
    n_c_vector = [a for idx, a in enumerate(n_c_vector) if idx in n_c_idx_l]
    columns = [False, True]
    
    for n_c_idx, n_c in enumerate(n_c_vector): # rows
        for with_W1_idx, with_W1 in enumerate(columns): # columns
                
            # Load density info
            print_status(start_time,"Loading '{}/phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.txt".format(config.raw_data_dest, L, n_c_idx_l[n_c_idx], config.n_ph_max, withW1="True" if with_W1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
            phon_dens = np.loadtxt('{}/phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, n_c_idx_l[n_c_idx], config.n_ph_max, withW1="True" if with_W1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), dtype=float)
            
            # Plot density graph
            x = np.zeros(((int(round((desired_r_max-config.delta_r)/config.delta_r))+1)*phon_dens.shape[1]*2))
            z = np.zeros(((int(round((desired_r_max-config.delta_r)/config.delta_r))+1)*phon_dens.shape[1]*2))
            run = 0
            for r in np.arange(0, int(round((desired_r_max-config.delta_r)/config.delta_r))+1):
                for theta in np.arange(0, int(round(config.theta_max/config.delta_theta))+1):
                    for phi in [0, np.pi]:
                        x[run] = (r*config.delta_r+config.delta_r)*np.sin(theta*config.delta_theta)*np.cos(phi) # you can take any value for phi
                        z[run] = (r*config.delta_r+config.delta_r)*np.cos(theta*config.delta_theta)
                        run += 1
            c = np.repeat(phon_dens[:(int(round((desired_r_max-config.delta_r)/config.delta_r))+1)], repeats = 2, axis = 1).flatten()
            ax[n_c_idx,with_W1_idx].scatter(x, z, c=c, cmap='inferno_r')
            
            text_label = text_label_l[n_c_idx]
            text(0.1, 0.9,'{0}'.format(text_label), ha='center', va='center', transform=ax[n_c_idx,with_W1_idx].transAxes)
            text(0.9, 0.9,'$\ln (\~{{n}})$: {:.2f}'.format(np.log(n_c)), ha='center', va='center', transform=ax[n_c_idx,with_W1_idx].transAxes)
            
            # Tick formatting
            if isleft(n_c_idx, with_W1_idx, n_c_idx_l, columns):
                if isbottom(n_c_idx, with_W1_idx, n_c_idx_l, columns):
                    ax[n_c_idx,with_W1_idx].set_xlabel(r'$x$')
                    ax[n_c_idx,with_W1_idx].set_ylabel(r'$z$')
                    ax[n_c_idx,with_W1_idx].tick_params(bottom=True, right=True, top=True, left = True, labelbottom=True, labelright = False, labeltop = False, labelleft=True, direction='in')
                else:
                    ax[n_c_idx,with_W1_idx].set_ylabel(r'$z$')
                    ax[n_c_idx,with_W1_idx].tick_params(bottom=True, right=True, top=True, left = True, labelbottom=False, labelright = False, labeltop = False, labelleft=True, direction='in')
            else:
                if isbottom(n_c_idx, with_W1_idx, n_c_idx_l, columns):
                    ax[n_c_idx,with_W1_idx].set_xlabel(r'$x$')
                    ax[n_c_idx,with_W1_idx].set_ylabel(r'$z$')
                    ax[n_c_idx,with_W1_idx].yaxis.set_label_position("right")
                    ax[n_c_idx,with_W1_idx].tick_params(bottom=True, right=True, top=True, left = True, labelbottom=True, labelright = True, labeltop = False, labelleft=False, direction='in')
                else:
                    ax[n_c_idx,with_W1_idx].set_ylabel(r'$z$')
                    ax[n_c_idx,with_W1_idx].yaxis.set_label_position("right")
                    ax[n_c_idx,with_W1_idx].tick_params(bottom=True, right=True, top=True, left = True, labelbottom=False, labelright = True, labeltop = False, labelleft=False, direction='in')                      
    #fig.gca().set_aspect('equal', adjustable='box')
    fig.savefig('phon_dens_L{0}N{1}.pdf'.format(L, config.n_ph_max), bbox_inches="tight")
    print_status(start_time,'Plot phonon density profile phon_dens_L{0}N{1}.pdf'.format(L, config.n_ph_max))
    print_status(start_time,'Finished plotDensities()')
    
def plotSpFunc(start_time):
    """
    Plot spectral function"""
    print_status(start_time,'Starting plotSpFunc()')
    
    # Define figure
    fig, axes = plt.subplots(nrows=config.L_max+1-config.L_max*config.L_max_only, figsize=(10, 10))
    cmaps = ['Greens', 'Reds', 'Blues'][:config.L_max+1-config.L_max*config.L_max_only]
    for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
        sp_func = np.loadtxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L_ind, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
        max_val = np.max(sp_func)
        sp_func = sp_func/max_val
        second_smallest = np.unique(sp_func)[0]
        sp_func[sp_func < second_smallest] = second_smallest
        if config.L_max+1-config.L_max*config.L_max_only > 1:
            pa = axes[L_ind].imshow(sp_func, interpolation='None', origin='upper', extent=[config.n_min,config.n_max,config.E_min,config.E_max], cmap = cmaps[L_ind])
            cba = fig.colorbar(pa,ax=axes[L_ind],fraction=0.036)
            cba.ax.set_ylabel('$A_{}(\~E)$'.format(L), labelpad = 1.0, rotation=90)
            axes[L_ind].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[L_ind].yaxis.set_major_locator(MaxNLocator(integer=True))
            if L == config.L_max:
                axes[L_ind].set_xlabel(r'$\ln(\tilde{n})$')
            axes[L_ind].set_ylabel(r'$\tilde{E}$')
        else:
            pa = axes.imshow(sp_func, interpolation='None', origin='upper', extent=[config.n_min,config.n_max,config.E_min,config.E_max], cmap = "inferno_r")
            cba = fig.colorbar(pa,ax=axes,fraction=0.036)
            cba.ax.set_ylabel('$A_{}(\~E)$'.format(L), labelpad = 1.0, rotation=90)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.yaxis.set_major_locator(MaxNLocator(integer=True))
            if L == config.L_max:
                axes.set_xlabel(r'$\ln(\tilde{n})$')
            axes.set_ylabel(r'$\tilde{E}$')
            
    fig.tight_layout()
    plt.savefig('{}/sp_func_L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
    
    fig, axes = plt.subplots(nrows=config.L_max+1-config.L_max*config.L_max_only, figsize=(10, 10))
    for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
        sp_func = np.loadtxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L_ind, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
        max_val = np.max(sp_func)
        sp_func = sp_func/max_val
        second_smallest = np.unique(sp_func)[0]
        sp_func[sp_func < second_smallest] = second_smallest
        if config.L_max+1-config.L_max*config.L_max_only > 1:
            pa = axes[L_ind].imshow(sp_func, interpolation='None', origin='upper', extent=[config.n_min,config.n_max,config.E_min,config.E_max], cmap = cmaps[L_ind], norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(sp_func)))
            cba = fig.colorbar(pa,ax=axes[L_ind],fraction=0.036)
            cba.ax.set_ylabel('$A_{}(\~E)$'.format(L), labelpad = 1.0, rotation=90)
            axes[L_ind].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[L_ind].yaxis.set_major_locator(MaxNLocator(integer=True))
            if L == config.L_max:
                axes[L_ind].set_xlabel(r'$\ln(\tilde{n})$')
            axes[L_ind].set_ylabel(r'$\tilde{E}$')
        else:
            pa = axes.imshow(sp_func, interpolation='None', origin='upper', extent=[config.n_min,config.n_max,config.E_min,config.E_max], cmap = "inferno_r", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(sp_func)))
            cba = fig.colorbar(pa,ax=axes,fraction=0.036)
            cba.ax.set_ylabel('$A_{}(\~E)$'.format(L), labelpad = 1.0, rotation=90)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.yaxis.set_major_locator(MaxNLocator(integer=True))
            if L == config.L_max:
                axes.set_xlabel(r'$\ln(\tilde{n})$')
            axes.set_ylabel(r'$\tilde{E}$')
        
    fig.tight_layout()
    plt.savefig('{}/sp_func_log_L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
    
    print_status(start_time,'Starting plotSpFunc(), overlap L-blocks')
    
    # Define figure
    plt.figure()
    sp_func = np.zeros_like(np.loadtxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, 0, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False")))
    for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
        sp_func += np.loadtxt('{}/sp_func_L{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L_ind, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
    max_val = np.max(sp_func)
    sp_func = sp_func/max_val
    second_smallest = np.unique(sp_func)[0]
    sp_func[sp_func < second_smallest] = second_smallest
    
    plt.imshow(sp_func, interpolation='None', origin='upper', extent=[config.n_min,config.n_max,config.E_min,config.E_max], cmap = "inferno_r")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'$\ln(\tilde{n})$')
    plt.ylabel(r'$\tilde{E}$')
    cbar = plt.colorbar(fraction=0.036)
    cbar.ax.set_ylabel('$A_L(\~E)$', rotation=90)
    plt.savefig('{}/sp_func_Ls_L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
        
    plt.figure()
    plt.imshow(sp_func, interpolation='None', origin='upper', extent=[config.n_min,config.n_max,config.E_min,config.E_max], cmap = "inferno_r", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(sp_func)))
        
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'$\ln(\tilde{n})$')
    plt.ylabel(r'$\tilde{E}$')
    cbar = plt.colorbar(fraction=0.036)
    cbar.ax.set_ylabel('$A_L(\~E)$', rotation=90)
    plt.savefig('{}/sp_func_log_Ls_L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
        
def plotDensMultFigs(start_time):
    """
    Plot phonon density profile at various bath densities and L quantum numbers in separate figures, as specified in config.py"""
    print_status(start_time,'Starting plotDensMultFigs()')
    
    log_n_c_tilde_vector = np.linspace(config.n_min, config.n_max, (abs(config.n_max-config.n_min)+1)+(config.n_c_div-1)*abs(config.n_max-config.n_min))
    n_c_vector = np.exp(log_n_c_tilde_vector)
    for L in range(config.L_max*config.L_max_only, config.L_max+1):
        for n_c_idx, n_c in enumerate(n_c_vector):
            # Parameters
            if L == 0:
                desired_r_max = 7
            else:
                desired_r_max = 45
            assert desired_r_max <= config.r_space_max, "Note that desired_r_max cannot be larger than config.r_space_max"
            text_label = '(a)'
            
            # Load density info
            phon_dens = np.loadtxt('{}/phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.txt'.format(config.raw_data_dest, L, n_c_idx, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), dtype=float)
            
            # Plot density graph
            x = np.zeros(((int(round((desired_r_max-config.delta_r)/config.delta_r))+1)*phon_dens.shape[1]*2))
            z = np.zeros(((int(round((desired_r_max-config.delta_r)/config.delta_r))+1)*phon_dens.shape[1]*2))
            run = 0
            for r in np.arange(0, int(round((desired_r_max-config.delta_r)/config.delta_r))+1):
                for theta in np.arange(0, int(round(config.theta_max/config.delta_theta))+1):
                    for phi in [0, np.pi]:
                        x[run] = (r*config.delta_r+config.delta_r)*np.sin(theta*config.delta_theta)*np.cos(phi) # you can take any value for phi
                        z[run] = (r*config.delta_r+config.delta_r)*np.cos(theta*config.delta_theta)
                        run += 1
                        
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            c = np.repeat(phon_dens[:(int(round((desired_r_max-config.delta_r)/config.delta_r))+1)], repeats = 2, axis = 1).flatten()
            
            img = ax.scatter(x, z, c=c, cmap='inferno_r')
            #fig.colorbar(img)
            #ax.set_xlabel(r'$x$')
            #ax.set_ylabel(r'$z$')
            ax.yaxis.set_label_position("right")
            #matplotlib.colorbar.Colorbar.remove(img)
            ax.tick_params(labelbottom='off', labelright='off', labelleft='off', bottom='on', top='on', right='on', direction='in')
            #text(0.1, 0.9,'{0}'.format(text_label), ha='center', va='center', transform=ax.transAxes)
            text(0.9, 0.9,'$\ln (\~{{n}})$: {:.2f}'.format(np.log(n_c)), ha='center', va='center', transform=ax.transAxes)
            fig.gca().set_aspect('equal', adjustable='box')
            fig.savefig('{}/phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, L, n_c_idx, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
            print_status(start_time,'Plot phonon density profile phon_dens_L{}n{}N{}wW1{withW1}wW2{withW2}.pdf'.format(L, n_c_idx, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"))
    print_status(start_time,'Finished plotDensities()')


def plotSpectrum(x, y, start_time):
    """
    Plot spectrum"""
    print_status(start_time,'Starting plotSpectrum()')
    plt.figure()
    area = np.pi*0.7
    plt.scatter(x, y, s=area, alpha=0.5) # There is: alpha = alpha blending value, between 0 (transparent) and 1 (opaque). There is: linewidths = linewidth of the marker edges. There is: edgecolors = the edge color of the marker. 
    plt.xlabel(r'$\ln(\tilde{n})$')
    plt.ylabel(r'$\tilde{E}$')
    plt.ylim((config.E_min, config.E_max))
    plt.savefig('{}/spectrum_L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
    
def plotPolDistance(x, y, start_time):
    """
    Plot distance between attractive polarons"""
    print_status(start_time,'Starting plotPolDistance()')
    if len(x) > 1 and len(y) > 1:
        plt.figure()
        x_interest = [[] for _ in range(config.L_max+1-config.L_max*config.L_max_only)]
        y_interest = [[] for _ in range(config.L_max+1-config.L_max*config.L_max_only)]
        for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
            # Find indices of non-unique entries in x
            unique_values = np.unique(x[L_ind])
            x_idxs = np.empty((0,), dtype = np.float32)
            for i, val in enumerate(unique_values):
                x_idxs = np.hstack((x_idxs, (x[L_ind] == val).nonzero()[0][0]))
            x_interest[L_ind].extend(np.array(x[L_ind])[np.int32(x_idxs)])
            y_interest[L_ind].extend(np.array(y[L_ind])[np.int32(x_idxs)])
        plt.plot(x_interest[0], np.array(y_interest[1])-np.array(y_interest[0])) # There is: alpha = alpha blending value, between 0 (transparent) and 1 (opaque). There is: linewidths = linewidth of the marker edges. There is: edgecolors = the edge color of the marker. 
        plt.xlabel(r'$\ln(\tilde{n})$')
        plt.ylabel(r'$\Delta E/(k_n^2/(2m_b))$')
        dist_av = np.average(np.array(y_interest[1])-np.array(y_interest[0]))
        plt.xlim((config.n_min, config.n_max))
        plt.ylim((dist_av-1, dist_av+1))
        plt.savefig('{}/dist_pol_B{:.2f}L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.B, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")
        
def plotResidue(one_over_eta_vec, residue, start_time):
    """
    Plot residue"""
    print_status(start_time,'Starting plotResidue()')
    cs_list = [colormap(i) for i in np.linspace(0.0, 0.9, config.L_max+1-config.L_max*config.L_max_only)]
    plt.figure()
    for L_ind, L in enumerate(range(config.L_max*config.L_max_only, config.L_max+1)):
        plt.plot(one_over_eta_vec, residue[L_ind], color = cs_list[L_ind], label = "L={}".format(L))
    plt.xlim((config.n_min, config.n_max))
    plt.xlabel(r'$\ln(\tilde{n})$')
    plt.ylabel(r'Residue $Z=|\phi|^2$')
    plt.legend(fontsize = 'small')
    plt.savefig('{}/residue_B{:.2f}L_max{}N{}wW1{withW1}wW2{withW2}.pdf'.format(config.viz_dest, config.B, config.L_max, config.n_ph_max, withW1="True" if config.with_W_1 == True else "False", withW2="True" if config.with_W_2 == True else "False"), bbox_inches="tight")