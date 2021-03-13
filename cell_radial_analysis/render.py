#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Name:     Cell Radial Analysis
# Purpose:  An analysis library to be used in conjunction with BayesianTracker
#	    cell tracking software (a.lowe@ucl.ac.uk).
#	    Principally designed to scan over a population of cells and return
#	    the spatio-temporal distribution of antagonistic cellular events.
#	    For example, the distribution of wild-type cells around
#	    a mutant (Scr) apoptosis, and the corresponding distribution of
#           mitoses and subsequent probability distribution.
#
# Authors:  Nathan J. Day (day.nathan@pm.me)
#
#
# Created:  12/02/2021
# ------------------------------------------------------------------------------


__author__ = "Nathan J. Day"
__email__ = "day.nathan@pm.me"

import matplotlib.pyplot as plt
import tools
import numpy as np
import os

def plot_cumulative(input_2d_hist, num_bins, radius, t_range, title, label, cb_label, save_path, SI):

        xlocs, xlabels, ylocs, ylabels = tools.kymo_labels(num_bins, 1, radius, t_range, SI= True)

        if SI == True:
            time_unit = '(Hours)'
            distance_unit = '(Micrometers)'
        else:
            time_unit = '(Frames)'
            distance_unit = '(Pixels)'
        plt.xticks(xlocs, xlabels, rotation = 'vertical')
        plt.yticks(ylocs, ylabels)
        plt.xlabel("Time since apoptosis "+ time_unit)
        plt.ylabel("Distance from apoptosis "+ distance_unit)
        plt.title(title)

        final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
        plt.imshow(final_hist)


        plt.colorbar(label = cb_label)

        if save_path == '':
            return plt.imshow(final_hist)
        else:
            plt.savefig(os.path.join(save_path,title+'.pdf'), dpi = 300, bbox_inches = 'tight')
            print("Plot saved at ", (os.path.join(save_path,title+'.pdf')) )
            return plt.imshow(final_hist)

def plot_N_cells(input_2d_hist, subject_cells, target_cell, focal_time, radius, t_range):

    if target_cell.ID < 0:
        cell_type = 'Scr'
    if target_cell.ID > 0:
        cell_type = 'WT'
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = tools.kymo_labels(num_bins, 2, radius, t_range, SI = False)

    #expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    #plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation = 'vertical')
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time)+ ' (frames)')
    plt.ylabel("Distance from focal event (pixels)")
    plt.title('Kymograph for '+target_cell.fate.name+' ' +cell_type+' ID:'+str(cell_ID)+" at t= "+ str(focal_time))

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = 'Probability of wild-type mitoses'
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = 'Probability of Scribble mitoses? AMEND THIS LABEL'

    if event == 'APOPTOSIS':
        raise Exception('Apoptosis event counter not configured yet')

    plt.colorbar(label = cb_label)

    return plt.imshow(final_hist)

def plot_N_events(input_2d_hist, event, subject_cells, target_cell, focal_time, radius, t_range):

    if target_cell.ID < 0:
        cell_type = 'Scr'
    if target_cell.ID > 0:
        cell_type = 'WT'
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = tools.kymo_labels(num_bins, 2, radius, t_range, SI = False)

    #expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    #plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation = 'vertical')
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time)+ ' (frames)')
    plt.ylabel("Distance from focal event (pixels)")
    plt.title('Kymograph for '+target_cell.fate.name+' ' +cell_type+' ID:'+str(cell_ID)+" at t= "+ str(focal_time))

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = 'Probability of wild-type mitoses'
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = 'Probability of Scribble mitoses? AMEND THIS LABEL'
    if event == 'APOPTOSIS':
        raise Exception('Apoptosis event counter not configured yet')

    plt.colorbar(label = cb_label)

    return plt.imshow(final_hist)

def plot_P_events(input_2d_hist, event, subject_cells, target_cell, focal_time, radius, t_range):

    if target_cell.ID < 0:
        cell_type = 'Scr'
    if target_cell.ID > 0:
        cell_type = 'WT'
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = tools.kymo_labels(num_bins, 2, radius, t_range)


    #expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    #plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation = 'vertical')
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time)+ ' (frames)')
    plt.ylabel("Distance from focal event (pixels)")
    plt.title('Kymograph for '+target_cell.fate.name+' ' +cell_type+' ID:'+str(cell_ID)+" at t= "+ str(focal_time))

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = 'Probability of wild-type mitoses'
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = 'Probability of Scribble mitoses? AMEND THIS LABEL'
    if event == 'APOPTOSIS':
        raise Exception('Apoptosis event counter not configured yet')

    plt.colorbar(label = cb_label)

    return plt.imshow(final_hist)
