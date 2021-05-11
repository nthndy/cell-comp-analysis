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
from pathlib import Path
import time

def auto_plot_cumulative(input_2d_hist, input_type, N, num_bins, radius, t_range, focal_cell, focal_event, subject_cell, subject_event, save_parent_dir, cbar_lim, SI):

        xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 1, radius, t_range, SI)

        ## formatting cell and event names
        focal_event_name = 'apoptoses' if 'apop' in focal_event.lower() else 'divisions' #focal_event == 'APOPTOSIS' or 'apop' else 'divisions'
        focal_cell_name = 'wild-type' if 'wt' in focal_cell.lower() else 'Scribble'
        subj_event_name = 'apoptoses' if 'apop' in subject_event.lower() else 'divisions'
        subj_cell_name = 'wild-type' if 'wt' in subject_cell.lower() else 'Scribble'

        if focal_event == 'control':
            focal_event_name = 'random time points'

        title = 'Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})'.format(subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N)

        ## output save path formatting
        save_dir_name = '{}_{}_{}_{}'.format(focal_cell.lower(), focal_event.lower()[0:3] if focal_event == 'DIVISION' else focal_event.lower()[0:4], subject_cell.lower(), subject_event.lower()[0:3] if subject_event == 'DIVISION' else subject_event.lower()[0:4])
        save_dir = '{}.{}.{}/{}'.format(radius,t_range,num_bins,save_dir_name)
        save_path = os.path.join(save_parent_dir,save_dir)
        Path(save_path).mkdir(parents=True, exist_ok=True)

        ## title formatting
        if input_type == 'N_cells':
            title = 'Spatiotemporal dist. of {} cells \n around {} {} (N={})'.format(subj_cell_name,focal_cell_name, focal_event_name, N)
            cb_label = 'Number of {} cell apperances'.format(subj_cell_name)
        if input_type == 'N_events':
            title = 'Spatiotemporal dist. of {} {} \n around {} {} (N={})'.format(subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N)
            cb_label = 'Number of {} {}'.format(subj_cell_name, subj_event_name)
        if input_type == 'P_events':
            title = 'Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})'.format(subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N)
            cb_label = 'Probability of {} {}'.format(subj_cell_name, subj_event_name)

        ## label unit formatting
        if SI == True:
            time_unit = '(Hours)'
            distance_unit = '(Micrometers)'
        else:
            time_unit = '(Frames)'
            distance_unit = '(Pixels)'

        ## plotting
        plt.xticks(xlocs, xlabels, rotation = 'vertical')
        plt.yticks(ylocs, ylabels)
        plt.xlabel("Time since apoptosis "+ time_unit)
        plt.ylabel("Distance from apoptosis "+ distance_unit)
        plt.title(title)

        final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
        plt.imshow(final_hist)

        if cbar_lim == '':
            plt.colorbar(label = cb_label)
        else:
            plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
            plt.colorbar(label = cb_label)

        ## apop location marker
        plt.scatter(num_bins/2-0.5, num_bins-0.75, s=20, c='white', marker='v')
        plt.text(num_bins+0.15, num_bins+1.5, 'Apoptosis location \nshown by inverted \nwhite triangle')

        fn = os.path.join(save_path,title+'.pdf')

        ## failsafe overwriting block
        if os.path.exists(fn):
            print("Filename", fn, "already exists, saving as updated copy")
            fn = fn.replace('.pdf', ' (updated {}).pdf'.format(time.strftime("%Y%m%d-%H%M%S")))


        ## save out?
        if save_parent_dir == '':
            return plt.imshow(final_hist)
        else:
            plt.savefig(fn, dpi = 300, bbox_inches = 'tight')
            print("Plot saved at ", fn)
            return plt.imshow(final_hist)


def plot_cumulative(input_2d_hist, num_bins, radius, t_range, title, label, cb_label, save_path, SI):

        xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 1, radius, t_range, SI)

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

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 2, radius, t_range, SI = False)

    #expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    #plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation = 'vertical')
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time)+ ' (frames)')
    plt.ylabel("Distance from focal event (pixels)")
    plt.title('Kymograph for '+target_cell.fate.name.lower()+' ' +cell_type+' ID:'+str(cell_ID)+" at t= "+ str(focal_time))

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = 'Number of wild-type cells'
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = 'Number of Scribble cells? AMEND THIS LABEL'

    # if event == 'APOPTOSIS':
    #     raise Exception('Apoptosis event counter not configured yet')

    plt.colorbar(label = cb_label)

    return plt.imshow(final_hist)

def plot_N_events(input_2d_hist, event, subject_cells, target_cell, focal_time, radius, t_range):

    if target_cell.ID < 0:
        cell_type = 'Scr'
    if target_cell.ID > 0:
        cell_type = 'WT'
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 2, radius, t_range, SI = False)

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

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 2, radius, t_range)


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


### labels in SI
def kymo_labels(num_bins, label_freq, radius, t_range, SI):
    label_freq =1
    radial_bin = radius / num_bins
    temporal_bin = t_range / num_bins

    if SI == True:
        time_scale_factor = 4/60 ## each frame is 4 minutes
        distance_scale_factor = 1/3 ## each pixel is 0.3recur micrometers
    else:
        time_scale_factor, distance_scale_factor = 1,1

    ### generate labels for axis micrometers/hours
    xlocs = range(0, num_bins,label_freq) ## step of 2 to avoid crowding
    xlabels = []
    for m in range(int(-num_bins/2), int(num_bins/2),label_freq):
        xlabels.append(str(int(((temporal_bin)*m)*time_scale_factor)) + "," + str(int(((temporal_bin)*m+temporal_bin)*time_scale_factor)))
    ylocs = range(0, num_bins, label_freq) ## step of 2 to avoid crowding
    ylabels = []
    for m in range(num_bins, 0, -label_freq):
        ylabels.append(str(int(((radial_bin)*m)*distance_scale_factor)) + "," + str(int(((radial_bin)*(m-1)*distance_scale_factor))))

    return xlocs, xlabels, ylocs, ylabels
