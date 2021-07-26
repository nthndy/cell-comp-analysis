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

import napari
import btrack
import numpy as np
from skimage.io import imread
import os
from btrack.utils import import_HDF, import_JSON, tracks_to_napari
from tqdm.notebook import tnrange, tqdm
import matplotlib.pyplot as plt
import tools
from datetime import datetime
import matplotlib.font_manager


"""
Graph rendering below

This section takes the final output of my radial analysis and renders the relevant graphs and labels
"""

def auto_plot_cumulative_defaulttext(input_2d_hist, input_type, N, num_bins, radius, t_range, focal_cell, focal_event, subject_cell, subject_event, save_parent_dir, cbar_lim, include_apop_bin, SI):

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
        plt.title(title+'\n', fontweight='bold')

        ## if include_apop_bin is true then the spatial bin containing the apoptotic cell (ie the central spatial bin of the radial scan) will be show in the graph, if false then it is cropped which ends up with a plot showing only the relevant local env not the site of apop (better imo)
        if include_apop_bin == True:
            final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
        else:
            final_hist = np.flipud(input_2d_hist[1:-1,:])

        plt.imshow(final_hist)

        if cbar_lim == '':
            #plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
            plt.colorbar(label = cb_label)
        else:
            plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
            plt.colorbar(label = cb_label)

        ## apop location marker
        if include_apop_bin == True:
            plt.scatter(num_bins/2-0.5, num_bins-0.75, s=20, c='white', marker='v')
            plt.text(num_bins+0.15, num_bins+1.5, 'Apoptosis location \nshown by inverted \nwhite triangle')
        else:
            plt.scatter(num_bins/2-0.5, num_bins-2-0.75, s=20, c='white', marker='v')
            plt.text(num_bins+0.15, num_bins+1.5-2, 'Apoptosis location \nshown by inverted \nwhite triangle')

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

def auto_plot_cumulative(input_2d_hist, input_type, N, num_bins, radius, t_range, focal_cell, focal_event, subject_cell, subject_event, save_parent_dir, cbar_lim, include_apop_bin, SI):

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
        font = {'fontname':'Liberation Mono'}
        plt.xticks(xlocs, xlabels, rotation = 'vertical', **font)
        plt.yticks(ylocs, ylabels, **font)
        plt.xlabel("Time since apoptosis "+ time_unit, **font)
        plt.ylabel("Distance from apoptosis "+ distance_unit, **font)
        plt.title(title+'\n', fontweight='bold', **font)

        ## if include_apop_bin is true then the spatial bin containing the apoptotic cell (ie the central spatial bin of the radial scan) will be show in the graph, if false then it is cropped which ends up with a plot showing only the relevant local env not the site of apop (better imo)
        if include_apop_bin == True:
            final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
        else:
            final_hist = np.flipud(input_2d_hist[1:-1,:])

        plt.imshow(final_hist)

        # if cbar_lim == '':
        #     plt.colorbar(label = cb_label, **font)
        # else:
        #     plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        #     plt.colorbar(label = cb_label, **font)




        ## apop location marker
        if include_apop_bin == True:
            plt.scatter(num_bins/2-0.5, num_bins-0.75, s=20, c='white', marker='v')
            plt.text(num_bins+0.15, num_bins+1.5, 'Apoptosis location \nshown by inverted \nwhite triangle', **font)
        else:
            plt.scatter(num_bins/2-0.5, num_bins-2-0.75, s=20, c='white', marker='v')
            plt.text(num_bins+0.15, num_bins+1.5-2, 'Apoptosis location \nshown by inverted \nwhite triangle', **font)


        # ### when manually editing the typeface of the colorbar
        # if cbar_lim == '':
        #     if include_apop_bin == False:
        #         plt.clim(vmin=np.min(P_events[1:-1,:]), vmax=np.max(P_events[1:-1,:]))
        #     else:
        #         plt.clim(vmin=np.min(P_events), vmax=np.max(P_events))
        #     plt.colorbar(label = cb_label)
        # else:
        #     plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        #     plt.colorbar(label = cb_label)

        ## colorbar
        if cbar_lim == '':
            if include_apop_bin == False:
                plt.clim(vmin=np.min(input_2d_hist[1:-1,:]), vmax=np.max(input_2d_hist[1:-1,:]))
            else:
                plt.clim(vmin=np.min(input_2d_hist), vmax=np.max(input_2d_hist))
            cb = plt.colorbar(label = cb_label)
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(family='Liberation Mono')
            text.set_font_properties(font)
            ax.set_yticklabels(np.round(ax.get_yticks(),5),**{'fontname':'Liberation Mono'}) ### cropped to 5dp
        else:
            plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
            cb = plt.colorbar(label = cb_label)
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(family='Liberation Mono')
            text.set_font_properties(font)
            ax.set_yticklabels(np.round(ax.get_yticks(),5),**{'fontname':'Liberation Mono'})

        ## filename
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

def kymo_labels(num_bins, label_freq, radius, t_range, SI):
    label_freq = 1
    radial_bin = radius / num_bins
    temporal_bin = t_range / num_bins

    if SI == True:
        time_scale_factor = 4/60 ## each frame is 4 minutes
        distance_scale_factor = 1/3 ## each pixel is 0.3recur micrometers
    else:
        time_scale_factor, distance_scale_factor = 1,1

    ### generate labels for axis micrometers/hours
    xlocs = np.arange(-0.5, num_bins, label_freq) ## -0.5 to start at far left border of first bin
    xlabels = []
    for m in np.arange(int(-num_bins/2), int(num_bins/2)+1,label_freq):
        xlabels.append(str(int(((temporal_bin)*m)*time_scale_factor)))# + "," + str(int(((temporal_bin)*m+temporal_bin)*time_scale_factor)))

    ylocs = np.arange(-0.5, num_bins, label_freq) ## -0.5 to start at far top border of first bin
    ylabels = []
    for m in np.arange(num_bins, 0-1, -label_freq):
        ylabels.append(str(int(((radial_bin)*m)*distance_scale_factor)))# + "," + str(int(((radial_bin)*(m-1)*distance_scale_factor))))

    return xlocs, xlabels, ylocs, ylabels

def old_kymo_labels(num_bins, label_freq, radius, t_range, SI):
    """
    This plots the labels in the middle of each bin whereas the new one plots labels on the edges of each bin
    """
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

def kymograph_compiler(raw_files_dir, radius, t_range, num_bins):
    """
    This function compiles kymograph from raw list files directory
    """
    ### find individual radial scan raw files
    N_cells_fns = natsorted(glob.glob(os.path.join(raw_files_dir, '*cells*')))
    #N_events_fns = natsorted(glob.glob(os.path.join(raw_files_dir,'*events*'))
    N_cells_cropped_hist_cumulative, N_events_cropped_hist_cumulative = np.zeros((num_bins, num_bins)), np.zeros((num_bins, num_bins))
    ### ascertain the maximum scan parameters of chosen files
    radius_max = int(re.findall(r'rad_(\d+)', N_cells_fns[0])[0])
    t_range_max = int(re.findall(r't_range_(\d+)', N_cells_fns[0])[0])

    # num_bins_uncropped = int(num_bins/radius * radius_max) ### maintains the same num_bins per radius over full extent of scan
    # N_cells_hist_cumulative, N_events_hist_cumulative = np.zeros((num_bins_uncropped, num_bins_uncropped)), np.zeros((num_bins_uncropped, num_bins_uncropped))

    N = len(N_cells_fns)
    ### iterate through all files compiling into cumulative
    for N_cells_fn in tqdm(N_cells_fns):

        ## find events fn
        N_events_fn = N_cells_fn.replace('cells','events')
        ## load data
        N_cells, N_events = [], []
        with open(N_cells_fn, newline='') as csvfile:
            N_cells_import_csv = csv.reader(csvfile, delimiter='\n')#, quotechar='|')
            for row in N_cells_import_csv:
                N_cells.append(eval(row[0]))
        with open(N_events_fn, newline='') as csvfile:
            N_events_import_csv = csv.reader(csvfile, delimiter='\n')#, quotechar='|')
            for row in N_events_import_csv:
                N_events.append(eval(row[0]))

        #focal_time = int(re.findall(r'focal_t_(\d+)', N_cells_fn)[0])
        ### temporary fix
        if N_cells[0][2] != 0: ## ie if the earliest point of scan isnt clipped by beginning of movie
            focal_time = int(t_range_max/2 + N_cells[0][2])
        elif N_cells[-1][2] != 1175: ### ie if the end of the scan isn't clipped by the end of the movie
            focal_time = int(N_cells[-1][2] - t_range_max/2)
        #print('maximum extent of scan and focal time:', radius_max, t_range_max, focal_time)

    #     ### make maximum extent distance/time lists
    #     N_cells_distance = [N_cells[i][1] for i in range(0,len(N_cells))]
    #     N_cells_time = [N_cells[i][2] for i in range(0,len(N_cells))]
    #     N_events_distance = [N_events[i][1] for i in range(0,len(N_events))]
    #     N_events_time = [N_events[i][2] for i in range(0,len(N_events))]
    #     ### full extent N_cells/N_events histograms
    #     time_bin_edges = np.linspace((-(int(t_range_max/2))+focal_time),(int(t_range_max/2)+focal_time), num_bins_uncropped+1) ## 2dimensionalise
    #     distance_bin_edges = np.linspace(0,radius_max, num_bins_uncropped+1) ## 2dimensionalise
    #     N_cells_hist, x_autolabels, y_autolabels = np.histogram2d(N_cells_distance, N_cells_time, bins=[distance_bin_edges, time_bin_edges])
    #     N_events_hist,  x_autolabels, y_autolabels = np.histogram2d(N_events_distance, N_events_time, bins=[distance_bin_edges, time_bin_edges])
    #     ### make cumulative histogram
    #     N_cells_hist_cumulative += N_cells_hist
    #     N_events_hist_cumulative += N_events_hist

        ### make cropped list of distance and time
        N_cells_cropped = [i for i in N_cells if i[1] < radius and (focal_time - (t_range/2)) <= i[2] < (focal_time + (t_range/2))]
        N_cells_distance_cropped = [N_cells_cropped[i][1] for i in range(0,len(N_cells_cropped))]
        N_cells_time_cropped = [N_cells_cropped[i][2] for i in range(0,len(N_cells_cropped))]
        N_events_cropped = [i for i in N_events if i[1] < radius and (focal_time - (t_range/2)) <= i[2] < (focal_time + (t_range/2))]
        N_events_distance_cropped = [N_events_cropped[i][1] for i in range(0,len(N_events_cropped))]
        N_events_time_cropped = [N_events_cropped[i][2] for i in range(0,len(N_events_cropped))]
        ### cropped N_cells/N_events histogram
        time_bin_edges = np.linspace((-(int(t_range/2))+focal_time),(int(t_range/2)+focal_time), num_bins+1) ## 2dimensionalise
        distance_bin_edges = np.linspace(0,radius, num_bins+1) ## 2dimensionalise
        N_cells_cropped_hist, x_autolabels, y_autolabels = np.histogram2d(N_cells_distance_cropped, N_cells_time_cropped, bins=[distance_bin_edges, time_bin_edges])
        N_events_cropped_hist,  x_autolabels, y_autolabels = np.histogram2d(N_events_distance_cropped, N_events_time_cropped, bins=[distance_bin_edges, time_bin_edges])
        ## make cumulative cropped histogram
        N_cells_cropped_hist_cumulative += N_cells_cropped_hist
        N_events_cropped_hist_cumulative += N_events_cropped_hist

    P_events = N_events_cropped_hist_cumulative/N_cells_cropped_hist_cumulative

    ### info for rendering of graph ### change with new fns when ready
    focal_cell = re.findall(r'(?<=Pos._)[a-zA-Z]+',N_cells_fns[0])[0]
    subject_cell = re.findall(r'(?<=N_cells_)[a-zA-Z]+',N_cells_fns[0])[0]
    focal_event = 'apoptosis'
    subject_event = 'division'

    return N_cells_cropped_hist_cumulative, N_events_cropped_hist_cumulative, P_events

"""
Raw data viewer rendering WIP below, needs work as doesnt recognise `viewer` atm

This module contains functions to assist in the rendering and display of data from my radial analysis of cell competition
"""

def find_apoptosis_time(target_track, index): ### if index is set to True then the index of the apoptotic time (wrt target_track) is returned
    """
    This takes a target track and finds the apoptosis time, returning it as an absolute time if index == False, or a time relative to cell life (index) if index == True
    """
    for i, j in enumerate(target_track.label):
        if j == 'APOPTOSIS' and target_track.label[i+1] == 'APOPTOSIS' and target_track.label[i+2] == 'APOPTOSIS': # and target_track.label[i+3] =='APOPTOSIS' and target_track.label[i+4] =='APOPTOSIS':
            apop_index = i
            break
    apop_time = target_track.t[apop_index]
    if index == True:
        return apop_index
    else:
        return apop_time

def find_nearby_wt_mitosis(target_track, delta_t, radius):
    """
    This takes a target track and finds the nearby wild type mitoses, returning both the wild-type tracks and mitoses in specified radius and delta_t time window
    """
    frame = find_apoptosis_time(target_track, index = False) + delta_t
    dividing_states = ('METAPHASE',) #('PROMETAPHASE', 'METAPHASE', 'DIVIDE')
    wt_tracks_in_radius = [wt_track for wt_track in wt_tracks if wt_track.in_frame(frame) if euclidean_distance(target_track, wt_track, frame)<radius]
    wt_mitosis_in_radius = [wt_track for wt_track in wt_tracks if wt_track.in_frame(frame) if euclidean_distance(target_track, wt_track, frame)<radius if wt_track.label[wt_track.t.index(frame)] in dividing_states if wt_track.fate.name == "DIVIDE"] ###check this

    return wt_tracks_in_radius, wt_mitosis_in_radius


def plot_mitoses(cell_type, cell_ID, radius, delta_t): ## this function plots mitosis events into the napari viewer
    """
    This function takes a cell_type, a focal cell ID, a spatial radius and a time window and finds all the mitotic cells belonging to cell type within the radius and time window, plotting them as points in napari.
    """
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    apop_time, apop_index = find_apoptosis_time(target_track, index = False), find_apoptosis_time(target_track, index = True)
    apop_event = target_track.t[apop_index], target_track.x[apop_index]+shift_y, target_track.y[apop_index]+shift_x ## with transposed shift
    wt_tracks_in_radius, wt_mitosis_in_radius = find_nearby_wt_mitosis(target_track, delta_t, radius)
    t_m, x_m, y_m = np.zeros(len(wt_mitosis_in_radius)), np.zeros(len(wt_mitosis_in_radius)), np.zeros(len(wt_mitosis_in_radius))
    mito_events = np.zeros((len(wt_mitosis_in_radius), 3)) ## 3 because of the 3 cartesian coords
    for i, wt_mitosis in enumerate(wt_mitosis_in_radius): ## this now assumes that the mitosis time point of relevance isnt the last frame of track but the time at delta_t, need to bolster definition of mitosis
        mito_index = [j for j, k in enumerate(wt_mitosis.t) if k == apop_event[0]+delta_t][0] ### [0] bc first item of list comprehension
        t_m[i], x_m[i], y_m[i] = wt_mitosis.t[mito_index], wt_mitosis.x[mito_index]+shift_y, wt_mitosis.y[mito_index]+shift_x ## plus transposed coordinate shift
        mito_events[i] = t_m[i], x_m[i], y_m[i]
    return viewer.add_points(mito_events, name = "Mitosis events", symbol = "cross", face_color = 'pink')

def plot_target_track(cell_type, cell_ID):
    """
    This takes a cell_type and target cell ID and plots it as a point in napari
    """
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    target_track_loc = [(target_track.t[i], target_track.x[i]+shift_y, target_track.y[i]+shift_x) for i in range(len(target_track.t))]
    return viewer.add_points(target_track_loc, name = "Track of interest", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)

def plot_stationary_apoptosis_point(cell_type, cell_ID): ## this function plots apoptotic event and surrounding local environment scope (determined by radius)
    """
    This takes a cell type and cell ID and plots the apoptosis point at the time of apoptosis only
    """
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    apop_time, apop_index = find_apoptosis_time(target_track, index = False), find_apoptosis_time(target_track, index = True)
    apop_event = [(t, target_track.x[apop_index]+shift_y, target_track.y[apop_index]+shift_x) for t in range(len(gfp))] ## marker for apoptosis over all frames
    return viewer.add_points(apop_event, name = "Stastionary apoptosis point", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)

def plot_stationary_apop_radius(cell_type, cell_ID, radius, delta_t, inner_radius):
    """
    This takes a cell type and cell ID and plots the apoptosis with a radius and optional inner ring at the time specified as delta_t either side of apop time
    """
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    apop_time, apop_index = find_apoptosis_time(target_track, index = False), find_apoptosis_time(target_track, index = True)
    apop_event = target_track.t[apop_index], target_track.x[apop_index]+shift_y, target_track.y[apop_index]+shift_x ## with transposed shift, just for the frame of apoptosis
    outer_radial_bin = [tuple(((apop_event[0]+t, apop_event[1]-radius, apop_event[2]-radius),
                               (apop_event[0]+t, apop_event[1]+radius, apop_event[2]-radius),
                               (apop_event[0]+t, apop_event[1]+radius, apop_event[2]+radius),
                               (apop_event[0]+t, apop_event[1]-radius, apop_event[2]+radius)))
                                for t in range(-abs(delta_t), +abs(delta_t)+1)]
    if inner_radius > 0:
        inner_radial_bin = [tuple(((apop_event[0]+t, apop_event[1]-inner_radius, apop_event[2]-inner_radius),
                                   (apop_event[0]+t, apop_event[1]+inner_radius, apop_event[2]-inner_radius),
                                   (apop_event[0]+t, apop_event[1]+inner_radius, apop_event[2]+inner_radius),
                                   (apop_event[0]+t, apop_event[1]-inner_radius, apop_event[2]+inner_radius)))
                                    for t in range(-abs(delta_t), +abs(delta_t)+1)]
        return viewer.add_shapes(outer_radial_bin,opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment'), viewer.add_shapes(inner_radial_bin, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Inner Radial environment')
    else:
        return viewer.add_shapes(outer_radial_bin, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment')

def plot_radius(cell_type, cell_ID, radius):
    """
    This takes a cell type and cell ID and plots a radius around that cell for the cells life time
    """
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [tuple(((t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x-radius),
                   (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x-radius),
                   (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x+radius),
                   (t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x+radius)))
                    for i,t in enumerate(range(target_track.t[0], target_track.t[-1]))]
    return viewer.add_shapes(radius_shape, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment')

def plot_post_track_radius(cell_type, cell_ID, radius):
    """
    This takes a cell type and cell ID and plots a radius around that cell after that cell had died/disappeared
    """
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [tuple(((t, target_track.x[-1]+shift_y-radius, target_track.y[-1]+shift_x-radius),
                   (t, target_track.x[-1]+shift_y+radius, target_track.y[-1]+shift_x-radius),
                   (t, target_track.x[-1]+shift_y+radius, target_track.y[-1]+shift_x+radius),
                   (t, target_track.x[-1]+shift_y-radius, target_track.y[-1]+shift_x+radius)))
                    for i,t in enumerate(range(target_track.t[-1],len(gfp)))]
    return viewer.add_shapes(radius_shape, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Post-apoptosis radial environment')

def plot_fragmented_track(list_of_IDs): ### not using this below as dont think output is correct
    """
    This takes a list of cell IDs as a fragmented track and plots a radius around the location of each fragment
    """
    compiled_frag_track_loc = []
    compiled_frag_radius_loc = []
    for cell_ID in list_of_IDs:
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
        #plot_radius(target_track)
        #plot_target_track(target_track)
        radius_loc = plot_frag_radius(target_track)
        compiled_frag_radius_loc+= radius_loc
        target_track_loc = plot_frag_target_track(target_track)
        compiled_frag_track_loc += target_track_loc
    return viewer.add_shapes(compiled_frag_radius_loc, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment'), viewer.add_points(compiled_frag_track_loc, name = "Track of interest", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)

def plot_frag_target_track(target_track):
    """
    This takes a fragmented track, currently modelled on example cell 17 and provides the location of the cell whilst it is existent and then provides an alternate fragmented track after
    """
    if target_track.ID == 17:
        target_track_loc = [(target_track.t[i], target_track.x[i]+shift_y, target_track.y[i]+shift_x) for i in range(len(target_track.t))]
        return target_track_loc #viewer.add_points(target_track_loc, name = "Track of interest", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)
    else:
        target_track_loc = [(target_track.t[i], target_track.x[i]+shift_y, target_track.y[i]+shift_x) for i in range(len(target_track.t)) if target_track.t[i]> 742]
        return target_track_loc#viewer.add_points(target_track_loc, name = "Track of interest", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)

def plot_frag_radius(target_track):
    """
    This takes a fragmented track, currently modelled on example cell 17 and provides the location of the cellradius whilst it is existent and then provides an alternate fragmented track after
    """
    if target_track.ID ==17:### this if condition is to avoid double plotting radii as fragmented tracks exist at same time
        radius_shape = [tuple(((t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x-radius),
                       (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x-radius),
                       (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x+radius),
                       (t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x+radius)))
                        for i,t in enumerate(range(target_track.t[0], target_track.t[-1]))]
        return radius_shape
    else:
        radius_shape = [tuple(((t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x-radius),
                       (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x-radius),
                       (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x+radius),
                       (t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x+radius)))
                        for i,t in enumerate(range(target_track.t[0], target_track.t[-1])) if t>741]
        return radius_shape

def plot_radii(cell_type, target_track, radius, num_bins):
    """
    This takes a cell type, target track, radius and number of bins and plots the radius/number of bins as concentric circles following the target track
    """
    print('This can be very time consuming for >10 bins, consider using single_frame radius')
    radii = range(int(radius/num_bins), radius+int(radius/num_bins), int(radius/num_bins))
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [tuple(((t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x-radius),
                   (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x-radius),
                   (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x+radius),
                   (t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x+radius)))
                    for i,t in enumerate(range(target_track.t[0], target_track.t[-1]))
                    for radius in radii]
    #return radius_shape
    return viewer.add_shapes(radius_shape, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment')

def plot_stationary_radii(cell_type, target_track, radius, num_bins):
    """
    This takes a cell type, target track, radius and number of bins and plots the radius/number of bins as concentric circles stationary after the target track ceases to exist
    """
    print('This can be very time consuming for >10 bins, consider using single_frame radius')
    radii = range(int(radius/num_bins), radius+int(radius/num_bins), int(radius/num_bins))
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [tuple(((t, target_track.x[-1]+shift_y-radius, target_track.y[-1]+shift_x-radius),
                   (t, target_track.x[-1]+shift_y+radius, target_track.y[-1]+shift_x-radius),
                   (t, target_track.x[-1]+shift_y+radius, target_track.y[-1]+shift_x+radius),
                   (t, target_track.x[-1]+shift_y-radius, target_track.y[-1]+shift_x+radius)))
                    for i,t in enumerate(range(target_track.t[-1]+1,len(gfp)))
                    for radius in radii]
    #return radius_shape
    return viewer.add_shapes(radius_shape, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment')

def plot_single_frame_radii(cell_type, target_track, radius, num_bins, frame):
    """
    This takes a cell type, target track, radius, number of bins and a frame and plots the radius/number of bins as concentric circles at that frame time point only
    """
    cell_ID = target_track
    t = frame
    if cell_type.lower() == 'scr':
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]

    try:
        i = target_track.t.index(t)
    except:
        i=-1
    radii = range(int(radius/num_bins), radius+int(radius/num_bins), int(radius/num_bins))
    radius_shape = [tuple(((t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x-radius),
                   (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x-radius),
                   (t, target_track.x[i]+shift_y+radius, target_track.y[i]+shift_x+radius),
                   (t, target_track.x[i]+shift_y-radius, target_track.y[i]+shift_x+radius)))
                    for radius in radii]
    #return radius_shape
    return viewer.add_shapes(radius_shape, opacity = 1, shape_type = 'ellipse', face_color = 'transparent', edge_color = 'cyan', edge_width = 5, name = 'Radial environment')
