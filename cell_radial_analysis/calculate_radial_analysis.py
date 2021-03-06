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

import numpy as np
import tools
import re, os
from tqdm import tqdm


def N_cells(subject_cells, target_cell, radius, t_range, focal_time, num_bins):
    ## output just npy

    N_cells = tools.cell_counter(subject_cells, target_cell, radius, t_range, focal_time)
    N_cells_distance = [N_cells[i][1] for i in range(0,len(N_cells))]
    N_cells_time = [N_cells[i][2] for i in range(0,len(N_cells))]

    time_bin_edges = np.linspace((-(int(t_range/2))+focal_time),(int(t_range/2)+focal_time), num_bins+1) ## 2dimensionalise
    distance_bin_edges = np.linspace(0,radius, num_bins+1) ## 2dimensionalise

    N_cells_hist, x_autolabels, y_autolabels = np.histogram2d(N_cells_distance, N_cells_time, bins=[distance_bin_edges, time_bin_edges])

    return N_cells_hist

def N_events(event, subject_cells, target_cell, radius, t_range, focal_time, num_bins):
    ## output just npy
    if event == 'DIVIDE':

        N_events = tools.event_counter(event, subject_cells, target_cell, radius, t_range, focal_time)

        N_events_distance = [N_events[i][1] for i in range(0,len(N_events))]
        N_events_time = [N_events[i][2] for i in range(0,len(N_events))]

        time_bin_edges = np.linspace((-(int(t_range/2))+focal_time),(int(t_range/2)+focal_time), num_bins+1) ## 2dimensionalise
        distance_bin_edges = np.linspace(0,radius, num_bins+1) ## 2dimensionalise

        N_events_hist, x_autolabels, y_autolabels = np.histogram2d(N_events_distance, N_events_time, bins=[distance_bin_edges, time_bin_edges])

        ### include this in plotting function ###
        ##N_events_hist = np.flipud(N_events_hist) ## flip for desired graph orientation

    if event == 'APOPTOSIS':
        raise Exception('Apoptosis event counter not configured yet')

    return N_events_hist

def P_event(event, subject_cells, target_cell, radius, t_range, focal_time, num_bins):

    N_cells_hist = N_cells(subject_cells, target_cell, radius, t_range, focal_time, num_bins)
    N_events_hist = N_events(event, subject_cells, target_cell, radius, t_range, focal_time, num_bins)

    P_events_hist = N_events_hist/(N_cells_hist+1E-10)

    return P_events_hist


def iterative_heatmap_generator(subject_cells, subject_event, apoptosis_time_dict, tracking_filelist, radius, t_range, num_bins, output_path):
    hdf5_file_path, error_log, success_log = [], [], []
    cell_count = 0

    for apop_ID in tqdm(apoptosis_time_dict):
        #try:
        expt = 'GV' +str(re.findall(r"GV(\d+)", apop_ID)[0])
        position = re.findall(r"Pos(\d+)", apop_ID)[0]

        position = 'Pos' + position

        expt_position = os.path.join(expt,position,'') ## additional '' here so that / added to end of string
        cell_ID = int((re.findall(r"(\d+)_.FP", apop_ID))[0])
        print("ID", apop_ID)

        if expt_position not in hdf5_file_path:
            ## load that track data
            print('Loading', expt_position)
            hdf5_file_path = [hdf5_file_path for hdf5_file_path in tracking_filelist if expt_position in hdf5_file_path][0]
            wt_cells, scr_cells, all_cells = tools.load_tracking_data(hdf5_file_path)
            print('Loaded', expt_position)

        if 'RFP' in apop_ID:
            cell_ID = -cell_ID

        focal_time = int(apoptosis_time_dict[apop_ID])
        try:
            target_cell = [cell for cell in all_cells if cell.ID == cell_ID][0]
        except:
            error_message = apop_ID + ' could not find cell_ID'
            error_log.append(error_message)
            continue
        if target_cell.in_frame(focal_time):
            ## calculate according to subject cell type
            if subject_cells == 'WT':
                N_cells_hist = N_cells(wt_cells, target_cell, radius, t_range, focal_time, num_bins)
                N_events_hist = N_events(subject_event, wt_cells, target_cell, radius, t_range, focal_time, num_bins)
            if subject_cells == 'Scr':
                N_cells_hist = N_cells(scr_cells, target_cell, radius, t_range, focal_time, num_bins)
                N_events_hist = N_events(subject_event, scr_cells, target_cell, radius, t_range, focal_time, num_bins)
            if subject_cells == 'All':
                N_cells_hist = N_cells(all_cells, target_cell, radius, t_range, focal_time, num_bins)
                N_events_hist = N_events(subject_event, all_cells, target_cell, radius, t_range, focal_time, num_bins)

            N_cells_fn = os.path.join(output_path, (apop_ID + '_N_cells'))
            N_events_fn = os.path.join(output_path, (apop_ID + '_N_events'))
            np.save(N_cells_fn, N_cells_hist)
            np.save(N_events_fn, N_events_hist)
            success_message = N_cells_fn + ' / ' + N_events_fn + ' saved out successfully'
            success_log.append(success_message)
            cell_count += 1
        else:
            print('Focal time not in frame')
            error_message = apop_ID + ' apoptosis time t=' +str(focal_time) + ' not in cell range ' + str(range(target_cell.t[0], target_cell.t[-1]))
            error_log.append(error_message)

    return cell_count, error_log, success_log

## Is the below obselete now???
def cumulative_division_counter(apoptosis_time_dict, tracking_filelist, apoptoses_metadata_filelist, radius, t_range, focal_time, num_bins):
    hdf5_file_path, error_log = [], []
    cell_count = 0
    cumulative_N_cells_hist = np.zeros((num_bins, num_bins))
    cumulative_N_events_hist = np.zeros((num_bins, num_bins))

    for apop_ID in apoptosis_time_dict:
        expt = 'GV' +str(re.findall(r"GV(\d+)", apop_ID)[0])
        position = re.findall(r"Pos(\d+)", apop_ID)[0]
        position = 'Pos' + position
        expt_position = os.path.join(expt,position,'') ## additional '' here so that / added to end of string
        print("ID", apop_ID)

        if expt_position not in hdf5_file_path:
            ## load that track data
            print('Loading', expt_position)
            hdf5_file_path = [hdf5_file_path for hdf5_file_path in tracking_filelist if expt_position in hdf5_file_path][0]
            wt_cells, scr_cells, all_cells = tools.load_tracking_data(hdf5_file_path)
            print('Loaded', expt_position)

        ## get truest cell_ID from npz file along with time to realign chris' apop time
        path_to_npz = [path_to_npz for path_to_npz in apoptoses_metadata_filelist if apop_ID.split('FP')[0] in path_to_npz][0] ## this pulls the npz path from a list of npz paths by matching the apop_ID str prior to final label (ie fake/long_apop) as some of the final labels arent reflected in npz fn
        with np.load(path_to_npz) as npz:
            t = npz['t']
            cell_ID = int(npz['ID'])
        if 'RFP' in apop_ID:
            cell_ID = -cell_ID

        focal_time = apoptosis_time_dict[apop_ID]
        try:
            target_cell = [cell for cell in all_cells if cell.ID == cell_ID][0]
        except:
            error_message = apop_ID + ' could not find cell_ID'
            error_log.append(error_message)
            continue
        if target_cell.in_frame(focal_time):
            ## calculate -- NEED SUBJEcT CeLL choice and event choice
            cumulative_N_cells_hist += N_cells(wt_cells, target_cell, radius, t_range, focal_time, num_bins)
            cumulative_N_events_hist += N_events('DIVIDE', wt_cells, target_cell, radius, t_range, focal_time, num_bins)
            cell_count += 1
        else:
            print('Focal time not in frame!!!!!!!!!!!')
            error_message = apop_ID + ' apoptosis time t=' +str(focal_time) + ' not in cell range ' + str(range(target_cell.t[0], target_cell.t[-1]))
            error_log.append(error_message)
    return cumulative_N_cells_hist, cumulative_N_events_hist, cell_count, error_log
