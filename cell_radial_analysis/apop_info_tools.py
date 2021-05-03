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
import random
import json
import glob
import os, re
from tqdm import tqdm

### loading chris apoptosis information

def hdf5_file_finder(hdf5_parent_folder):
    ### load my list of hdf5 files from a typical directory tree with different experiments
    aligned_hdf5_file_list = glob.glob(os.path.join(hdf5_parent_folder, 'GV****/Pos*/*aligned/HDF/segmented.hdf5'))
    unaligned_hdf5_file_list = glob.glob(os.path.join(hdf5_parent_folder, 'GV****/Pos*/HDF/segmented.hdf5'))
    hdf5_file_list = aligned_hdf5_file_list + unaligned_hdf5_file_list

    return hdf5_file_list

def xy_position_counter(apoptosis_time_dict, tracking_filelist, apop_dict, num_bins):
    hdf5_file_path, error_log = [], []
    cell_count, expt_count = 0, 0
    cumulative_N_cells_hist = np.zeros((num_bins, num_bins))
    cumulative_N_events_hist = np.zeros((num_bins, num_bins))
    list_xy = []
    for apop_ID in tqdm(apoptosis_time_dict):
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
            wt_cells, scr_cells, all_cells = load_tracking_data(hdf5_file_path)
            print('Loaded', expt_position)
            expt_count += 1

        if 'RFP' in apop_ID:
            cell_ID = -cell_ID

        focal_time = int(apop_dict[apop_ID])
        try:
            target_cell = [cell for cell in all_cells if cell.ID == cell_ID][0]
        except:
            error_message = apop_ID + ' could not find cell_ID'
            error_log.append(error_message)
            continue
        if target_cell.in_frame(focal_time):

            target_cell.t.index(focal_time)
            x, y = target_cell.x[target_cell.t.index(focal_time)], target_cell.y[target_cell.t.index(focal_time)]
            list_xy.append([x,y])
            cell_count += 1
        else:
            print('Focal time not in frame!!!!!!!!!!!')
            error_message = apop_ID + ' apoptosis time t=' +str(focal_time) + ' not in cell range ' + str(range(target_cell.t[0], target_cell.t[-1]))
            error_log.append(error_message)
    return list_xy, cell_count, expt_count, error_log

def apoptosis_list_loader(path_to_apop_lists, cell_type):
    expts_apop_lists = os.listdir(path_to_apop_lists)
    apop_dict = {}
    N_apops = len(expts_apop_lists)
    for expt_apop_list in expts_apop_lists:
        apop_list = open(os.path.join(path_to_apop_lists, expt_apop_list), 'r')
        for apop_ID in apop_list:
            if cell_type in apop_ID:
                if 'stitched' not in apop_ID: ## relic of apoptosis finding (stitched = tracks that apoptosis switches into post apop)
                    apop_dict[apop_ID.split()[0]] = apop_ID.split()[1]
    return apop_dict

###obselete from below?

def apoptosis_list_loader2(path_to_apop_list, filter_out):
     ## loads a chris style JSON file with apop_ID and apop_index (from glimpse)
    ## filter option excludes GFP apoptoses and any apoptoses where glimpse does not match associated metadata
    ## assumes metadata stored in dir next to JSON called 'All_Apop_Npzs'

    with open(path_to_apop_list) as file:
        apop_dict = json.load(file)

    if filter_out != '':
        ## filter out GFP or whatever str is in filter_out
        for i in [i for i in apop_dict if filter_out in i]: # iterating over list with 'GFP' in apop_ID
            del apop_dict[i]

    return apop_dict

def old_apoptosis_list_loader(path_to_apop_list, filter_out):
    ## loads a chris style JSON file with apop_ID and apop_index (from glimpse)
    ## filter option excludes GFP apoptoses and any apoptoses where glimpse does not match associated metadata
    ## assumes metadata stored in dir next to JSON called 'All_Apop_Npzs'

    with open(path_to_apop_list) as file:
        apop_dict = json.load(file)

    if filter_out != '':
        ## filter out GFP or whatever str is in filter_out
        for i in [i for i in apop_dict if filter_out in i]: # iterating over list with 'GFP' in apop_ID
            del apop_dict[i]

    ## filter out (delete dict entry) if glimpse does not match its metadata
    ## metadata in the form of npz files stored in 'All_Apop_Npzs' dir in path_to_apop_list
    apop_npz_path = os.path.join(os.path.split(path_to_apop_list)[0], 'All_Apop_Npzs')
    apop_npz_list = glob.glob(os.path.join(apop_npz_path,'*.npz'))
    corrupt_metadata_list = []
    for apop_ID in apop_dict:
        path_to_npz = [path_to_npz for path_to_npz in apop_npz_list if apop_ID.split('FP')[0] in path_to_npz] ## this pulls the npz path from a list of npz paths by matching the apop_ID str prior to final label (ie fake/long_apop) as some of the final labels arent reflected in npz fn
        if len(path_to_npz) == 0: ## no metadata found
            corrupt_metadata_list.append(apop_ID)
            continue
        with np.load(path_to_npz[0]) as npz:
            t = npz['t']
            glimpse_enc = npz['glimpse_encoding'][0]
        if len(glimpse_enc) > len(t): ## if the glimpse encoding is longer than the time metadata then the two do not match and i cannot find the true apoptosis time of that apop_ID
            #print(apop_ID, ' metadata does not match glimpse, apoptosis time cannot be found')
            corrupt_metadata_list.append(apop_ID)
    for i in corrupt_metadata_list:
        if i in apop_dict:
            del apop_dict[i]

    return apop_dict

### out to be used in conjuction with previous funcs
def apop_time_realign(apop_dict, path_to_metadata):
    ### Chris' apop list gives time of apoptosis relative to start of glimpse
    ### This function reads metadata of glimpse and finds t0 for glimpse
    ### true_apop_time = t0(glimpse) + glimpse apop_time
    true_apop_dict = {}
    apop_npz_list = glob.glob(os.path.join(path_to_metadata,'*.npz'))
    for apop_ID in apop_dict:
        path_to_npz = [path_to_npz for path_to_npz in apop_npz_list if apop_ID.split('FP')[0] in path_to_npz][0] ## this pulls the npz path from a list of npz paths by matching the apop_ID str prior to final label (ie fake/long_apop) as some of the final labels arent reflected in npz fn
        with np.load(path_to_npz) as npz:
            t = npz['t']
            ID = npz['ID']
        true_apop_time = int(t[0]) + apop_dict[apop_ID]
        true_apop_dict[apop_ID] = true_apop_time

    return true_apop_dict
