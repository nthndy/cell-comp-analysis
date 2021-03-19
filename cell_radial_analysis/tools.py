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
import os

def load_tracking_data(tracks_path):
    import btrack
    from btrack.utils import import_HDF, import_JSON
    print("Btrack version no.:", btrack.__version__)

    with btrack.dataio.HDF5FileHandler(tracks_path, 'r', obj_type = "obj_type_1") as h:
        wt_cells = h.tracks
    with btrack.dataio.HDF5FileHandler(tracks_path, 'r', obj_type = "obj_type_2") as h:
        scr_cells = h.tracks
    for i in range(len(scr_cells)):
        scr_cells[i].ID = -(scr_cells[i].ID)
    all_cells = wt_cells + scr_cells ### negative IDs are scribble cells
    print("Track information loaded and ordered according to cell type (WT IDs >0> Scr IDs")
    return wt_cells, scr_cells, all_cells

def select_target_cell(cell_type, cell_ID, all_cells):
    if cell_type == 'Scr':
        if int(cell_ID) > 0:
            cell_ID = -int(cell_ID)
        target_cell = [scr_track for scr_track in all_cells if scr_track.ID == cell_ID][0]
        ### if a scr cell is picked, the focal timepoint is its apoptosis
        try: ## try to find apoptosis time ## replace with chris' definitive definitions of apoptosis?
            if target_cell.label[0] == 'APOPTOSIS': ## if the first classification is apoptosis then thats a duff track
                print("False apoptosis (first classification is apoptosis) Scr ID:", target_cell.ID)
                focal_index = focal_time = False
            else:
                for i, j in enumerate(target_cell.label):
                    if j == 'APOPTOSIS' and target_cell.label[i+1] == 'APOPTOSIS' and target_cell.label[i+2] == 'APOPTOSIS': # and target_cell.label[i+3] =='APOPTOSIS' and target_cell.label[i+4] =='APOPTOSIS':
                        focal_index = i
                        break
                focal_time = target_cell.t[focal_index]
        except:
            print("False apoptosis (could not find t-apoptosis) Scr ID:", target_cell.ID)
            focal_index = focal_time = False
    elif cell_type == 'WT':
        cell_ID = int(cell_ID)
        import random
        target_cell = [wt_track for wt_track in all_cells if wt_track.ID == cell_ID][0]
        ### if a wt cell is picked, the focal time point is a random point in its track as this measurement will serve as a control
        focal_time = random.choice(target_cell.t)
        focal_index = target_cell.t.index(focal_time)
    else:
        raise Exception("Cell type not recognised, enter either 'WT' or 'Scr'")
    #if focal_time == False:
        #raise Exception("False apoptosis Scr ID it gone done messed up:", target_cell.ID)

    return target_cell, focal_time

def euc_dist(target_cell, other_cell, frame, focal_time):
    focal_index = target_cell.t.index(focal_time)
    try:
        idx0 = focal_index #target_cell.t.index(apop_time) ## could also do just ## apop_index
        idx1 = other_cell.t.index(frame) ## t.index provides the index of that frame
    except:
        return np.inf

    dx = target_cell.x[idx0] - other_cell.x[idx1]
    dy = target_cell.y[idx0] - other_cell.y[idx1]

    return np.sqrt(dx**2 + dy**2)

def cell_counter(subject_cells, target_cell, radius, t_range, focal_time):
    ## subject should equal wt_cells, scr_cells or all_cells
    focal_index = target_cell.t.index(focal_time)
    cells = [tuple(((cell.ID),
              (round((euc_dist(target_cell, cell, (focal_time+delta_t), focal_time)),2)),
              ((focal_time + delta_t))))
               for delta_t in range(-int(t_range/2), int(t_range/2))
               for cell in subject_cells
                   if euc_dist(target_cell, cell, focal_time + delta_t, focal_time)<radius
               ]
    return cells

def event_counter(event, subject_cells, target_cell, radius, t_range, focal_time):
    focal_index = target_cell.t.index(focal_time)
    ### the issue here is that it assumes mitosis time is the last time frame in a track (accurate) which does not work with apoptosis
    if event == 'APOPTOSIS':
        return print("Need to configure apoptosis counter, as current apoptosis timepoints are inaccurate")
    else:
    ## subject should equal wt_cells, scr_cells or all_cells
        events = [tuple(((cell.ID),
                          (round((euc_dist(target_cell, cell, cell.t[-1], focal_time)),2)),
                          ((cell.t[-1]))))
                           for cell in subject_cells
                               if euc_dist(target_cell, cell, cell.t[-1], focal_time)<radius and
                                  cell.t[-1] in range(focal_time-int(t_range/2), focal_time+ int(t_range/2)) and
                                  cell.fate.name == event
                           ]
        return events

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


### loading chris apoptosis information

def hdf5_file_finder(hdf5_parent_folder):
    ### load my list of hdf5 files from a typical directory tree with different experiments
    aligned_hdf5_file_list = glob.glob(os.path.join(hdf5_parent_folder, 'GV****/Pos*/*aligned/HDF/segmented.hdf5'))
    unaligned_hdf5_file_list = glob.glob(os.path.join(hdf5_parent_folder, 'GV****/Pos*/HDF/segmented.hdf5'))
    hdf5_file_list = aligned_hdf5_file_list + unaligned_hdf5_file_list

    return hdf5_file_list

def xy_position_counter(apoptosis_time_dict, tracking_filelist):
    hdf5_file_path, error_log = [], []
    cell_count = 0
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
            wt_cells, scr_cells, all_cells = tools.load_tracking_data(hdf5_file_path)
            print('Loaded', expt_position)

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
    return list_xy, cell_count, error_log

###obselete from below?

def apoptosis_list_loader(path_to_apop_list, filter_out):
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
