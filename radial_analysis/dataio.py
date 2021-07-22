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
import btrack
from btrack.utils import import_HDF, import_JSON, tracks_to_napari


def load_tracking_data(tracks_path):
    """
    Tracks loader takes a path and returns the ordered tracking data for two populations of cells (i.e. wild-type and scr-kd)
    The wild-type cells will have positive integer IDs and the mutant population will have negative integer IDs
    """
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
    print("Track information loaded and ordered according to cell type (WT IDs >0> Scr IDs)")
    return wt_cells, scr_cells, all_cells

def apoptosis_list_loader(path_to_apop_lists, focal_cell):
    """
    load a list of apoptoses from typical list of .txt files detailing apoptosis times
    """
    if focal_cell == 'Scr':
        cell_type = 'RFP'
    if focal_cell == 'WT':
        cell_type = 'GFP '
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

def hdf5_file_finder(hdf5_parent_folder):
    """
    load my list of hdf5 tracking files from a typical directory tree with different experiments
    """
    aligned_hdf5_file_list = glob.glob(os.path.join(hdf5_parent_folder, 'GV****/Pos*/*aligned/HDF/segmented.hdf5'))
    unaligned_hdf5_file_list = glob.glob(os.path.join(hdf5_parent_folder, 'GV****/Pos*/HDF/segmented.hdf5'))
    hdf5_file_list = aligned_hdf5_file_list + unaligned_hdf5_file_list
    if len(hdf5_file_list) < 1:
        print("No HDF5 files found")
    return hdf5_file_list

def xy_position_counter(apoptosis_time_dict, tracking_filelist, apop_dict, num_bins):
    """
    A combination of functions from above that extract the xy positions of apoptoses
    """
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


def load_and_align_tracks(tracks_path, image_stack):
    """
    This takes the path to a tracking file and loads the tracks, realigning them to be displayed over image_stack, which must also be provided so that the shape can be assessed
    """

    global wt_tracks, scr_tracks, shift_y, shift_x

    with btrack.dataio.HDF5FileHandler(tracks_path, 'r', obj_type = "obj_type_1") as hdf:
        wt_tracks = hdf.tracks
    with btrack.dataio.HDF5FileHandler(tracks_path, 'r', obj_type = "obj_type_2") as hdf:
        scr_tracks = hdf.tracks

    ## this method casues problems as the viewer for tracks doesnt like negative numbers
    #wt_tracks, scr_tracks, all_tracks = tools.load_tracking_data(tracks_path)

    print("Tracks loaded")

    ### finding coord range of aligned images, coords switched already ## need to sort out the order of try excepts
    try:
        align_x_range, align_y_range = image_stack.shape[2], image_stack.shape[1]
    except:
        print("Error: no image data loaded to map tracks to")
        return


    ### finding maximum extent of tracking coords
    tracks_x_range = round(max([max(track.x) for track in wt_tracks]))
    tracks_y_range = round(max([max(track.y) for track in wt_tracks])) + 2 ## sort this hack out later

    ### coord switch
    tmp = tracks_y_range
    tracks_y_range = tracks_x_range
    tracks_x_range = tmp

    print("tracks range:", (tracks_x_range), (tracks_y_range))
    print("aligned image range:", (align_x_range), (align_y_range))

    shift_x = int((align_x_range - tracks_x_range)/2)
    shift_y = int((align_y_range - tracks_y_range)/2)

    print("shift in x and y:", shift_x, shift_y)

    global wt_data, scr_data, properties, graph

    wt_data, properties, graph = tracks_to_napari(wt_tracks, ndim = 2)
    scr_data, properties, graph = tracks_to_napari(scr_tracks, ndim = 2)

    tmp = wt_data[:,2].copy() ## copy the true_y coord
    wt_data[:,2] = wt_data[:,3]  ##assign the old_y coord as the true_x
    wt_data[:,3] = tmp ## assign the old_x as true_y

    wt_data[:,2] += shift_y ## TRUE_Y (vertical axis)
    wt_data[:,3] += shift_x ## TRUE_X (horizontal axis)

    tmp = scr_data[:,2].copy()
    scr_data[:,2] = scr_data[:,3]
    scr_data[:,3] = tmp

    scr_data[:,2] += shift_y ## TRUE_Y (vertical axis)
    scr_data[:,3] += shift_x ## TRUE_X (horizontal axis)

    print("coordinate shift applied")

    return wt_tracks, scr_tracks, wt_data, scr_data, shift_x, shift_y