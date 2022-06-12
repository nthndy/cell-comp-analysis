#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Name:     Cell Radial Analysis
# Purpose:  An analysis library to be used in conjunction with BayesianTracker
# 	    cell tracking software (a.lowe@ucl.ac.uk).
# 	    Principally designed to scan over a population of cells and return
# 	    the spatio-temporal distribution of antagonistic cellular events.
# 	    For example, the distribution of wild-type cells around
# 	    a mutant (Scr) apoptosis, and the corresponding distribution of
#           mitoses and subsequent probability distribution.
#
# Authors:  Nathan J. Day (day.nathan@pm.me)
#
#
# Created:  12/02/2021
# ------------------------------------------------------------------------------


__author__ = "Nathan J. Day"
__email__ = "day.nathan@pm.me"

import glob
import json
import os
import random
import re

import btrack
import numpy as np
import pandas as pd
from tools import basic_euc_dist, focal_xyt_finder
#from btrack.utils import import_HDF, import_JSON, tracks_to_napari
from tqdm.auto import tqdm


def eliminate_duplicates(N_cells_df, R_max, t_range, bins):
    """
    In the radial analysis, division events (or other rare events) are naturally
    counted only once as they exists as a transient state classifications.
    However cell counts are repeated for every frame that cell appears in over
    any one temporal bin. This function eliminates duplicate counts for a given
    cell count dataframe by segregating it into single temporal bin dataframes,
    removing the duplicates for a single temporal bin, then concatenating those
    dataframes back together

    N_cells_df : pd.DataFrame
        Radial scan dataframe of cells or divisions from focal scan, need Cell ID
        and Focal ID and formatted in the final form with Time Since Apop and
        Distance since apoptosis etc

    R_max : int
        Maximum spatial scan of DataFrame

    t_range : tuple
        Temporal range of scan

    bins : int
        Number of bins to seperate temporal range into, crucial variable as need
        to know which duplicates to delete per bin

    """
    ### discretise the temporal range
    disc_t_range = np.linspace(t_range[0], t_range[1], bins+1)
    ### placeholder dataframe
    N_cells_unique_df = pd.DataFrame(columns=N_cells_df.columns)
    for n, i in enumerate(disc_t_range):
        ### create new df with only one spatial bin
        disc_temp_df = N_cells_df.loc[(N_cells_df['Time since apoptosis'] >= disc_t_range[n]) & (N_cells_df['Time since apoptosis'] <= disc_t_range[n+1]) & (N_cells_df['Distance from apoptosis'] <= R_max)]
        ### drop duplicate cells in that one temporal bin
        disc_temp_df.drop_duplicates(subset=['Cell ID', 'Focal ID'], inplace = True, keep = 'first')
        ### append that unique temporal bin to new df
        N_cells_unique_df = N_cells_unique_df.append(disc_temp_df, ignore_index=True)
        ### break the iterations as the last temporal bin reached
        if n == len(disc_t_range) -2:
            break
    return N_cells_unique_df


def load_focal_df(focal_ID):
    """
    load dataframe with focal cell spatiotemp information in it

    focal_ID : str
        String with the necessary info to find focal id csv file
        ie. GV0819_Pos3_Scr_-1735
    """

    if 'Scr' in focal_ID:
        focal_ID = focal_ID.replace('Scr_-','')+'_RFP'
        focal_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/apoptotic_tracks',
                                 f'{focal_ID}.csv')
    if 'wt' in focal_ID:
        focal_ID = focal_ID.replace('wt_','')+'_GFP'
        if 'wt_control_wt_div' in file:
            focal_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/control_event_tracks',
                                 f'{focal_ID}.csv')
        if 'wt_apop_wt_div' in file:
            focal_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/apoptotic_tracks',
                                     f'{focal_ID}.csv')
    focal_df = pd.read_csv(focal_xyt_fn)
    del focal_df['Unnamed: 0']

    return focal_df

def convert_csv_to_feather(file_list):
    """
    Convert all csv files in file list to the feather format for quicker io of dataframes
    """
    ### convert to feather
    for file in tqdm(file_list):
        ### apop id
        apop_ID = file.split('/')[-1].split('_N_')[0]
        ### if filtering is present but not strict then just exclude time points outside fov
        apo_t, apo_x, apo_y = [int(re.search(r'txy_(\d+)_(\d+)_(\d+)', file)[i]) for i in range(1,4)]
        ### load csv
        df = pd.read_csv(file, names = ['Cell ID', 'Distance from apoptosis', 'Frame', 'x', 'y'])
        if len(df) == 0:
            print(f"{file} seems to be an empty file, not creating feather df")
            # df['Time since apoptosis'] = df['Frame'] - apo_t
            # df['Focal ID'] = [apop_ID] * len(df)
            # new_fn = file.replace('.csv','.feather')
            # df.reset_index()
            # df.to_feather(new_fn)
            continue
        ### tidy up dataframe
        df['Cell ID'] = df['Cell ID'].str.replace('[()]', '')
        df['y'] = df['y'].str.replace('[()]', '')
        df = df.astype(int)
        ### normalise time
        df['Time since apoptosis'] = df['Frame'] - apo_t
        ### add apop ID to dataframe
        df['Focal ID'] = [apop_ID] * len(df)
        ### rename columns to fit with kfunct convention
        #df.rename(columns={'Distance from apoptosis':'dij', 'Time since apoptosis':'tij'})
        ### rename and save out as feather
        new_fn = (file.replace('xyt/1600.1600', 'xyt/feather')).replace('.csv','.feather')
        df.to_feather(new_fn)

def feather_load_radial_df(single_scan_file_list,
                            maximum_R = None,
                            crop_amount = 10,
                            fixed_apop_location = False,
                            streamline = True,
                            strict_filtering = False,
                            frame_filtering = False,
                            remove_duplicates = False):

    """
    A QUICKER function to load each individual radial analysis scan and concatenate them into a cumulative pandas dataframe.
    Needs csv to be saved out as binary feather files first

    single_scan_file_list : list of paths
        A list of .csv/feather files of the individual single-focal-cell radial scans to concatenate together

    maximum_R : int
        Defines the maximum radius of a focal scan in pixels, if this radius leaves the field of view
        then that focal apoptosis is either a) excluded from the cumulative scan (if strict_filtering == True)
        or b) The time points at which is leaves the FOV are excluded (if frame_filtering == True).
        Currently only really makes a difference for post apoptotic times as cell moves around
        before apoptosis so could leave the FOV prior to the xy used as a centoid here

    crop_amount : int
        Defines the cropping amount necessary to exclude boundary effect cells/divisions from
        the radial scan

    streamlined : bool
        If True this returns just the distance from apoptosis and time since apoptosis variables
        in the data frame. If False then it returns data frame as in the .csv file.

    fixed_apop_location : bool
        If True this changes the distance variable from one that follows the apoptotic cell pre-apop
        to one that is the distance between the event and the fixed apoptotic location

    strict_filtering : bool
        This, if True, excludes any focal cells that leave the FOV at any time point by removing those frames from each df

    frame_fitlering : bool
        if true deletes any frames in the main df that mean a scan of radius r is outside the fov

    remove_duplicates : bool or dict
        sorry this isnt the best implementation but default this is false, if not
        then you need to provide the R_max, t_range and bins you want the dataframe
        to be seperated into in order to remove duplicate counts of cells per each
        temporal bin
    """

    radial_scan_df = []

    ### if R is not provided then do not filter any focal apoptoses
    if not maximum_R:
        maximum_R = 0
    ### use R to exclude any focal apoptoses that are out of the FOV at the time of apoptosis
    x_range = range(maximum_R+crop_amount, 1200 - (maximum_R+crop_amount))
    y_range = range(maximum_R+crop_amount, 1600 - (maximum_R+crop_amount))
    ### sometimes only provide one df to this function so list will just be str
    if type(single_scan_file_list) == str:
    ### if this is the case make a list of that one fn item
        single_scan_file_list = [single_scan_file_list]

    for file in tqdm(single_scan_file_list):
        file = file.replace('scr_apop_wt_div_xyt/1600.1600/', 'scr_apop_wt_div_xyt/feather/')
        ### get apop id to load apop xyt file
        apop_ID = file.split('/')[-1].split('_N_')[0]
        if maximum_R > 0:
            ### reformat apop ID to fit in previous convention
            if 'Scr' in apop_ID:
                apop_ID = apop_ID.replace('Scr_-','')+'_RFP'
                apo_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/apoptotic_tracks',
                                         f'{apop_ID}.csv')
            if 'wt' in apop_ID:
                apop_ID = apop_ID.replace('wt_','')+'_GFP'
                if 'wt_control_wt_div' in file:
                    apo_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/control_event_tracks',
                                         f'{apop_ID}.csv')
                if 'wt_apop_wt_div' in file:
                    apo_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/apoptotic_tracks',
                                             f'{apop_ID}.csv')
            apo_df = pd.read_csv(apo_xyt_fn).astype(int)
            del apo_df['Unnamed: 0']
            ### see if any of the frames of the focal cell leave the FOV
            frames_outside_fov = []
            for apo_x, apo_y, apo_t in zip(apo_df['x'], apo_df['y'], apo_df['t']):
                if int(apo_x) not in x_range or int(apo_y) not in  y_range:
                    frames_outside_fov.append(apo_t)
            #print('Frames outside the FOV for ', apop_ID, frames_outside_fov)
            ### if filtering is strict then exclude any focal cell that leves fov
            if strict_filtering == True:
                if len(frames_outside_fov) > 0:
                    continue
        ### if filtering is present but not strict then just exclude time points outside fov
        apo_t, apo_x, apo_y = [int(re.search(r'txy_(\d+)_(\d+)_(\d+)', file)[i]) for i in range(1,4)]
        ### load dataframe
        try:
            df = pd.read_feather(file)
        except:
            print('Not a feather file so loading slower .csv instead')
            df = pd.read_csv(file, names = ['Cell ID', 'Distance from apoptosis', 'Frame', 'x', 'y'])
            ### tidy up dataframe
            df['Cell ID'] = df['Cell ID'].str.replace('[()]', '')
            df['y'] = df['y'].str.replace('[()]', '')
            df = df.astype(int)
            ### normalise time
            df['Time since apoptosis'] = df['Frame'] - apo_t
            ### add apop ID to dataframe
            df['Focal ID'] = [apop_ID] * len(df)
        ### eliminate boundary counts spatially
        df = df.loc[(df['x'] >= crop_amount) & (df['x'] <= 1200-crop_amount) & (df['y'] >= crop_amount) & (df['y'] <= 1600-crop_amount)]
        ### eliminate boundary effects temporally (ie. if scan exits fov at any time point) by deleting those frames from the scan
        if frame_filtering and maximum_R > 0:
            df = df[~df['Frame'].isin(frames_outside_fov)]

        ### other way of cutting down on size of df
        df = df.loc[(df['Distance from apoptosis'] <= 1600) & (df['Time since apoptosis'] <= 800) & (df['Time since apoptosis'] >= -800)]

        ## convert to SI
        df['Time since apoptosis'] = df['Time since apoptosis']*(4/60)
        df['Distance from apoptosis'] = df['Distance from apoptosis']/3
         ### remove unnecessary data=

        if fixed_apop_location == True:
            for i, row in df.iterrows():#, total = len(df), desc = 'Switching distance (dij) from following apop. cell to fixed apop. location'):
                ### only do preapoptotic times as scan if fixed location post apop
                if row['Time since apoptosis'] < 0:
                    focal_ID = row['Focal ID']
                    focal_x, focal_y, focal_t = focal_xyt_finder(focal_ID, single_scan_file_list)
                    df.at[i, 'Distance from apoptosis'] = basic_euc_dist(row['x'], row['y'], focal_x, focal_y) / 3 ### scaled from pixels to micrometers

        if not remove_duplicates == False:
            R_max = remove_duplicates['R_max']
            t_range = remove_duplicates['t_range']
            bins = remove_duplicates['bins']
            df = eliminate_duplicates(df, R_max, t_range, bins)

        if streamline:
            del df['x'], df['y'], df['Cell ID'], df['Frame']
            df = df.round(decimals = 2)
            df = df.astype({'Distance from apoptosis':'int'})

        ### append to larger df
        radial_scan_df.append(df)

    radial_scan_df = pd.concat(radial_scan_df, axis = 0, ignore_index = True)
    N_focal_cells = len(set(radial_scan_df['Focal ID']))
    print('Number of focal cells included in cumulative scan:', N_focal_cells)


    return radial_scan_df

def load_radial_df(single_scan_file_list, maximum_R = None, crop_amount = 20, fixed_apop_location = False, streamline = True, strict_filtering = False, weights = False):

    """
    A function to load each individual radial analysis scan and concatenate them into a cumulative pandas dataframe.

    single_scan_file_list : list of paths
        A list of .csv files of the individual single-focal-cell radial scans to concatenate together

    maximum_R : int
        Defines the maximum radius of a focal scan in pixels, if this radius leaves the field of view
        then that focal apoptosis is either a) excluded from the cumulative scan (if strict_filtering == True)
        or b) The time points at which is leaves the FOV are excluded (if R is given but strict_filtering == False).
        Currently only really makes a difference for post apoptotic times as cell moves around
        before apoptosis so could leave the FOV prior to the xy used as a centoid here

    crop_amount : int
        Defines the cropping amount necessary to exclude boundary effect cells/divisions from
        the radial scan

    streamlined : bool
        If True this returns just the distance from apoptosis and time since apoptosis variables
        in the data frame. If False then it returns data frame as in the .csv file.

    fixed_apop_location : bool
        If True this changes the distance variable from one that follows the apoptotic cell pre-apop
        to one that is the distance between the event and the fixed apoptotic location

    strict_filtering : bool
        This, if True, excludes any focal cells that leave the FOV at any time point by removing those frames from each df

    weights : bool
        This, if True, skips over the previous strict_filtering and leaves all frames in to be weighted


    """

    radial_scan_df = []
    N_focal_cells = 0

    ### if R is not provided then do not filter any focal apoptoses
    if not maximum_R:
        maximum_R = 0
    ### use R to exclude any focal apoptoses that are out of the FOV at the time of apoptosis
    x_range = range(maximum_R+crop_amount, 1200 - (maximum_R+crop_amount))
    y_range = range(maximum_R+crop_amount, 1600 - (maximum_R+crop_amount))
    ### sometimes only provide one df to this function so list will just be str
    if type(single_scan_file_list) == str:
    ### if this is the case make a list of that one fn item
        single_scan_file_list = [single_scan_file_list]

    for file in tqdm(single_scan_file_list):
        ### get apop id to load apop xyt file
        apop_ID = file.split('/')[-1].split('_N_')[0]
        if maximum_R > 0:
            ### reformat apop ID to fit in previous convention ### load full apo xyt
            if 'Scr' in apop_ID:
                apop_ID = apop_ID.replace('Scr_-','')+'_RFP'
                apo_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/apoptotic_tracks',
                                         f'{apop_ID}.csv')
            if 'wt' in apop_ID:
                apop_ID = apop_ID.replace('wt_','')+'_GFP'
                if 'wt_control_wt_div' in file:
                    apo_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/control_event_tracks',
                                         f'{apop_ID}.csv')
                if 'wt_apop_wt_div' in file:
                    apo_xyt_fn = os.path.join('/home/nathan/data/kraken/scr/h2b/giulia/experiment_information/apoptoses/apoptotic_tracks',
                                             f'{apop_ID}.csv')

            apo_df = pd.read_csv(apo_xyt_fn).astype(int)
            del apo_df['Unnamed: 0']
            ### see if any of the frames of the focal cell leave the FOV
            frames_outside_fov = []
            for apo_x, apo_y, apo_t in zip(apo_df['x'], apo_df['y'], apo_df['t']):
                if int(apo_x) not in x_range or int(apo_y) not in  y_range:
                    frames_outside_fov.append(apo_t)
            #print('Frames outside the FOV for ', apop_ID, frames_outside_fov)
            ### if filtering is strict then exclude any focal cell that leves fov
            if strict_filtering == True:
                if len(frames_outside_fov) > 0:
                    continue
        ### if filtering is present but not strict then just exclude time points outside fov
        apo_t, apo_x, apo_y = [int(re.search(r'txy_(\d+)_(\d+)_(\d+)', file)[i]) for i in range(1,4)]
        ### load dataframe
        df = pd.read_csv(file, names = ['Cell ID', 'Distance from apoptosis', 'Frame', 'x', 'y'])
        ### tidy up dataframe
        df['Cell ID'] = df['Cell ID'].str.replace('[()]', '')
        df['y'] = df['y'].str.replace('[()]', '')
        df = df.astype(int)
        ### normalise time
        df['Time since apoptosis'] = df['Frame'] - apo_t
        ### eliminate boundary counts spatially
        df = df.loc[(df['x'] >= crop_amount) & (df['x'] <= 1200-crop_amount) & (df['y'] >= crop_amount) & (df['y'] <= 1600-crop_amount)]
        ### eliminate boundary effects temporally (ie. if scan exits fov at any time point) by deleting those frames from the scan
        if maximum_R > 0 and weights == False:
            df = df[~df['Frame'].isin(frames_outside_fov)]
        ### remove unnecessary data
        if streamline:
            del df['x'], df['y'], df['Cell ID'], df['Frame']
        ### add apop ID to dataframe
        df['Focal ID'] = [apop_ID] * len(df)
        ### append to larger df
        radial_scan_df.append(df)
        N_focal_cells +=1
    radial_scan_df = pd.concat(radial_scan_df, axis = 0, ignore_index = True)
    radial_scan_df['Time since apoptosis'] = radial_scan_df['Time since apoptosis']*(4/60)
    radial_scan_df['Distance from apoptosis'] = radial_scan_df['Distance from apoptosis']/3
    print('Number of focal cells included in cumulative scan:', N_focal_cells)

    if fixed_apop_location == True:
        for i, row in tqdm(radial_scan_df.iterrows(), total = len(radial_scan_df), desc = 'Switching distance (dij) from following apop. cell to fixed apop. location'):
            ### only do preapoptotic times as scan if fixed location post apop
            if row['Time since apoptosis'] < 0:
                focal_ID = row['Focal ID']
                focal_x, focal_y, focal_t = focal_xyt_finder(focal_ID, file_list)
                radial_scan_df.at[i, 'Distance from apoptosis'] = basic_euc_dist(row['x'], row['y'], focal_x, focal_y) / 3 ### scaled from pixels to micrometers

    if weights:
        print(frames_outside_fov)

    return radial_scan_df

def load_tracking_data(tracks_path):
    """
    Tracks loader takes a path and returns the ordered tracking data for two populations of cells (i.e. wild-type and scr-kd)
    The wild-type cells will have positive integer IDs and the mutant population will have negative integer IDs
    """
    import btrack

    # from btrack.utils import import_HDF, import_JSON

    print("Btrack version no.:", btrack.__version__)

    with btrack.dataio.HDF5FileHandler(tracks_path, "r", obj_type="obj_type_1") as h:
        wt_cells = h.tracks
    with btrack.dataio.HDF5FileHandler(tracks_path, "r", obj_type="obj_type_2") as h:
        scr_cells = h.tracks
    for i in range(len(scr_cells)):
        scr_cells[i].ID = -(scr_cells[i].ID)
    all_cells = wt_cells + scr_cells  ### negative IDs are scribble cells
    print(
        "Track information loaded and ordered according to cell type (WT IDs >0> Scr IDs)"
    )
    return wt_cells, scr_cells, all_cells


def apoptosis_list_loader(path_to_apop_lists, focal_cell):
    """
    load a list of apoptoses from typical list of .txt files detailing apoptosis times
    """
    if focal_cell == "Scr":
        cell_type = "RFP"
    if focal_cell == "WT":
        cell_type = "GFP "
    expts_apop_lists = os.listdir(path_to_apop_lists)
    apop_dict = {}
    N_apops = len(expts_apop_lists)
    for expt_apop_list in expts_apop_lists:
        apop_list = open(os.path.join(path_to_apop_lists, expt_apop_list))
        for apop_ID in apop_list:
            if cell_type in apop_ID:
                if (
                    "stitched" not in apop_ID
                ):  ## relic of apoptosis finding (stitched = tracks that apoptosis switches into post apop)
                    apop_dict[apop_ID.split()[0]] = apop_ID.split()[1]
    return apop_dict

def event_dict_loader(path_to_dict):
    import json
    with open(path_to_dict) as f:
        event_dict = json.load(f)
    return event_dict

def hdf5_file_finder(hdf5_parent_folder):
    """
    load my list of hdf5 tracking files from a typical directory tree with different experiments
    """
    aligned_hdf5_file_list = glob.glob(
        os.path.join(hdf5_parent_folder, "GV****/Pos*/*aligned/HDF/segmented.hdf5")
    )
    unaligned_hdf5_file_list = glob.glob(
        os.path.join(hdf5_parent_folder, "GV****/Pos*/HDF/segmented.hdf5")
    )
    ### new format of h5 tracking file
    h5_file_list = glob.glob(
        os.path.join(hdf5_parent_folder, "******/Pos*/tracks.h5")
    )
    hdf5_file_list = aligned_hdf5_file_list + unaligned_hdf5_file_list + h5_file_list
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
        expt = "GV" + str(re.findall(r"GV(\d+)", apop_ID)[0])
        position = re.findall(r"Pos(\d+)", apop_ID)[0]
        position = "Pos" + position
        expt_position = os.path.join(
            expt, position, ""
        )  ## additional '' here so that / added to end of string
        cell_ID = int((re.findall(r"(\d+)_.FP", apop_ID))[0])
        print("ID", apop_ID)

        if expt_position not in hdf5_file_path:
            ## load that track data
            print("Loading", expt_position)
            hdf5_file_path = [
                hdf5_file_path
                for hdf5_file_path in tracking_filelist
                if expt_position in hdf5_file_path
            ][0]
            wt_cells, scr_cells, all_cells = load_tracking_data(hdf5_file_path)
            print("Loaded", expt_position)
            expt_count += 1

        if "RFP" in apop_ID:
            cell_ID = -cell_ID

        focal_time = int(apop_dict[apop_ID])
        try:
            target_cell = [cell for cell in all_cells if cell.ID == cell_ID][0]
        except:
            error_message = apop_ID + " could not find cell_ID"
            error_log.append(error_message)
            continue
        if target_cell.in_frame(focal_time):

            target_cell.t.index(focal_time)
            x, y = (
                target_cell.x[target_cell.t.index(focal_time)],
                target_cell.y[target_cell.t.index(focal_time)],
            )
            list_xy.append([x, y])
            cell_count += 1
        else:
            print("Focal time not in frame!!!!!!!!!!!")
            error_message = (
                apop_ID
                + " apoptosis time t="
                + str(focal_time)
                + " not in cell range "
                + str(range(target_cell.t[0], target_cell.t[-1]))
            )
            error_log.append(error_message)
    return list_xy, cell_count, expt_count, error_log


def load_and_align_tracks(tracks_path, image_stack):
    """
    This takes the path to a tracking file and loads the tracks, realigning them to be displayed over image_stack, which must also be provided so that the shape can be assessed
    """

    global wt_tracks, scr_tracks, shift_y, shift_x

    with btrack.dataio.HDF5FileHandler(tracks_path, "r", obj_type="obj_type_1") as hdf:
        wt_tracks = hdf.tracks
    with btrack.dataio.HDF5FileHandler(tracks_path, "r", obj_type="obj_type_2") as hdf:
        scr_tracks = hdf.tracks

    ## this method casues problems as the viewer for tracks doesnt like negative numbers
    # wt_tracks, scr_tracks, all_tracks = tools.load_tracking_data(tracks_path)

    print("Tracks loaded")

    ### finding coord range of aligned images, coords switched already ## need to sort out the order of try excepts
    try:
        align_x_range, align_y_range = image_stack.shape[2], image_stack.shape[1]
    except:
        print("Error: no image data loaded to map tracks to")
        return

    ### finding maximum extent of tracking coords
    tracks_x_range = round(max([max(track.x) for track in wt_tracks]))
    tracks_y_range = (
        round(max([max(track.y) for track in wt_tracks])) + 2
    )  ## sort this hack out later

    ### coord switch
    tmp = tracks_y_range
    tracks_y_range = tracks_x_range
    tracks_x_range = tmp

    print("tracks range:", (tracks_x_range), (tracks_y_range))
    print("aligned image range:", (align_x_range), (align_y_range))

    shift_x = int((align_x_range - tracks_x_range) / 2)
    shift_y = int((align_y_range - tracks_y_range) / 2)

    print("shift in x and y:", shift_x, shift_y)

    global wt_data, scr_data, properties, graph

    wt_data, properties, graph = tracks_to_napari(wt_tracks, ndim=2)
    scr_data, properties, graph = tracks_to_napari(scr_tracks, ndim=2)

    tmp = wt_data[:, 2].copy()  ## copy the true_y coord
    wt_data[:, 2] = wt_data[:, 3]  ##assign the old_y coord as the true_x
    wt_data[:, 3] = tmp  ## assign the old_x as true_y

    wt_data[:, 2] += shift_y  ## TRUE_Y (vertical axis)
    wt_data[:, 3] += shift_x  ## TRUE_X (horizontal axis)

    tmp = scr_data[:, 2].copy()
    scr_data[:, 2] = scr_data[:, 3]
    scr_data[:, 3] = tmp

    scr_data[:, 2] += shift_y  ## TRUE_Y (vertical axis)
    scr_data[:, 3] += shift_x  ## TRUE_X (horizontal axis)

    print("coordinate shift applied")

    return wt_tracks, scr_tracks, wt_data, scr_data, shift_x, shift_y
