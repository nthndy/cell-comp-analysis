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

def select_target_cell(cell_ID, all_cells, cell_type = 'Scr'):
    """
    Select target cell takes a cell ID and variable containing a population of loaded cells from previous function and returns the corresponding mutant cell by default, or the wild-type if specified.
    """
    if cell_type == 'Scr':
        if int(cell_ID) > 0:
            cell_ID = -int(cell_ID)
        target_cell = [track for track in all_cells if track.ID == cell_ID][0]
    elif cell_type == 'WT':
        target_cell = [track for track in all_cells if track.ID == cell_ID][0]
    return target_cell

def apoptosis_time(target_cell):
    """
    This function will try and find an apoptosis (defined as three sequential apoptosis classifications from the btrack information) time point.
    """
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
        raise Exception("Could not find apoptosis time, try manual dictionary of apoptosis IDs")
    return focal_time

def random_time(target_cell):
    """
    This will return a random time point in the target cells existence, useful for generating control plots in the latter analysis.
    """
    focal_time = random.choice(target_cell.t)
    return focal_time

def euc_dist(target_cell, other_cell, frame, focal_time):
    """
    An important function that calculates the distance between two cells at a given frame in the timelapse microscope movie. If the cell has gone through an apoptosis, it's location will be returned as the apoptosis location. If the cell does not exist at that frame then a `np.inf` distance will be returned.
    """
    try:
        if frame > target_cell.t[-1]: ## if the frame of the scan is beyond the final frame of the apoptotic cell then use the apoptosis location (ie if the cell has died then fix the scan location at the apoptotic frame location)
            idx0 = target_cell.t.index(focal_time) ## could also do just ## apop_index
        else:
            idx0 = target_cell.t.index(frame)
        idx1 = other_cell.t.index(frame) ## t.index provides the index of that frame
    except:
        return np.inf ## if the other_cell does not exist for frame then returns the euc dist as np.inf

    dx = target_cell.x[idx0] - other_cell.x[idx1]
    dy = target_cell.y[idx0] - other_cell.y[idx1]

    return np.sqrt(dx**2 + dy**2)

def cell_counter(subject_cells, target_cell, radius, t_range, focal_time):
    """
    Takes a population of subject cells and a single target cell and counts the spatial and temporal distance between each subject cell and the target cell over a specified radius and time range centered around a focal time.
    """
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
    event = event.upper()
    ### the issue here is that it assumes mitosis time is the last time frame in a track (accurate) which does not work with apoptosis
    if event == 'APOPTOSIS':
        print("Current apoptosis timepoints are inaccurate so proceed with this analysis with caution")
        events = [tuple(((cell.ID),
                          (round((euc_dist(target_cell, cell, cell.t[-1], focal_time)),2)),
                          ((cell.t[-1]))))
                           for cell in subject_cells
                               if euc_dist(target_cell, cell, cell.t[-1], focal_time)<radius and
                                  cell.t[-1] in range(focal_time-int(t_range/2), focal_time+ int(t_range/2)) and
                                  cell.fate.name == event
                           ]
        return events
    elif event == 'DIVIDE':
        events = [tuple(((cell.ID),
                          (round((euc_dist(target_cell, cell, cell.t[-1], focal_time)),2)),
                          ((cell.t[-1]))))
                           for cell in subject_cells
                               if euc_dist(target_cell, cell, cell.t[-1], focal_time)<radius and
                                  cell.t[-1] in range(focal_time-int(t_range/2), focal_time+ int(t_range/2)) and
                                  cell.fate.name == event
                           ]
        return events
    else:
        return print('Event type not recognised, please try again with either "apoptosis" or "divide"')
