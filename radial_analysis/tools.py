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

import numpy as np
from tqdm import tqdm

def focal_xyt_finder(focal_ID, file_list):
    """
    Simple function to pull xy coords from correctly formatted radial scan fn
    """
    ### need to reformat focal ID to match fn style
    if 'RFP' in focal_ID:
        focal_ID = f'{focal_ID.split("_")[0]}_{focal_ID.split("_")[1]}_Scr_-{focal_ID.split("_")[2]}_N'
    elif 'GFP' in focal_ID:
        focal_ID = f'{focal_ID.split("_")[0]}_{focal_ID.split("_")[1]}_wt_{focal_ID.split("_")[2]}_N'
    else:
        ### this is the scenario where the fn has already been reformatted to have scr or wt in
        focal_ID = focal_ID + '_N_'
        ### this reformat needs to happen to bookend the number so for eg id 15 isnt confused with 155
    focal_fn = [fn for fn in file_list if focal_ID in fn][0]
    t, x, y = re.findall(r'\d+', focal_fn)[-3:]

    return int(x), int(y), int(t)

def select_target_cell(cell_ID, all_cells, cell_type="Scr"):
    """
    Select target cell takes a cell ID and variable containing a population of loaded cells from previous function and returns the corresponding mutant cell by default, or the wild-type if specified.
    """
    if cell_type == "Scr":
        if int(cell_ID) > 0:
            cell_ID = -int(cell_ID)
        target_cell = [track for track in all_cells if track.ID == cell_ID][0]
    elif cell_type == "WT":
        target_cell = [track for track in all_cells if track.ID == cell_ID][0]
    return target_cell


def apoptosis_time(target_cell):
    """
    This function will try and find an apoptosis (defined as three sequential
    apoptosis classifications from the btrack information) time point.
    """
    try:  ## try to find apoptosis time ## replace with chris' definitive definitions of apoptosis?
        if (
            target_cell.label[0] == "APOPTOSIS"
        ):  ## if the first classification is apoptosis then thats a duff track
            print(
                "False apoptosis (first classification is apoptosis) Scr ID:",
                target_cell.ID,
            )
            focal_index = focal_time = False
        else:
            for i, j in enumerate(target_cell.label):
                if (
                    j == "APOPTOSIS"
                    and target_cell.label[i + 1] == "APOPTOSIS"
                    and target_cell.label[i + 2] == "APOPTOSIS"
                ):  # and target_cell.label[i+3] =='APOPTOSIS' and target_cell.label[i+4] =='APOPTOSIS':
                    focal_index = i
                    break
            focal_time = target_cell.t[focal_index]
    except:
        raise Exception(
            "Could not find apoptosis time, try manual dictionary of apoptosis IDs"
        )
    return focal_time


def random_time(target_cell):
    """
    This will return a random time point in the target cells existence, useful
    for generating control plots in the latter analysis.
    """
    focal_time = random.choice(target_cell.t)
    return focal_time

def basic_euc_dist(x1, y1, x2, y2):
    """
    Simplified version of euc_dist that just requires coordinate input
    """
    delta_x = x1 - x2
    delta_y = y1 - y2

    return np.sqrt(delta_x**2+delta_y**2)


def euc_dist(target_cell, other_cell, frame, focal_time):
    """
    An important function that calculates the distance between two cells at a
    given frame in the timelapse microscope movie. If the cell has gone through
    an apoptosis, it's location will be returned as the apoptosis location. If
    the cell does not exist at that frame then a `np.inf` distance will be
    returned.
    """
    try:
        if (
            frame > focal_time
        ):  ## this way ensures analytical continuity for control instance ##  target_cell.t[-1]: ## if the frame of the scan is beyond the final frame of the apoptotic cell then use the apoptosis location (ie if the cell has died then fix the scan location at the apoptotic frame location)
            idx0 = target_cell.t.index(focal_time)  ## could also do just ## apop_index
        else:
            idx0 = target_cell.t.index(frame)
        idx1 = other_cell.t.index(frame)  ## t.index provides the index of that frame
    except:
        return (
            np.inf
        )  ## if the other_cell does not exist for frame then returns the euc dist as np.inf

    dx = target_cell.x[idx0] - other_cell.x[idx1]
    dy = target_cell.y[idx0] - other_cell.y[idx1]

    return np.sqrt(dx ** 2 + dy ** 2)


def cell_counter(subject_cells, target_cell, radius, t_range, focal_time):
    """
    Takes a population of subject cells and a single target cell and counts the
    spatial and temporal distance between each subject cell and the target cell
    over a specified radius and time range centered around a focal time.
    """
    ## subject should equal wt_cells, scr_cells or all_cells
    focal_index = target_cell.t.index(focal_time)
    cells = [
        tuple(
            (
                (cell.ID),
                (
                    round(
                        (
                            euc_dist(
                                target_cell, cell, (focal_time + delta_t), focal_time
                            )
                        ),
                        2,
                    )
                ),
                (focal_time + delta_t),
                int(cell.x[cell.t.index(focal_time + delta_t)]),
                int(cell.y[cell.t.index(focal_time + delta_t)]),
            )
        )
        for delta_t in range(-int(t_range / 2), int(t_range / 2))
        for cell in subject_cells
        if euc_dist(target_cell, cell, focal_time + delta_t, focal_time) < radius
        and cell.ID != target_cell.ID
    ]
    return cells


def event_counter(event, subject_cells, target_cell, radius, t_range, focal_time):
    """
    Similar to the previous function, except this counts cellular events
    specified by an extra positional argument:
    "event"
    Where event can either be a string input of `apoptosis` or `divide`.
    """

    focal_index = target_cell.t.index(focal_time)
    event = event.upper()
    ### the issue here is that it assumes mitosis time is the last time frame in
    ### a track (accurate) which does not work with apoptosis, need to test this
    ### apoptosis_time function fix
    if event == "APOPTOSIS":
        print(
            "Current apoptosis timepoints are inaccurate so proceed with this analysis with caution"
        )
        events = [
            tuple(
                (
                    (cell.ID),
                    (round((euc_dist(target_cell, cell, apoptosis_time(cell), focal_time)), 2)),
                    apoptosis_time(cell),
                    int(cell.x[cell.t.index(apoptosis_time(cell))]),
                    int(cell.y[cell.t.index(apoptosis_time(cell))]),
                )
            )
            for cell in subject_cells
            if euc_dist(target_cell, cell, apoptosis_time(cell), focal_time) < radius
            and apoptosis_time(cell) #
            in range(focal_time - int(t_range / 2), focal_time + int(t_range / 2))
            and cell.fate.name == event
            and cell.ID != target_cell.ID
        ]
        return events
    elif event == "DIVIDE":
        events = [
            tuple(
                (
                    (cell.ID),
                    (round((euc_dist(target_cell, cell, cell.t[-1], focal_time)), 2)),
                    (cell.t[-1]),
                    int(cell.x[-1]),
                    int(cell.y[-1]),
                )
            )  ### this point of division is reliably the last frame of the dividing cell
            for cell in subject_cells
            if euc_dist(target_cell, cell, cell.t[-1], focal_time) < radius
            and cell.t[-1]
            in range(focal_time - int(t_range / 2), focal_time + int(t_range / 2))
            and cell.fate.name == event
            and cell.ID != target_cell.ID
        ]
        return events
    else:
        return print(
            'Event type not recognised, please try again with either "apoptosis" or "divide"'
        )

def quality_counter():
    """
    WIP, want to measure other qualities as well as divisions or apoptoses
    """
