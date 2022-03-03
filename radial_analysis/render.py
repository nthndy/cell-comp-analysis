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

import csv
import glob
import os
import re
import time
from datetime import datetime
from pathlib import Path

import btrack
import calculate_radial_analysis as calculate
import matplotlib.font_manager
import matplotlib.pyplot as plt
import napari
import numpy as np
import tools
#from btrack.utils import import_HDF, import_JSON, tracks_to_napari
from natsort import natsorted
from skimage.io import imread
from tqdm.notebook import tnrange, tqdm, tqdm_notebook

"""
Graph rendering below

This section takes the final output of my radial analysis
and renders the relevant graphs and labels

Two main components to this file, The first is a newer
class-based method of plotting that I am testing. The second is the old function based way.
"""

class Heatmap:
    def __init__(self, array, radius, t_range,
                 **scan_details):
        # the actual array to be plotted
        self.array = array
        # various components of the array that are necessary for plotting
        self.radius = radius
        self.t_range = t_range
        self.num_bins = array.shape[0]
        # optional dict input of scan details (like old plot method)
        self.scan_details = scan_details

    def xy_labels(self, label_freq: int = 1, SI:bool = True):
        radial_bin = self.radius / self.num_bins
        temporal_bin = self.t_range / self.num_bins

        if SI == True:
            time_scale_factor = 4 / 60  ## each frame is 4 minutes
            distance_scale_factor = 1 / 3  ## each pixel is 0.3recur micrometers
        else:
            time_scale_factor, distance_scale_factor = 1, 1

        ### generate labels for axis micrometers/hours
        xlocs = np.arange(
            -0.5, self.num_bins, label_freq
        )  ## -0.5 to start at far left border of first bin
        xlabels = []
        for m in np.arange(int(-self.num_bins / 2), int(self.num_bins / 2) + 1, label_freq):
            xlabels.append(
                str(int(((temporal_bin) * m) * time_scale_factor))
            )  # + "," + str(int(((temporal_bin)*m+temporal_bin)*time_scale_factor)))

        ylocs = np.arange(
            -0.5, self.num_bins, label_freq
        )  ## -0.5 to start at far top border of first bin
        ylabels = []
        for m in np.arange(self.num_bins, 0 - 1, -label_freq):
            ylabels.append(
                str(int(((radial_bin) * m) * distance_scale_factor))
            )  # + "," + str(int(((radial_bin)*(m-1)*distance_scale_factor))))

        self.xlocs = xlocs
        self.xlabels = xlabels
        self.ylocs = ylocs
        self.ylabels = ylabels

        return xlocs, xlabels, ylocs, ylabels

    def plot_titles(self, SI: bool = True):
        # if all the details arent present in input details then manually enter
        if {'focal_cell', 'focal_event', 'input_type', 'N', 'subject_cell', 'subject_event'} != self.scan_details.keys():
            title = input('Input the title of your plot')
            cb_label = input('Input the label for your colorbar')
            focal_event_name = input('What is the focal event of the radial scan?')
            x_axis_label = f"Time since {focal_event_name} "
            y_axis_label = f"Distance from {focal_event_name} "
        #else get focal cell etc details from dict input
        else:
            focal_cell_name = self.scan_details.get('focal_cell')#, input('What is the focal cell of the radial scan?'))
            focal_event_name = self.scan_details.get('focal_event')#, input('What is the focal event of the radial scan?'))
            subject_cell_name = self.scan_details.get('subject_cell')#, input('What is the subject cell of the radial scan?'))
            subject_event_name = self.scan_details.get('subject_event')#, input('What is the subject event of the radial scan?'))
            N = int(self.scan_details.get('N'))#, input('Enter the number of focal events')))
            # now check which input type is specified
            if self.scan_details.get('input_type') == "N_cells":
                title = f"Spatiotemporal dist. of {subject_cell_name} cells \n around {focal_cell_name} {focal_event_name} (N={N})"
                cb_label = f"Number of {subject_cell_name} cell apperances"

            elif self.scan_details.get('input_type') == "N_events":
                title = f"Spatiotemporal dist. of {subject_cell_name} {subject_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
                cb_label = f"Number of {subject_cell_name} {subject_event_name}"

            elif self.scan_details.get('input_type') == "P_events":
                title = f"Spatiotemporal dist. of probability of {subject_cell_name} {subject_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
                cb_label = f"Probability of {subject_cell_name} {subject_event_name}"

            elif self.scan_details.get('input_type') == "CV":
                title = f"Coefficient of variation of probability of {subject_cell_name} {subject_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
                cb_label = "Coefficient of variation"

            elif self.scan_details.get('input_type') == "stat_rel":
                title = f"Statisticall relevant areas of probability of {subject_cell_name} {subject_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
                cb_label = "Relevant areas are set equal to 1"

            elif self.scan_details.get('input_type') == "dP":
                title = f"Difference in probability between \ncanonical and control analysis \ni.e. probability of {subject_event_name} above background"
                cb_label = "Difference in probability\n above background"
            else:
                print('input_type not recognised')
        if SI == True:
            time_unit = "(Hours)"
            distance_unit = "(Micrometers)"
        else:
            time_unit = "(Frames)"
            distance_unit = "(Pixels)"

        x_axis_label = f"Time since {focal_event_name} {time_unit}"
        y_axis_label = f"Distance from {focal_event_name} {distance_unit}"

        ## is this correct usage here? dont think so but im in too deep to stop now!!!
        self.title = title
        self.cb_label = cb_label
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.distance_unit = distance_unit
        self.time_unit = time_unit
        self.focal_event_name = focal_event_name

        return title, cb_label, x_axis_label, y_axis_label

    def render_plot(self, fontname: str = "Liberation Mono", include_apop_bin: bool = True, cbar_lim = False, bin_labels:bool = False, output_path: str = None):
        #import matplotlib
        import matplotlib.font_manager
        import matplotlib.pyplot as plt

        ## call the labels and title functions
        self.xy_labels()
        #xy_labels
        self.plot_titles()

        #begin plotting
        plt.clf()
        font = {"fontname": fontname}
        plt.xticks(self.xlocs, self.xlabels, rotation="vertical", **font)
        plt.yticks(self.ylocs, self.ylabels, **font)
        plt.xlabel(self.x_axis_label, **font)
        plt.ylabel(self.y_axis_label, **font)
        plt.title(self.title + "\n", fontweight="bold", **font)

        # crop heatmap if defined so in redner plot funct options
        final_hist = np.flipud(self.array) if include_apop_bin == True else np.flipud(self.array[1:-1, :])

        ## apop location marker
        if self.num_bins == 10:
            if include_apop_bin == True:
                plt.scatter(
                self.num_bins / 2 - 0.5, self.num_bins - 0.75, s=20, c="white", marker="v"
            )
                plt.text(
                    self.num_bins + 0.15,
                    self.num_bins + 1.5,
                    f"{self.focal_event_name} location \nshown by inverted \nwhite triangle",
                    **font,
                )
            else:
                plt.scatter(
                    self.num_bins / 2 - 0.5, self.num_bins - 2 - 0.75, s=20, c="white", marker="v"
                )
                plt.text(
                    self.num_bins + 0.15,
                    self.num_bins + 1.5 - 2,
                    f"{self.focal_event_name} location \nshown by inverted \nwhite triangle",
                    **font,
                )
        if self.num_bins == 20:
            if include_apop_bin == True:
                plt.scatter(self.num_bins / 2 - 0.5, self.num_bins - 0.9, s=20, c="white", marker="v")
                plt.text(
                    self.num_bins + 0.3,
                    self.num_bins + 3.5,
                    f"{self.focal_event_name} location \nshown by inverted \nwhite triangle",
                    **font,
                )
            else:
                plt.scatter(self.num_bins / 2 - 0.5, self.num_bins - 1.8, s=20, c="white", marker="v")
                plt.text(
                    self.num_bins + 0.3,
                    self.num_bins + 2.5,
                    f"{self.focal_event_name} location \nshown by inverted \nwhite triangle",
                    **font,
                )

        ## colorbar
        if cbar_lim == False:
            if include_apop_bin == False:
                plt.clim(
                    vmin=np.min(self.array[1:, :]), vmax=np.max(self.array[1:, :])
                )
            else:
                plt.clim(vmin=np.min(self.array), vmax=np.max(self.array))
            cb = plt.colorbar(
                label=self.cb_label
            )  ### matplotlib.cm.ScalarMappable(norm = ???cmap='PiYG'), use this in conjunction with norm to set cbar equal to diff piyg coloourscheme
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(family=fontname)
            text.set_font_properties(font)
            ax.set_yticklabels(
                np.round(ax.get_yticks(), 5), **{"fontname":fontname}
            )  ### cropped to 5dp
        else:
            # manual input of cbar lim as tuple
            plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
            cb = plt.colorbar(label=self.cb_label)
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(family=fontname)
            text.set_font_properties(font)
            ax.set_yticklabels(
                np.round(ax.get_yticks(), 5), **{"fontname": fontname}
            )

        ## bin labels
        if bin_labels == True:
            flipped = np.flipud(self.array)
            if self.scan_details.get('input_type') == "P_events":
                for i in range(len(input_2d_hist)):
                    for j in range(len(input_2d_hist)):
                        text = plt.text(
                            j,
                            i,
                            round(flipped[i, j], 5),
                            ha="center",
                            va="center",
                            color="w",
                            fontsize="xx-small",
                        )
            elif self.scan_details.get('input_type') == "dP":
                for i in range(len(input_2d_hist)):
                    for j in range(len(input_2d_hist)):
                        text = plt.text(
                            j,
                            i,
                            round(flipped[i, j], 6),
                            ha="center",
                            va="center",
                            color="w",
                            fontsize="xx-small",
                        )
            elif self.scan_details.get('input_type') == "CV":
                for i in range(len(input_2d_hist)):
                    for j in range(len(input_2d_hist)):
                        text = plt.text(
                            j,
                            i,
                            round(flipped[i, j], 3),
                            ha="center",
                            va="center",
                            color="w",
                            fontsize="xx-small",
                        )
            if self.scan_details.get('input_type') == "stat_rel":
                for i in range(len(input_2d_hist)):
                    for j in range(len(input_2d_hist)):
                        text = plt.text(
                            j,
                            i,
                            int(flipped[i, j]),
                            ha="center",
                            va="center",
                            color="w",
                            fontsize="xx-small",
                        )
            else:
                for i in range(len(input_2d_hist)):
                    for j in range(len(input_2d_hist)):
                        text = plt.text(
                            j,
                            i,
                            int(flipped[i, j]),
                            ha="center",
                            va="center",
                            color="w",
                            fontsize="xx-small",
                        )
        ## save out?
        if not output_path:
            plt.imshow(final_hist)
            return   # ,cmap = 'PiYG')
        else:
            ## output save path formatting
            save_parent_dir = output_path
            if {'focal_cell', 'focal_event', 'input_type', 'N', 'subject_cell', 'subject_event'} != self.scan_details.keys():
                print('No automatic sub directory made for plot as no scan details found')
                save_path = save_parent_dir
            else:
                save_dir_name = "{}_{}_{}_{}".format(
                    self.scan_details.get('focal_cell').lower(),
                    self.scan_details.get('focal_event').lower(),
                    self.scan_details.get('subject_cell').lower(),
                    self.scan_details.get('subject_event').lower()
                )
                save_path = os.path.join(save_parent_dir, save_dir_name)

            # create output dir
            Path(save_path).mkdir(parents=True, exist_ok=True)

            ## filename
            fn = save_path + "/" + self.title + f" {self.radius}.{self.t_range}.{self.num_bins}.pdf"
            ## failsafe overwriting block
            if os.path.exists(fn):
                print("Filename", fn, "already exists, saving as updated copy")
                fn = fn.replace(
                    ".pdf", " (updated {}).pdf".format(time.strftime("%Y%m%d-%H%M%S"))
                )
            plt.imshow(final_hist)
            plt.plot()
            plt.savefig(fn, dpi=300, bbox_inches="tight")
            print("Plot saved at ", fn)

            return plt.imshow(final_hist)

def cumulative_kymo_compiler(raw_files_dir, radius, t_range, num_bins):
    """
    This function compiles kymograph from raw list files directory
    """
    ### find individual radial scan raw files
    N_cells_fns = natsorted(glob.glob(os.path.join(raw_files_dir, "*cells*")))
    N_events_fns = natsorted(glob.glob(os.path.join(raw_files_dir, "*events*")))
    N_cells_cropped_hist_cumulative, N_events_cropped_hist_cumulative = (
        np.zeros((num_bins, num_bins)),
        np.zeros((num_bins, num_bins)),
    )
    ### ascertain the maximum scan parameters of chosen files
    radius_max = int(re.findall(r"rad_(\d+)", N_cells_fns[0])[0])
    t_range_max = int(re.findall(r"t_range_(\d+)", N_cells_fns[0])[0])

    ### iterate through all files compiling into cumulative
    N = len(N_cells_fns)
    # ### set iteration updates from tqdm
    # progress_bar = tqdm(total = N)
    for i, N_cells_fn in tqdm(enumerate(N_cells_fns), total=N):  # tqdm(N_cells_fns):
        ## find events fn
        # N_events_fn = N_cells_fn.replace('cells_wt','events_wtdiv')
        N_events_fn = N_events_fns[
            i
        ]  ## more sustainable method for when subj event is not division
        ### checking if fns are for same scan
        if N_cells_fn.split("_N_")[0] != N_events_fn.split("_N_")[0]:
            print(
                "Filename error, could not guarantee that N_cells and N_events are from the same scan. Filenames:",
                N_cells_fn,
                N_events_fn,
            )
        elif int(re.findall(r"focal_t_(\d+)", N_cells_fn)[0]) != int(
            re.findall(r"focal_t_(\d+)", N_events_fn)[0]
        ):
            print(
                "Filename error, could not guarantee that N_cells and N_events have same focal time. Filenames:",
                N_cells_fn,
                N_events_fn,
            )

        ## find focal time
        focal_time = int(re.findall(r"focal_t_(\d+)", N_cells_fn)[0])

        ## load data
        # N_cells, N_events = [], []
        (
            N_cells_distance_cropped,
            N_cells_time_cropped,
            N_events_distance_cropped,
            N_events_time_cropped,
        ) = ([], [], [], [])
        with open(N_cells_fn, newline="") as csvfile:
            N_cells_import_csv = csv.reader(csvfile, delimiter="\n")  # , quotechar='|')
            for row in N_cells_import_csv:
                if eval(row[0])[1] < radius and (focal_time - (t_range / 2)) <= eval(
                    row[0]
                )[2] < (
                    focal_time + (t_range / 2)
                ):  ### filter here to speed up execution
                    # N_cells.append(eval(row[0]))
                    N_cells_distance_cropped.append(eval(row[0])[1])
                    N_cells_time_cropped.append(eval(row[0])[2])
        with open(N_events_fn, newline="") as csvfile:
            N_events_import_csv = csv.reader(
                csvfile, delimiter="\n"
            )  # , quotechar='|')
            for row in N_events_import_csv:
                if eval(row[0])[1] < radius and (focal_time - (t_range / 2)) <= eval(
                    row[0]
                )[2] < (focal_time + (t_range / 2)):
                    # N_events.append(eval(row[0]))
                    N_events_distance_cropped.append(eval(row[0])[1])
                    N_events_time_cropped.append(eval(row[0])[2])

        ### make cropped list of distance and time
        # N_cells_cropped = [i for i in N_cells if i[1] < radius and (focal_time - (t_range/2)) <= i[2] < (focal_time + (t_range/2))]
        # N_cells_distance_cropped = [N_cells_cropped[i][1] for i in range(0,len(N_cells_cropped))]
        # N_cells_time_cropped = [N_cells_cropped[i][2] for i in range(0,len(N_cells_cropped))]
        # N_events_cropped = [i for i in N_events if i[1] < radius and (focal_time - (t_range/2)) <= i[2] < (focal_time + (t_range/2))]
        # N_events_distance_cropped = [N_events_cropped[i][1] for i in range(0,len(N_events_cropped))]
        # N_events_time_cropped = [N_events_cropped[i][2] for i in range(0,len(N_events_cropped))]
        ### cropped N_cells/N_events histogram
        time_bin_edges = np.linspace(
            (-(int(t_range / 2)) + focal_time),
            (int(t_range / 2) + focal_time),
            num_bins + 1,
        )  ## 2dimensionalise
        distance_bin_edges = np.linspace(0, radius, num_bins + 1)  ## 2dimensionalise
        N_cells_cropped_hist, x_autolabels, y_autolabels = np.histogram2d(
            N_cells_distance_cropped,
            N_cells_time_cropped,
            bins=[distance_bin_edges, time_bin_edges],
        )
        N_events_cropped_hist, x_autolabels, y_autolabels = np.histogram2d(
            N_events_distance_cropped,
            N_events_time_cropped,
            bins=[distance_bin_edges, time_bin_edges],
        )
        ## make cumulative cropped histogram
        N_cells_cropped_hist_cumulative += N_cells_cropped_hist
        N_events_cropped_hist_cumulative += N_events_cropped_hist

        ### update progress bar
        # progress_bar.update(int(N/10))

    P_events = N_events_cropped_hist_cumulative / N_cells_cropped_hist_cumulative

    ### info for rendering of graph ### change with new fns when ready
    focal_cell = re.findall(r"(?<=Pos._)[a-zA-Z]+", N_cells_fns[0])[0]
    subject_cell = re.findall(r"(?<=N_cells_)[a-zA-Z]+", N_cells_fns[0])[0]
    focal_event = "apoptosis"
    subject_event = "division"

    ### close progress bar
    # progress_bar.close()

    return N_cells_cropped_hist_cumulative, N_events_cropped_hist_cumulative, P_events


def old_cumulative_kymo_compiler(raw_files_dir, radius, t_range, num_bins):
    """
    This function compiles kymograph from raw list files directory
    """
    ### find individual radial scan raw files
    N_cells_fns = natsorted(glob.glob(os.path.join(raw_files_dir, "*cells*")))
    N_events_fns = natsorted(glob.glob(os.path.join(raw_files_dir, "*events*")))
    N_cells_cropped_hist_cumulative, N_events_cropped_hist_cumulative = (
        np.zeros((num_bins, num_bins)),
        np.zeros((num_bins, num_bins)),
    )
    ### ascertain the maximum scan parameters of chosen files
    radius_max = int(re.findall(r"rad_(\d+)", N_cells_fns[0])[0])
    t_range_max = int(re.findall(r"t_range_(\d+)", N_cells_fns[0])[0])

    ### iterate through all files compiling into cumulative
    N = len(N_cells_fns)
    # ### set iteration updates from tqdm
    # progress_bar = tqdm(total = N)
    for i, N_cells_fn in tqdm_notebook(
        enumerate(N_cells_fns), total=N
    ):  # tqdm(N_cells_fns):

        ## find events fn
        # N_events_fn = N_cells_fn.replace('cells_wt','events_wtdiv')
        N_events_fn = N_events_fns[
            i
        ]  ## more sustainable method for when subj event is not division
        ### checking if fns are for same scan
        if N_cells_fn.split("_N_")[0] != N_events_fn.split("_N_")[0]:
            print(
                "Filename error, could not guarantee that N_cells and N_events are from the same scan. Filenames:",
                N_cells_fn,
                N_events_fn,
            )
        elif int(re.findall(r"focal_t_(\d+)", N_cells_fn)[0]) != int(
            re.findall(r"focal_t_(\d+)", N_events_fn)[0]
        ):
            print(
                "Filename error, could not guarantee that N_cells and N_events have same focal time. Filenames:",
                N_cells_fn,
                N_events_fn,
            )

        ## find focal time
        focal_time = int(re.findall(r"focal_t_(\d+)", N_cells_fn)[0])

        ## load data
        N_cells, N_events = [], []
        with open(N_cells_fn, newline="") as csvfile:
            N_cells_import_csv = csv.reader(csvfile, delimiter="\n")  # , quotechar='|')
            for row in N_cells_import_csv:
                N_cells.append(eval(row[0]))
        with open(N_events_fn, newline="") as csvfile:
            N_events_import_csv = csv.reader(
                csvfile, delimiter="\n"
            )  # , quotechar='|')
            for row in N_events_import_csv:
                N_events.append(eval(row[0]))

        ### make cropped list of distance and time
        N_cells_cropped = [
            i
            for i in N_cells
            if i[1] < radius
            and (focal_time - (t_range / 2)) <= i[2] < (focal_time + (t_range / 2))
        ]
        N_cells_distance_cropped = [
            N_cells_cropped[i][1] for i in range(0, len(N_cells_cropped))
        ]
        N_cells_time_cropped = [
            N_cells_cropped[i][2] for i in range(0, len(N_cells_cropped))
        ]
        N_events_cropped = [
            i
            for i in N_events
            if i[1] < radius
            and (focal_time - (t_range / 2)) <= i[2] < (focal_time + (t_range / 2))
        ]
        N_events_distance_cropped = [
            N_events_cropped[i][1] for i in range(0, len(N_events_cropped))
        ]
        N_events_time_cropped = [
            N_events_cropped[i][2] for i in range(0, len(N_events_cropped))
        ]
        ## cropped N_cells/N_events histogram
        time_bin_edges = np.linspace(
            (-(int(t_range / 2)) + focal_time),
            (int(t_range / 2) + focal_time),
            num_bins + 1,
        )  ## 2dimensionalise
        distance_bin_edges = np.linspace(0, radius, num_bins + 1)  ## 2dimensionalise
        N_cells_cropped_hist, x_autolabels, y_autolabels = np.histogram2d(
            N_cells_distance_cropped,
            N_cells_time_cropped,
            bins=[distance_bin_edges, time_bin_edges],
        )
        N_events_cropped_hist, x_autolabels, y_autolabels = np.histogram2d(
            N_events_distance_cropped,
            N_events_time_cropped,
            bins=[distance_bin_edges, time_bin_edges],
        )
        ## make cumulative cropped histogram
        N_cells_cropped_hist_cumulative += N_cells_cropped_hist
        N_events_cropped_hist_cumulative += N_events_cropped_hist

        ### update progress bar
        # progress_bar.update(int(N/10))

    P_events = N_events_cropped_hist_cumulative / N_cells_cropped_hist_cumulative

    ### info for rendering of graph ### change with new fns when ready
    focal_cell = re.findall(r"(?<=Pos._)[a-zA-Z]+", N_cells_fns[0])[0]
    subject_cell = re.findall(r"(?<=N_cells_)[a-zA-Z]+", N_cells_fns[0])[0]
    focal_event = "apoptosis"
    subject_event = "division"

    ### close progress bar
    # progress_bar.close()

    return N_cells_cropped_hist_cumulative, N_events_cropped_hist_cumulative, P_events


def auto_plot_cumulative_defaulttext(
    input_2d_hist,
    input_type,
    N,
    num_bins,
    radius,
    t_range,
    focal_cell,
    focal_event,
    subject_cell,
    subject_event,
    save_parent_dir,
    cbar_lim,
    include_apop_bin,
    SI,
):

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 1, radius, t_range, SI)

    ## formatting cell and event names
    focal_event_name = (
        "apoptoses" if "apop" in focal_event.lower() else "divisions"
    )  # focal_event == 'APOPTOSIS' or 'apop' else 'divisions'
    focal_cell_name = "wild-type" if "wt" in focal_cell.lower() else "Scribble"
    subj_event_name = "apoptoses" if "apop" in subject_event.lower() else "divisions"
    subj_cell_name = "wild-type" if "wt" in subject_cell.lower() else "Scribble"

    if focal_event == "control":
        focal_event_name = "random time points"

    title = (
        "Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
    )

    ## output save path formatting
    save_dir_name = "{}_{}_{}_{}".format(
        focal_cell.lower(),
        focal_event.lower()[0:3]
        if focal_event == "DIVISION"
        else focal_event.lower()[0:4],
        subject_cell.lower(),
        subject_event.lower()[0:3]
        if subject_event == "DIVISION"
        else subject_event.lower()[0:4],
    )
    save_dir = f"{radius}.{t_range}.{num_bins}/{save_dir_name}"
    save_path = os.path.join(save_parent_dir, save_dir)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    ## title formatting
    if input_type == "N_cells":
        title = "Spatiotemporal dist. of {} cells \n around {} {} (N={})".format(
            subj_cell_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Number of {subj_cell_name} cell apperances"
    if input_type == "N_events":
        title = "Spatiotemporal dist. of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Number of {subj_cell_name} {subj_event_name}"
    if input_type == "P_events":
        title = "Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Probability of {subj_cell_name} {subj_event_name}"

    ## label unit formatting
    if SI == True:
        time_unit = "(Hours)"
        distance_unit = "(Micrometers)"
    else:
        time_unit = "(Frames)"
        distance_unit = "(Pixels)"

    ## plotting
    plt.xticks(xlocs, xlabels, rotation="vertical")
    plt.yticks(ylocs, ylabels)
    plt.xlabel(
        "Time since:42 Module docstring appears after code  apoptosis " + time_unit
    )
    plt.ylabel("Distance from apoptosis " + distance_unit)
    plt.title(title + "\n", fontweight="bold")

    ## if include_apop_bin is true then the spatial bin containing the apoptotic cell (ie the central spatial bin of the radial scan) will be show in the graph, if false then it is cropped which ends up with a plot showing only the relevant local env not the site of apop (better imo)
    if include_apop_bin == True:
        final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    else:
        final_hist = np.flipud(input_2d_hist[1:-1, :])

    plt.imshow(final_hist)

    if cbar_lim == "":
        # plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        plt.colorbar(label=cb_label)
    else:
        plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        plt.colorbar(label=cb_label)

    ## apop location marker
    if include_apop_bin == True:
        plt.scatter(num_bins / 2 - 0.5, num_bins - 0.75, s=20, c="white", marker="v")
        plt.text(
            num_bins + 0.15,
            num_bins + 1.5,
            "Apoptosis location \nshown by inverted \nwhite triangle",
        )
    else:
        plt.scatter(
            num_bins / 2 - 0.5, num_bins - 2 - 0.75, s=20, c="white", marker="v"
        )
        plt.text(
            num_bins + 0.15,
            num_bins + 1.5 - 2,
            "Apoptosis location \nshown by inverted \nwhite triangle",
        )

    fn = os.path.join(save_path, title + ".pdf")

    ## failsafe overwriting block
    if os.path.exists(fn):
        print("Filename", fn, "already exists, saving as updated copy")
        fn = fn.replace(
            ".pdf", " (updated {}).pdf".format(time.strftime("%Y%m%d-%H%M%S"))
        )

    ## save out?
    if save_parent_dir == "":
        return plt.imshow(final_hist)
    else:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        print("Plot saved at ", fn)
        return plt.imshow(final_hist)

def auto_plot_cumulative(
input_dict
):
    plt.clf()
    # check input has necessary components
    assert 'input_2d_hist' in input_dict, "No input histogram found!"
    # check input is correct type
    if not isinstance(input_dict['input_2d_hist'], np.ndarray):
        raise ValueError(f"Input heatmap {input_dict['input_2d_hist']}, is not a ndarray")
    ### read input parameters and data
    input_2d_hist = input_dict.get('input_2d_hist')
    # load other params necessary for plotting
    radius = int(input_dict.get('radius', input('Enter the scan radius')))
    t_range = int(input_dict.get('t_range', input('Enter the scan time range')))

    ### default to have standard measures
    SI = input_dict.get('SI', True)

    ### default to read the number of bins from the array shape
    num_bins = input_dict.get('num_bins', input_2d_hist.shape[0])
    ### default is to have no cbar_lim
    cbar_lim = input_dict.get('cbar_lim', False)

    ### set the label frequency according to the num bins
    label_freq = 4 if num_bins > 20 else 1

    # get correctly spaced labels
    xlocs, xlabels, ylocs, ylabels = kymo_labels(
        num_bins, label_freq, radius, t_range, SI
    )

    # get scan details for labelling
    # ask for title if you havent specified input type
    if 'input_type' not in input_dict:
        title = input('Input the title of your plot')
        cb_label = input('Input the label for your colorbar')
        focal_event_name = input_dict.get('focal_event', input('What is the focal event of the radial scan?'))
        x_axis_label = f"Time since {focal_event_name} "
        y_axis_label = f"Distance from {focal_event_name} "
    #else get focal cell etc details or ask for them if not present
    else:
        focal_cell_name = input_dict.get('focal_cell', input('What is the focal cell of the radial scan?'))
        focal_event_name = input_dict.get('focal_event', input('What is the focal event of the radial scan?'))
        subject_cell_name = input_dict.get('subject_cell', input('What is the subject cell of the radial scan?'))
        subject_event_name = input_dict.get('subject_event', input('What is the subject event of the radial scan?'))
        N = int(input_dict.get('N', input('Enter the number of focal events')))
        # now check which input type is specified
        if input_dict.get('input_type') == "N_cells":
            title = f"Spatiotemporal dist. of {subj_cell_name} cells \n around {focal_cell_name} {focal_event_name} (N={N})"
            cb_label = f"Number of {subj_cell_name} cell apperances"

        elif input_dict.get('input_type') == "N_events":
            title = f"Spatiotemporal dist. of {subj_cell_name} {subj_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
            cb_label = f"Number of {subj_cell_name} {subj_event_name}"

        elif input_dict.get('input_type') == "P_events":
            title = f"Spatiotemporal dist. of probability of {subj_cell_name} {subj_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
            cb_label = f"Probability of {subj_cell_name} {subj_event_name}"

        elif input_dict.get('input_type') == "CV":
            title = f"Coefficient of variation of probability of {subj_cell_name} {subj_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
            cb_label = "Coefficient of variation"

        elif input_dict.get('input_type') == "stat_rel":
            title = f"Statisticall relevant areas of probability of {subj_cell_name} {subj_event_name} \n around {focal_cell_name} {focal_event_name} (N={N})"
            cb_label = "Relevant areas are set equal to 1"

        elif input_dict.get('input_type') == "dP":
            title = f"Difference in probability between \ncanonical and control analysis \ni.e. probability of {subject_event_name} above background"
            cb_label = "Difference in probability\n above background"
        else:
            print('input_type not recognised')

    ## label unit formatting
    if SI == True:
        time_unit = "(Hours)"
        distance_unit = "(Micrometers)"
    else:
        time_unit = "(Frames)"
        distance_unit = "(Pixels)"

    ## plotting
    font = {"fontname": "Liberation Mono"}
    plt.xticks(xlocs, xlabels, rotation="vertical", **font)
    plt.yticks(ylocs, ylabels, **font)
    plt.xlabel(x_axis_label + time_unit, **font)
    plt.ylabel(y_axis_label + distance_unit, **font)
    plt.title(title + "\n", fontweight="bold", **font)

    ### default is to include the apoptotic spatial bin
    include_apop_bin = input_dict.get('include_apop_bin', True)
    ## if include_apop_bin is true then the spatial bin containing the apoptotic cell (ie the central spatial bin of the radial scan) will be show in the graph, if false then it is cropped which ends up with a plot showing only the relevant local env not the site of apop (better imo)
    if include_apop_bin == True:
        final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    else:
        final_hist = np.flipud(input_2d_hist[1:, :])

    ## apop location marker
    if num_bins == 10:
        if include_apop_bin == True:
            plt.scatter(
                num_bins / 2 - 0.5, num_bins - 0.75, s=20, c="white", marker="v"
            )
            plt.text(
                num_bins + 0.15,
                num_bins + 1.5,
                f"{focal_event_name} location \nshown by inverted \nwhite triangle",
                **font,
            )
        else:
            plt.scatter(
                num_bins / 2 - 0.5, num_bins - 2 - 0.75, s=20, c="white", marker="v"
            )
            plt.text(
                num_bins + 0.15,
                num_bins + 1.5 - 2,
                f"{focal_event_name} location \nshown by inverted \nwhite triangle",
                **font,
            )
    if num_bins == 20:
        if include_apop_bin == True:
            plt.scatter(num_bins / 2 - 0.5, num_bins - 0.9, s=20, c="white", marker="v")
            plt.text(
                num_bins + 0.3,
                num_bins + 3.5,
                f"{focal_event_name} location \nshown by inverted \nwhite triangle",
                **font,
            )
        else:
            plt.scatter(num_bins / 2 - 0.5, num_bins - 1.8, s=20, c="white", marker="v")
            plt.text(
                num_bins + 0.3,
                num_bins + 2.5,
                f"{focal_event_name} location \nshown by inverted \nwhite triangle",
                **font,
            )

    ### default is to have no cbar_lim
    cbar_lim = input_dict.get('cbar_lim', False)
    ## colorbar
    if cbar_lim == False:
        if include_apop_bin == False:
            plt.clim(
                vmin=np.min(input_2d_hist[1:-1, :]), vmax=np.max(input_2d_hist[1:-1, :])
            )
        else:
            plt.clim(vmin=np.min(input_2d_hist), vmax=np.max(input_2d_hist))
        cb = plt.colorbar(
            label=cb_label
        )  ### matplotlib.cm.ScalarMappable(norm = ???cmap='PiYG'), use this in conjunction with norm to set cbar equal to diff piyg coloourscheme
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family="Liberation Mono")
        text.set_font_properties(font)
        ax.set_yticklabels(
            np.round(ax.get_yticks(), 5), **{"fontname": "Liberation Mono"}
        )  ### cropped to 5dp
    else:
        cbar_lim = input_dict['cbar_lim']
        plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        cb = plt.colorbar(label=cb_label)
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family="Liberation Mono")
        text.set_font_properties(font)
        ax.set_yticklabels(
            np.round(ax.get_yticks(), 5), **{"fontname": "Liberation Mono"}
        )

    ### default to have no bin labels
    bin_labels = input_dict.get('bin_labels', False)
    ## bin labels
    if bin_labels == True:
        flipped = np.flipud(input_2d_hist)
        if input_type == "P_events":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 5),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        elif input_type == "dP":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 6),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        elif input_type == "CV":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 3),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        if input_type == "stat_rel":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        int(flipped[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        else:
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        int(flipped[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
    ## save out?
    if 'save_parent_dir' not in input_dict:
        plt.imshow(final_hist)
        return   # ,cmap = 'PiYG')
    else:
        ## output save path formatting
        save_parent_dir = input_dict['save_parent_dir']
        save_dir_name = "{}_{}_{}_{}".format(
            focal_cell.lower(),
            focal_event.lower()[0:3]
            if focal_event == "DIVISION"
            else focal_event.lower()[0:4],
            subject_cell.lower(),
            subject_event.lower()[0:3]
            if subject_event == "DIVISION"
            else subject_event.lower()[0:4],
        )
        save_path = os.path.join(save_parent_dir, save_dir_name)
        # if (
        #     not input_type == "dP"
        # ):  ### combined type does not require segregated folders for canon control
        Path(save_path).mkdir(parents=True, exist_ok=True)

        ## filename
        fn = save_path + "/" + title + f" {radius}.{t_range}.{num_bins}.pdf"
        ## failsafe overwriting block
        if os.path.exists(fn):
            print("Filename", fn, "already exists, saving as updated copy")
            fn = fn.replace(
                ".pdf", " (updated {}).pdf".format(time.strftime("%Y%m%d-%H%M%S"))
            )
        plt.imshow(final_hist)
        plt.plot()
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        print("Plot saved at ", fn)

        return plt.imshow(final_hist)

def auto_plot_cumulative_legacy(
    input_dict
    ):
    """
    Older version without manual entry options
    """

    plt.clf()
    ### read input parameters and data
    input_2d_hist = input_dict['input_2d_hist']
    input_type = input_dict['input_type']
    N = input_dict['N']
    radius = input_dict['radius']
    t_range = input_dict['t_range']
    focal_cell = input_dict['focal_cell']
    focal_event = input_dict['focal_event']
    subject_cell = input_dict['subject_cell']
    subject_event = input_dict['subject_event']

    ### set the default params if param entry in dict is empty
    ### default is to include the apoptotic spatial bin
    if 'include_apop_bin' not in input_dict:
        include_apop_bin = True
    else:
        include_apop_bin = input_dict['include_apop_bin']
    ### default to have standard measures
    if 'SI' not in input_dict:
        SI = True
    else:
        SI = input_dict['SI']
    ### default to have no bin labels
    if 'bin_labels' not in input_dict:
        bin_labels = False
    else:
        bin_labels = input_dict['bin_labels']
    ### default to read the number of bins from the array shape
    if 'num_bins' not in input_dict:
        num_bins = input_2d_hist.shape[0]
    else:
        num_bins = input_dict['num_bins']
    ### default is to have no cbar_lim
    if 'cbar_lim' not in input_dict:
        cbar_lim = False
    else:
        cbar_lim = input_dict['cbar_lim']

    ### set the label frequency according to the num bins
    if num_bins > 20:
        label_freq = 4
    else:
        label_freq = 1
    xlocs, xlabels, ylocs, ylabels = kymo_labels(
        num_bins, label_freq, radius, t_range, SI
    )

    ## formatting cell and event names
    focal_event_name = (
        "apoptoses" if "apop" in focal_event.lower() else "divisions"
    )  # focal_event == 'APOPTOSIS' or 'apop' else 'divisions'
    focal_cell_name = "wild-type" if "wt" in focal_cell.lower() else "Scribble"
    subj_event_name = "apoptoses" if "apop" in subject_event.lower() else "divisions"
    subj_cell_name = "wild-type" if "wt" in subject_cell.lower() else "Scribble"

    if focal_event == "control":
        focal_event_name = "random time points"

    title = (
        "Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
    )

    ## title formatting
    if input_type == "N_cells":
        title = "Spatiotemporal dist. of {} cells \n around {} {} (N={})".format(
            subj_cell_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Number of {subj_cell_name} cell apperances"
    if input_type == "N_events":
        title = "Spatiotemporal dist. of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Number of {subj_cell_name} {subj_event_name}"
    if input_type == "P_events":
        title = "Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Probability of {subj_cell_name} {subj_event_name}"
    if input_type == "CV":
        title = "Coefficient of variation of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = "Coefficient of variation"
    if input_type == "stat_rel":
        title = "Statisticall relevant areas of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = "Relevant areas are set equal to 1"
    if input_type == "dP":
        title = "Difference in probability between \ncanonical and control analysis \ni.e. probability of division above background".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = "Difference in probability\n above background"

    ## label unit formatting
    if SI == True:
        time_unit = "(Hours)"
        distance_unit = "(Micrometers)"
    else:
        time_unit = "(Frames)"
        distance_unit = "(Pixels)"

    ## plotting
    font = {"fontname": "Liberation Mono"}
    plt.xticks(xlocs, xlabels, rotation="vertical", **font)
    plt.yticks(ylocs, ylabels, **font)
    plt.xlabel("Time since apoptosis " + time_unit, **font)
    plt.ylabel("Distance from apoptosis " + distance_unit, **font)
    plt.title(title + "\n", fontweight="bold", **font)

    ## if include_apop_bin is true then the spatial bin containing the apoptotic cell (ie the central spatial bin of the radial scan) will be show in the graph, if false then it is cropped which ends up with a plot showing only the relevant local env not the site of apop (better imo)
    if include_apop_bin == True:
        final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    else:
        final_hist = np.flipud(input_2d_hist[1:-1, :])

    ## apop location marker
    if num_bins == 10:
        if include_apop_bin == True:
            plt.scatter(
                num_bins / 2 - 0.5, num_bins - 0.75, s=20, c="white", marker="v"
            )
            plt.text(
                num_bins + 0.15,
                num_bins + 1.5,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )
        else:
            plt.scatter(
                num_bins / 2 - 0.5, num_bins - 2 - 0.75, s=20, c="white", marker="v"
            )
            plt.text(
                num_bins + 0.15,
                num_bins + 1.5 - 2,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )
    if num_bins == 20:
        if include_apop_bin == True:
            plt.scatter(num_bins / 2 - 0.5, num_bins - 0.9, s=20, c="white", marker="v")
            plt.text(
                num_bins + 0.3,
                num_bins + 3.5,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )
        else:
            plt.scatter(num_bins / 2 - 0.5, num_bins - 1.8, s=20, c="white", marker="v")
            plt.text(
                num_bins + 0.3,
                num_bins + 2.5,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )

    ## colorbar
    if cbar_lim == False:
        if include_apop_bin == False:
            plt.clim(
                vmin=np.min(input_2d_hist[1:-1, :]), vmax=np.max(input_2d_hist[1:-1, :])
            )
        else:
            plt.clim(vmin=np.min(input_2d_hist), vmax=np.max(input_2d_hist))
        cb = plt.colorbar(
            label=cb_label
        )  ### matplotlib.cm.ScalarMappable(norm = ???cmap='PiYG'), use this in conjunction with norm to set cbar equal to diff piyg coloourscheme
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family="Liberation Mono")
        text.set_font_properties(font)
        ax.set_yticklabels(
            np.round(ax.get_yticks(), 5), **{"fontname": "Liberation Mono"}
        )  ### cropped to 5dp
    else:
        cbar_lim = input_dict['cbar_lim']
        plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        cb = plt.colorbar(label=cb_label)
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family="Liberation Mono")
        text.set_font_properties(font)
        ax.set_yticklabels(
            np.round(ax.get_yticks(), 5), **{"fontname": "Liberation Mono"}
        )

    ## bin labels
    if bin_labels == True:
        flipped = np.flipud(input_2d_hist)
        if input_type == "P_events":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 5),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        elif input_type == "dP":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 6),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        elif input_type == "CV":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 3),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        if input_type == "stat_rel":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        int(flipped[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        else:
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        int(flipped[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
    ## save out?
    if 'save_parent_dir' not in input_dict:
        plt.imshow(final_hist)
        return   # ,cmap = 'PiYG')
    else:
        ## output save path formatting
        save_parent_dir = input_dict['save_parent_dir']
        save_dir_name = "{}_{}_{}_{}".format(
            focal_cell.lower(),
            focal_event.lower()[0:3]
            if focal_event == "DIVISION"
            else focal_event.lower()[0:4],
            subject_cell.lower(),
            subject_event.lower()[0:3]
            if subject_event == "DIVISION"
            else subject_event.lower()[0:4],
        )
        save_path = os.path.join(save_parent_dir, save_dir_name)
        # if (
        #     not input_type == "dP"
        # ):  ### combined type does not require segregated folders for canon control
        Path(save_path).mkdir(parents=True, exist_ok=True)

        ## filename
        fn = save_path + "/" + title + f" {radius}.{t_range}.{num_bins}.pdf"
        ## failsafe overwriting block
        if os.path.exists(fn):
            print("Filename", fn, "already exists, saving as updated copy")
            fn = fn.replace(
                ".pdf", " (updated {}).pdf".format(time.strftime("%Y%m%d-%H%M%S"))
            )
        plt.imshow(final_hist)
        plt.plot()
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        print("Plot saved at ", fn)

        return plt.imshow(final_hist)

def MEGAPLOT(
    input_dict
):

    #### MEGAPLOT
    print("Thank you for choosing MEGAPLOT\n\n")
    ### load input arrays
    N_cells = input_dict['N_cells']
    N_events = input_dict['N_events']
    P_events = input_dict['P_events']
    N_cells_c = input_dict['N_cells_c']
    N_events_c = input_dict['N_events_c']
    P_events_c = input_dict['P_events_c']
    N, N_c = input_dict['N'], input_dict['N_c']
    save_parent_dir = input_dict['save_parent_dir']

    ### set few default options (until they are changed)
    input_dict['include_apop_bin'] = True
    input_dict['bin_labels'] = False
    input_dict['SI'] = True
    input_dict['cbar_lim'] = False

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "uncrop_unlim")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_events
    input_dict['input_2d_hist'] = N_events
    input_dict['input_type'] = 'N_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_cells
    input_dict['input_2d_hist'] = N_cells
    input_dict['input_type'] = 'N_cells'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_events
    input_dict['input_2d_hist'] = N_events_c
    input_dict['input_type'] = 'N_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_cells
    input_dict['input_2d_hist'] = N_cells_c
    input_dict['input_type'] = 'N_cells'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###################### plot with cbar_lim ##################################
    input_dict['cbar_lim'] = tuple((0, max(np.amax(P_events_c), np.amax(P_events))))

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "cbar_lim")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### reset previous params
    input_dict['cbar_lim'] = False

    ######################## plot with crop ########################
    input_dict['include_apop_bin'] = False

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "crop")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### reset previous params
    input_dict['include_apop_bin'] = True

    ##################### plot with cbar lim crop #####################
    input_dict['cbar_lim'] = tuple((0, max(np.amax(P_events_c), np.amax(P_events))))
    input_dict['include_apop_bin'] = False

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "cbar_lim_crop")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_eventsve(
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### reset previous params
    input_dict['include_apop_bin'] = True
    input_dict['cbar_lim'] = False

    ########################### coeff var ##########################

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "CV")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    #### calculate canon CV
    cv = np.nan_to_num(np.sqrt((1 - P_events) / (P_events * N_cells)), posinf=1)

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type cv
    input_dict['input_2d_hist'] = cv
    input_dict['input_type'] = 'CV'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### calculate control CV
    cv_c = np.nan_to_num(np.sqrt((1 - P_events_c) / (P_events_c * N_cells_c)), posinf=1)

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type cv_c
    input_dict['input_2d_hist'] = cv_c
    input_dict['input_type'] = 'CV'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### combined coeff var

    stat_rel = calculate.stat_relevance_calc(input_dict['num_bins'], P_events, P_events_c, cv, cv_c)

    ### params for canon - necessary here?
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type stat_rel
    input_dict['input_2d_hist'] = stat_rel
    input_dict['input_type'] = 'stat_rel'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ########################## combined plot ##########################

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "combined")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### calculate dP
    dP_events = P_events - P_events_c

    ### params for canon - necessary here?
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type stat_rel
    input_dict['input_2d_hist'] = dP_events
    input_dict['input_type'] = 'dP'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ########################## LABELLED PLOTS #############################
    print("Saving out labelled plots\n\n")
    input_dict['bin_labels'] = True
    save_parent_dir = os.path.join(save_parent_dir, 'labelled')


    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "uncrop_unlim")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_events
    input_dict['input_2d_hist'] = N_events
    input_dict['input_type'] = 'N_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_cells
    input_dict['input_2d_hist'] = N_cells
    input_dict['input_type'] = 'N_cells'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_events
    input_dict['input_2d_hist'] = N_events_c
    input_dict['input_type'] = 'N_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### specify graph type N_cells
    input_dict['input_2d_hist'] = N_cells_c
    input_dict['input_type'] = 'N_cells'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###################### plot with cbar_lim ##################################
    input_dict['cbar_lim'] = tuple((0, max(np.amax(P_events_c), np.amax(P_events))))

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "cbar_lim")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### reset previous params
    input_dict['cbar_lim'] = False

    ######################## plot with crop ########################
    input_dict['include_apop_bin'] = False

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "crop")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### reset previous params
    input_dict['include_apop_bin'] = True

    ##################### plot with cbar lim crop #####################
    input_dict['cbar_lim'] = tuple((0, max(np.amax(P_events_c), np.amax(P_events))))
    input_dict['include_apop_bin'] = False

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "cbar_lim_crop")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type P_events
    input_dict['input_2d_hist'] = P_events_c
    input_dict['input_type'] = 'P_events'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### reset previous params
    input_dict['include_apop_bin'] = True
    input_dict['cbar_lim'] = False

    ########################### coeff var ##########################

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "CV")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    #### calculate canon CV
    cv = np.nan_to_num(np.sqrt((1 - P_events) / (P_events * N_cells)), posinf=1)

    ### params for canon
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type cv
    input_dict['input_2d_hist'] = cv
    input_dict['input_type'] = 'CV'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ###### calculate control CV
    cv_c = np.nan_to_num(np.sqrt((1 - P_events_c) / (P_events_c * N_cells_c)), posinf=1)

    ###### params for control
    input_dict['focal_cell'] = "wt"
    input_dict['focal_event'] = "control"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N_c

    ### specify graph type cv_c
    input_dict['input_2d_hist'] = cv_c
    input_dict['input_type'] = 'CV'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ### combined coeff var

    stat_rel = calculate.stat_relevance_calc(input_dict['num_bins'], P_events, P_events_c, cv, cv_c)

    ### params for canon - necessary here?
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type stat_rel
    input_dict['input_2d_hist'] = stat_rel
    input_dict['input_type'] = 'stat_rel'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()

    ########################## combined plot ##########################

    ### make output dir
    input_dict['save_parent_dir'] = os.path.join(save_parent_dir, "combined")
    if not os.path.exists(input_dict['save_parent_dir']):
        os.makedirs(input_dict['save_parent_dir'])

    ### calculate dP
    dP_events = P_events - P_events_c

    ### params for canon - necessary here?
    input_dict['focal_cell'] = "Scr"
    input_dict['focal_event'] = "apop"
    input_dict['subject_cell'] = "wt"
    input_dict['subject_event'] = "div"
    input_dict['N'] = N

    ### specify graph type stat_rel
    input_dict['input_2d_hist'] = dP_events
    input_dict['input_type'] = 'dP'
    auto_plot_cumulative(
        input_dict
    )
    plt.clf()
    return print("Plots saved out")


def auto_plot_cumulative_old(
    input_2d_hist,
    input_type,
    N,
    num_bins,
    radius,
    t_range,
    focal_cell,
    focal_event,
    subject_cell,
    subject_event,
    save_parent_dir,
    cbar_lim,
    include_apop_bin,
    bin_labels,
    SI,
):

    if num_bins > 20:
        label_freq = 4
    else:
        label_freq = 1
    xlocs, xlabels, ylocs, ylabels = kymo_labels(
        num_bins, label_freq, radius, t_range, SI
    )

    ## formatting cell and event names
    focal_event_name = (
        "apoptoses" if "apop" in focal_event.lower() else "divisions"
    )  # focal_event == 'APOPTOSIS' or 'apop' else 'divisions'
    focal_cell_name = "wild-type" if "wt" in focal_cell.lower() else "Scribble"
    subj_event_name = "apoptoses" if "apop" in subject_event.lower() else "divisions"
    subj_cell_name = "wild-type" if "wt" in subject_cell.lower() else "Scribble"

    if focal_event == "control":
        focal_event_name = "random time points"

    title = (
        "Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
    )

    ## output save path formatting
    save_dir_name = "{}_{}_{}_{}".format(
        focal_cell.lower(),
        focal_event.lower()[0:3]
        if focal_event == "DIVISION"
        else focal_event.lower()[0:4],
        subject_cell.lower(),
        subject_event.lower()[0:3]
        if subject_event == "DIVISION"
        else subject_event.lower()[0:4],
    )
    save_path = os.path.join(save_parent_dir, save_dir_name)
    if (
        not input_type == "dP"
    ):  ### combined type does not require segregated folders for canon control
        Path(save_path).mkdir(parents=True, exist_ok=True)

    ## title formatting
    if input_type == "N_cells":
        title = "Spatiotemporal dist. of {} cells \n around {} {} (N={})".format(
            subj_cell_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Number of {subj_cell_name} cell apperances"
    if input_type == "N_events":
        title = "Spatiotemporal dist. of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Number of {subj_cell_name} {subj_event_name}"
    if input_type == "P_events":
        title = "Spatiotemporal dist. of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = f"Probability of {subj_cell_name} {subj_event_name}"
    if input_type == "CV":
        title = "Coefficient of variation of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = "Coefficient of variation"
    if input_type == "stat_rel":
        title = "Statisticall relevant areas of probability of {} {} \n around {} {} (N={})".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = "Relevant areas are set equal to 1"
    if input_type == "dP":
        title = "Difference in probability between \ncanonical and control analysis \ni.e. probability of division above background".format(
            subj_cell_name, subj_event_name, focal_cell_name, focal_event_name, N
        )
        cb_label = "Difference in probability\n above background"
        save_path = save_parent_dir
    # else:
    #     title = ''
    #     cb_label = ''

    ## label unit formatting
    if SI == True:
        time_unit = "(Hours)"
        distance_unit = "(Micrometers)"
    else:
        time_unit = "(Frames)"
        distance_unit = "(Pixels)"

    ## plotting
    font = {"fontname": "Liberation Mono"}
    plt.xticks(xlocs, xlabels, rotation="vertical", **font)
    plt.yticks(ylocs, ylabels, **font)
    plt.xlabel("Time since apoptosis " + time_unit, **font)
    plt.ylabel("Distance from apoptosis " + distance_unit, **font)
    plt.title(title + "\n", fontweight="bold", **font)

    ## if include_apop_bin is true then the spatial bin containing the apoptotic cell (ie the central spatial bin of the radial scan) will be show in the graph, if false then it is cropped which ends up with a plot showing only the relevant local env not the site of apop (better imo)
    if include_apop_bin == True:
        final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    else:
        final_hist = np.flipud(input_2d_hist[1:-1, :])

    ## apop location marker
    if num_bins == 10:
        if include_apop_bin == True:
            plt.scatter(
                num_bins / 2 - 0.5, num_bins - 0.75, s=20, c="white", marker="v"
            )
            plt.text(
                num_bins + 0.15,
                num_bins + 1.5,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )
        else:
            plt.scatter(
                num_bins / 2 - 0.5, num_bins - 2 - 0.75, s=20, c="white", marker="v"
            )
            plt.text(
                num_bins + 0.15,
                num_bins + 1.5 - 2,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )
    if num_bins == 20:
        if include_apop_bin == True:
            plt.scatter(num_bins / 2 - 0.5, num_bins - 0.9, s=20, c="white", marker="v")
            plt.text(
                num_bins + 0.3,
                num_bins + 3.5,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )
        else:
            plt.scatter(num_bins / 2 - 0.5, num_bins - 1.8, s=20, c="white", marker="v")
            plt.text(
                num_bins + 0.3,
                num_bins + 2.5,
                "Apoptosis location \nshown by inverted \nwhite triangle",
                **font,
            )

    ## colorbar
    if cbar_lim == "":
        if include_apop_bin == False:
            plt.clim(
                vmin=np.min(input_2d_hist[1:-1, :]), vmax=np.max(input_2d_hist[1:-1, :])
            )
        else:
            plt.clim(vmin=np.min(input_2d_hist), vmax=np.max(input_2d_hist))
        cb = plt.colorbar(
            label=cb_label
        )  ### matplotlib.cm.ScalarMappable(norm = ???cmap='PiYG'), use this in conjunction with norm to set cbar equal to diff piyg coloourscheme
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family="Liberation Mono")
        text.set_font_properties(font)
        ax.set_yticklabels(
            np.round(ax.get_yticks(), 5), **{"fontname": "Liberation Mono"}
        )  ### cropped to 5dp
    else:
        plt.clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        cb = plt.colorbar(label=cb_label)
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family="Liberation Mono")
        text.set_font_properties(font)
        ax.set_yticklabels(
            np.round(ax.get_yticks(), 5), **{"fontname": "Liberation Mono"}
        )

    ## filename
    fn = save_path + "/" + title + f" {radius}.{t_range}.{num_bins}.pdf"
    ## failsafe overwriting block
    if os.path.exists(fn):
        print("Filename", fn, "already exists, saving as updated copy")
        fn = fn.replace(
            ".pdf", " (updated {}).pdf".format(time.strftime("%Y%m%d-%H%M%S"))
        )

    ## bin labels
    if bin_labels == True:
        flipped = np.flipud(input_2d_hist)
        # if num_bins != 10:
        #     print('Plotting bin labels only works for num_bins=10')
        # else:
        if input_type == "P_events":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 5),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        elif input_type == "dP":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 6),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        elif input_type == "CV":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        round(flipped[i, j], 3),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        if input_type == "stat_rel":
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        int(flipped[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
        else:
            for i in range(len(input_2d_hist)):
                for j in range(len(input_2d_hist)):
                    text = plt.text(
                        j,
                        i,
                        int(flipped[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )
    ## save out?
    if save_parent_dir == "":
        return plt.imshow(final_hist)  # ,cmap = 'PiYG')
    else:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        print("Plot saved at ", fn)
        return plt.imshow(final_hist)

def plot_cumulative(
    input_2d_hist, num_bins, radius, t_range, title, label, cb_label, save_path, SI
):

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 1, radius, t_range, SI)

    if SI == True:
        time_unit = "(Hours)"
        distance_unit = "(Micrometers)"
    else:
        time_unit = "(Frames)"
        distance_unit = "(Pixels)"
    plt.xticks(xlocs, xlabels, rotation="vertical")
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since apoptosis " + time_unit)
    plt.ylabel("Distance from apoptosis " + distance_unit)
    plt.title(title)

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    plt.colorbar(label=cb_label)

    if save_path == "":
        return plt.imshow(final_hist)
    else:
        plt.savefig(
            os.path.join(save_path, title + ".pdf"), dpi=300, bbox_inches="tight"
        )
        print("Plot saved at ", (os.path.join(save_path, title + ".pdf")))
        return plt.imshow(final_hist)


def plot_N_cells(
    input_2d_hist, subject_cells, target_cell, focal_time, radius, t_range
):

    if target_cell.ID < 0:
        cell_type = "Scr"
    if target_cell.ID > 0:
        cell_type = "WT"
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 2, radius, t_range, SI=False)

    # expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    # plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation="vertical")
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time) + " (frames)")
    plt.ylabel("Distance from focal event (pixels)")
    plt.title(
        "Kymograph for "
        + target_cell.fate.name.lower()
        + " "
        + cell_type
        + " ID:"
        + str(cell_ID)
        + " at t= "
        + str(focal_time)
    )

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = "Number of wild-type cells"
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = "Number of Scribble cells? AMEND THIS LABEL"

    # if event == 'APOPTOSIS':
    #     raise Exception('Apoptosis event counter not configured yet')

    plt.colorbar(label=cb_label)

    return plt.imshow(final_hist)


def plot_N_events(
    input_2d_hist, event, subject_cells, target_cell, focal_time, radius, t_range
):

    if target_cell.ID < 0:
        cell_type = "Scr"
    if target_cell.ID > 0:
        cell_type = "WT"
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 2, radius, t_range, SI=False)

    # expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    # plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation="vertical")
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time) + " (frames)")
    plt.ylabel("Distance from focal event (pixels)")
    plt.title(
        "Kymograph for "
        + target_cell.fate.name
        + " "
        + cell_type
        + " ID:"
        + str(cell_ID)
        + " at t= "
        + str(focal_time)
    )

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = "Probability of wild-type mitoses"
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = "Probability of Scribble mitoses? AMEND THIS LABEL"
    if event == "APOPTOSIS":
        raise Exception("Apoptosis event counter not configured yet")

    plt.colorbar(label=cb_label)

    return plt.imshow(final_hist)


def plot_P_events(
    input_2d_hist, event, subject_cells, target_cell, focal_time, radius, t_range
):

    if target_cell.ID < 0:
        cell_type = "Scr"
    if target_cell.ID > 0:
        cell_type = "WT"
    cell_ID = target_cell.ID

    num_bins = len(input_2d_hist)

    xlocs, xlabels, ylocs, ylabels = kymo_labels(num_bins, 2, radius, t_range)

    # expt_label = 'expt:' + expt_ID + '\n 90:10 WT:Scr\n'
    # plt.text(num_bins+1,num_bins+4,expt_label)
    plt.xticks(xlocs, xlabels, rotation="vertical")
    plt.yticks(ylocs, ylabels)
    plt.xlabel("Time since t = " + str(focal_time) + " (frames)")
    plt.ylabel("Distance from focal event (pixels)")
    plt.title(
        "Kymograph for "
        + target_cell.fate.name
        + " "
        + cell_type
        + " ID:"
        + str(cell_ID)
        + " at t= "
        + str(focal_time)
    )

    final_hist = np.flipud(input_2d_hist)  ## flip for desired graph orientation
    plt.imshow(final_hist)

    if min([cell.ID for cell in subject_cells]) > 0:
        cb_label = "Probability of wild-type mitoses"
    if min([cell.ID for cell in subject_cells]) < 0:
        cb_label = "Probability of Scribble mitoses? AMEND THIS LABEL"
    if event == "APOPTOSIS":
        raise Exception("Apoptosis event counter not configured yet")

    plt.colorbar(label=cb_label)

    return plt.imshow(final_hist)


def kymo_labels(num_bins, label_freq, radius, t_range, SI):
    # label_freq = 1
    radial_bin = radius / num_bins
    temporal_bin = t_range / num_bins

    if SI == True:
        time_scale_factor = 4 / 60  ## each frame is 4 minutes
        distance_scale_factor = 1 / 3  ## each pixel is 0.3recur micrometers
    else:
        time_scale_factor, distance_scale_factor = 1, 1

    ### generate labels for axis micrometers/hours
    xlocs = np.arange(
        -0.5, num_bins, label_freq
    )  ## -0.5 to start at far left border of first bin
    xlabels = []
    for m in np.arange(int(-num_bins / 2), int(num_bins / 2) + 1, label_freq):
        xlabels.append(
            str(int(((temporal_bin) * m) * time_scale_factor))
        )  # + "," + str(int(((temporal_bin)*m+temporal_bin)*time_scale_factor)))

    ylocs = np.arange(
        -0.5, num_bins, label_freq
    )  ## -0.5 to start at far top border of first bin
    ylabels = []
    for m in np.arange(num_bins, 0 - 1, -label_freq):
        ylabels.append(
            str(int(((radial_bin) * m) * distance_scale_factor))
        )  # + "," + str(int(((radial_bin)*(m-1)*distance_scale_factor))))

    return xlocs, xlabels, ylocs, ylabels


def old_kymo_labels(num_bins, label_freq, radius, t_range, SI):
    """
    This plots the labels in the middle of each bin whereas the new one plots labels on the edges of each bin
    """
    label_freq = 1
    radial_bin = radius / num_bins
    temporal_bin = t_range / num_bins

    if SI == True:
        time_scale_factor = 4 / 60  ## each frame is 4 minutes
        distance_scale_factor = 1 / 3  ## each pixel is 0.3recur micrometers
    else:
        time_scale_factor, distance_scale_factor = 1, 1

    ### generate labels for axis micrometers/hours
    xlocs = range(0, num_bins, label_freq)  ## step of 2 to avoid crowding
    xlabels = []
    for m in range(int(-num_bins / 2), int(num_bins / 2), label_freq):
        xlabels.append(
            str(int(((temporal_bin) * m) * time_scale_factor))
            + ","
            + str(int(((temporal_bin) * m + temporal_bin) * time_scale_factor))
        )
    ylocs = range(0, num_bins, label_freq)  ## step of 2 to avoid crowding
    ylabels = []
    for m in range(num_bins, 0, -label_freq):
        ylabels.append(
            str(int(((radial_bin) * m) * distance_scale_factor))
            + ","
            + str(int((radial_bin) * (m - 1) * distance_scale_factor))
        )

    return xlocs, xlabels, ylocs, ylabels


"""
Raw data viewer rendering WIP below, needs work as doesnt recognise `viewer` atm

This module contains functions to assist in the rendering and display of data from my radial analysis of cell competition
"""


def find_apoptosis_time(
    target_track, index
):  ### if index is set to True then the index of the apoptotic time (wrt target_track) is returned
    """
    This takes a target track and finds the apoptosis time, returning it as an absolute time if index == False, or a time relative to cell life (index) if index == True
    """
    for i, j in enumerate(target_track.label):
        if (
            j == "APOPTOSIS"
            and target_track.label[i + 1] == "APOPTOSIS"
            and target_track.label[i + 2] == "APOPTOSIS"
        ):  # and target_track.label[i+3] =='APOPTOSIS' and target_track.label[i+4] =='APOPTOSIS':
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
    frame = find_apoptosis_time(target_track, index=False) + delta_t
    dividing_states = ("METAPHASE",)  # ('PROMETAPHASE', 'METAPHASE', 'DIVIDE')
    wt_tracks_in_radius = [
        wt_track
        for wt_track in wt_tracks
        if wt_track.in_frame(frame)
        if euclidean_distance(target_track, wt_track, frame) < radius
    ]
    wt_mitosis_in_radius = [
        wt_track
        for wt_track in wt_tracks
        if wt_track.in_frame(frame)
        if euclidean_distance(target_track, wt_track, frame) < radius
        if wt_track.label[wt_track.t.index(frame)] in dividing_states
        if wt_track.fate.name == "DIVIDE"
    ]  ###check this

    return wt_tracks_in_radius, wt_mitosis_in_radius


def plot_mitoses(
    cell_type, cell_ID, radius, delta_t
):  ## this function plots mitosis events into the napari viewer
    """
    This function takes a cell_type, a focal cell ID, a spatial radius and a time window and finds all the mitotic cells belonging to cell type within the radius and time window, plotting them as points in napari.
    """
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    apop_time, apop_index = (
        find_apoptosis_time(target_track, index=False),
        find_apoptosis_time(target_track, index=True),
    )
    apop_event = (
        target_track.t[apop_index],
        target_track.x[apop_index] + shift_y,
        target_track.y[apop_index] + shift_x,
    )  ## with transposed shift
    wt_tracks_in_radius, wt_mitosis_in_radius = find_nearby_wt_mitosis(
        target_track, delta_t, radius
    )
    t_m, x_m, y_m = (
        np.zeros(len(wt_mitosis_in_radius)),
        np.zeros(len(wt_mitosis_in_radius)),
        np.zeros(len(wt_mitosis_in_radius)),
    )
    mito_events = np.zeros(
        (len(wt_mitosis_in_radius), 3)
    )  ## 3 because of the 3 cartesian coords
    for i, wt_mitosis in enumerate(
        wt_mitosis_in_radius
    ):  ## this now assumes that the mitosis time point of relevance isnt the last frame of track but the time at delta_t, need to bolster definition of mitosis
        mito_index = [
            j for j, k in enumerate(wt_mitosis.t) if k == apop_event[0] + delta_t
        ][
            0
        ]  ### [0] bc first item of list comprehension
        t_m[i], x_m[i], y_m[i] = (
            wt_mitosis.t[mito_index],
            wt_mitosis.x[mito_index] + shift_y,
            wt_mitosis.y[mito_index] + shift_x,
        )  ## plus transposed coordinate shift
        mito_events[i] = t_m[i], x_m[i], y_m[i]
    return viewer.add_points(
        mito_events, name="Mitosis events", symbol="cross", face_color="pink"
    )


def plot_target_track(cell_type, cell_ID):
    """
    This takes a cell_type and target cell ID and plots it as a point in napari
    """
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    target_track_loc = [
        (target_track.t[i], target_track.x[i] + shift_y, target_track.y[i] + shift_x)
        for i in range(len(target_track.t))
    ]
    return viewer.add_points(
        target_track_loc,
        name="Track of interest",
        size=40,
        symbol="o",
        face_color="transparent",
        edge_color="cyan",
        edge_width=2,
    )


def plot_stationary_apoptosis_point(
    cell_type, cell_ID
):  ## this function plots apoptotic event and surrounding local environment scope (determined by radius)
    """
    This takes a cell type and cell ID and plots the apoptosis point at the time of apoptosis only
    """
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    apop_time, apop_index = (
        find_apoptosis_time(target_track, index=False),
        find_apoptosis_time(target_track, index=True),
    )
    apop_event = [
        (t, target_track.x[apop_index] + shift_y, target_track.y[apop_index] + shift_x)
        for t in range(len(gfp))
    ]  ## marker for apoptosis over all frames
    return viewer.add_points(
        apop_event,
        name="Stastionary apoptosis point",
        size=40,
        symbol="o",
        face_color="transparent",
        edge_color="cyan",
        edge_width=2,
    )


def plot_stationary_apop_radius(cell_type, cell_ID, radius, delta_t, inner_radius):
    """
    This takes a cell type and cell ID and plots the apoptosis with a radius and optional inner ring at the time specified as delta_t either side of apop time
    """
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    apop_time, apop_index = (
        find_apoptosis_time(target_track, index=False),
        find_apoptosis_time(target_track, index=True),
    )
    apop_event = (
        target_track.t[apop_index],
        target_track.x[apop_index] + shift_y,
        target_track.y[apop_index] + shift_x,
    )  ## with transposed shift, just for the frame of apoptosis
    outer_radial_bin = [
        tuple(
            (
                (apop_event[0] + t, apop_event[1] - radius, apop_event[2] - radius),
                (apop_event[0] + t, apop_event[1] + radius, apop_event[2] - radius),
                (apop_event[0] + t, apop_event[1] + radius, apop_event[2] + radius),
                (apop_event[0] + t, apop_event[1] - radius, apop_event[2] + radius),
            )
        )
        for t in range(-abs(delta_t), +abs(delta_t) + 1)
    ]
    if inner_radius > 0:
        inner_radial_bin = [
            tuple(
                (
                    (
                        apop_event[0] + t,
                        apop_event[1] - inner_radius,
                        apop_event[2] - inner_radius,
                    ),
                    (
                        apop_event[0] + t,
                        apop_event[1] + inner_radius,
                        apop_event[2] - inner_radius,
                    ),
                    (
                        apop_event[0] + t,
                        apop_event[1] + inner_radius,
                        apop_event[2] + inner_radius,
                    ),
                    (
                        apop_event[0] + t,
                        apop_event[1] - inner_radius,
                        apop_event[2] + inner_radius,
                    ),
                )
            )
            for t in range(-abs(delta_t), +abs(delta_t) + 1)
        ]
        return viewer.add_shapes(
            outer_radial_bin,
            opacity=1,
            shape_type="ellipse",
            face_color="transparent",
            edge_color="cyan",
            edge_width=5,
            name="Radial environment",
        ), viewer.add_shapes(
            inner_radial_bin,
            opacity=1,
            shape_type="ellipse",
            face_color="transparent",
            edge_color="cyan",
            edge_width=5,
            name="Inner Radial environment",
        )
    else:
        return viewer.add_shapes(
            outer_radial_bin,
            opacity=1,
            shape_type="ellipse",
            face_color="transparent",
            edge_color="cyan",
            edge_width=5,
            name="Radial environment",
        )


def plot_radius(cell_type, cell_ID, radius):
    """
    This takes a cell type and cell ID and plots a radius around that cell for the cells life time
    """
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [
        tuple(
            (
                (
                    t,
                    target_track.x[i] + shift_y - radius,
                    target_track.y[i] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y + radius,
                    target_track.y[i] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y + radius,
                    target_track.y[i] + shift_x + radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y - radius,
                    target_track.y[i] + shift_x + radius,
                ),
            )
        )
        for i, t in enumerate(range(target_track.t[0], target_track.t[-1]))
    ]
    return viewer.add_shapes(
        radius_shape,
        opacity=1,
        shape_type="ellipse",
        face_color="transparent",
        edge_color="cyan",
        edge_width=5,
        name="Radial environment",
    )


def plot_post_track_radius(cell_type, cell_ID, radius):
    """
    This takes a cell type and cell ID and plots a radius around that cell after that cell had died/disappeared
    """
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [
        tuple(
            (
                (
                    t,
                    target_track.x[-1] + shift_y - radius,
                    target_track.y[-1] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[-1] + shift_y + radius,
                    target_track.y[-1] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[-1] + shift_y + radius,
                    target_track.y[-1] + shift_x + radius,
                ),
                (
                    t,
                    target_track.x[-1] + shift_y - radius,
                    target_track.y[-1] + shift_x + radius,
                ),
            )
        )
        for i, t in enumerate(range(target_track.t[-1], len(gfp)))
    ]
    return viewer.add_shapes(
        radius_shape,
        opacity=1,
        shape_type="ellipse",
        face_color="transparent",
        edge_color="cyan",
        edge_width=5,
        name="Post-apoptosis radial environment",
    )


def plot_fragmented_track(
    list_of_IDs,
):  ### not using this below as dont think output is correct
    """
    This takes a list of cell IDs as a fragmented track and plots a radius around the location of each fragment
    """
    compiled_frag_track_loc = []
    compiled_frag_radius_loc = []
    for cell_ID in list_of_IDs:
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
        # plot_radius(target_track)
        # plot_target_track(target_track)
        radius_loc = plot_frag_radius(target_track)
        compiled_frag_radius_loc += radius_loc
        target_track_loc = plot_frag_target_track(target_track)
        compiled_frag_track_loc += target_track_loc
    return viewer.add_shapes(
        compiled_frag_radius_loc,
        opacity=1,
        shape_type="ellipse",
        face_color="transparent",
        edge_color="cyan",
        edge_width=5,
        name="Radial environment",
    ), viewer.add_points(
        compiled_frag_track_loc,
        name="Track of interest",
        size=40,
        symbol="o",
        face_color="transparent",
        edge_color="cyan",
        edge_width=2,
    )


def plot_frag_target_track(target_track):
    """
    This takes a fragmented track, currently modelled on example cell 17 and provides the location of the cell whilst it is existent and then provides an alternate fragmented track after
    """
    if target_track.ID == 17:
        target_track_loc = [
            (
                target_track.t[i],
                target_track.x[i] + shift_y,
                target_track.y[i] + shift_x,
            )
            for i in range(len(target_track.t))
        ]
        return target_track_loc  # viewer.add_points(target_track_loc, name = "Track of interest", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)
    else:
        target_track_loc = [
            (
                target_track.t[i],
                target_track.x[i] + shift_y,
                target_track.y[i] + shift_x,
            )
            for i in range(len(target_track.t))
            if target_track.t[i] > 742
        ]
        return target_track_loc  # viewer.add_points(target_track_loc, name = "Track of interest", size = 40, symbol = 'o', face_color = "transparent", edge_color = 'cyan', edge_width = 2)


def plot_frag_radius(target_track):
    """
    This takes a fragmented track, currently modelled on example cell 17 and provides the location of the cellradius whilst it is existent and then provides an alternate fragmented track after
    """
    if (
        target_track.ID == 17
    ):  ### this if condition is to avoid double plotting radii as fragmented tracks exist at same time
        radius_shape = [
            tuple(
                (
                    (
                        t,
                        target_track.x[i] + shift_y - radius,
                        target_track.y[i] + shift_x - radius,
                    ),
                    (
                        t,
                        target_track.x[i] + shift_y + radius,
                        target_track.y[i] + shift_x - radius,
                    ),
                    (
                        t,
                        target_track.x[i] + shift_y + radius,
                        target_track.y[i] + shift_x + radius,
                    ),
                    (
                        t,
                        target_track.x[i] + shift_y - radius,
                        target_track.y[i] + shift_x + radius,
                    ),
                )
            )
            for i, t in enumerate(range(target_track.t[0], target_track.t[-1]))
        ]
        return radius_shape
    else:
        radius_shape = [
            tuple(
                (
                    (
                        t,
                        target_track.x[i] + shift_y - radius,
                        target_track.y[i] + shift_x - radius,
                    ),
                    (
                        t,
                        target_track.x[i] + shift_y + radius,
                        target_track.y[i] + shift_x - radius,
                    ),
                    (
                        t,
                        target_track.x[i] + shift_y + radius,
                        target_track.y[i] + shift_x + radius,
                    ),
                    (
                        t,
                        target_track.x[i] + shift_y - radius,
                        target_track.y[i] + shift_x + radius,
                    ),
                )
            )
            for i, t in enumerate(range(target_track.t[0], target_track.t[-1]))
            if t > 741
        ]
        return radius_shape


def plot_radii(cell_type, target_track, radius, num_bins):
    """
    This takes a cell type, target track, radius and number of bins and plots the radius/number of bins as concentric circles following the target track
    """
    print(
        "This can be very time consuming for >10 bins, consider using single_frame radius"
    )
    radii = range(
        int(radius / num_bins), radius + int(radius / num_bins), int(radius / num_bins)
    )
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [
        tuple(
            (
                (
                    t,
                    target_track.x[i] + shift_y - radius,
                    target_track.y[i] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y + radius,
                    target_track.y[i] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y + radius,
                    target_track.y[i] + shift_x + radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y - radius,
                    target_track.y[i] + shift_x + radius,
                ),
            )
        )
        for i, t in enumerate(range(target_track.t[0], target_track.t[-1]))
        for radius in radii
    ]
    # return radius_shape
    return viewer.add_shapes(
        radius_shape,
        opacity=1,
        shape_type="ellipse",
        face_color="transparent",
        edge_color="cyan",
        edge_width=5,
        name="Radial environment",
    )


def plot_stationary_radii(cell_type, target_track, radius, num_bins):
    """
    This takes a cell type, target track, radius and number of bins and plots the radius/number of bins as concentric circles stationary after the target track ceases to exist
    """
    print(
        "This can be very time consuming for >10 bins, consider using single_frame radius"
    )
    radii = range(
        int(radius / num_bins), radius + int(radius / num_bins), int(radius / num_bins)
    )
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]
    radius_shape = [
        tuple(
            (
                (
                    t,
                    target_track.x[-1] + shift_y - radius,
                    target_track.y[-1] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[-1] + shift_y + radius,
                    target_track.y[-1] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[-1] + shift_y + radius,
                    target_track.y[-1] + shift_x + radius,
                ),
                (
                    t,
                    target_track.x[-1] + shift_y - radius,
                    target_track.y[-1] + shift_x + radius,
                ),
            )
        )
        for i, t in enumerate(range(target_track.t[-1] + 1, len(gfp)))
        for radius in radii
    ]
    # return radius_shape
    return viewer.add_shapes(
        radius_shape,
        opacity=1,
        shape_type="ellipse",
        face_color="transparent",
        edge_color="cyan",
        edge_width=5,
        name="Radial environment",
    )


def plot_single_frame_radii(cell_type, target_track, radius, num_bins, frame):
    """
    This takes a cell type, target track, radius, number of bins and a frame and plots the radius/number of bins as concentric circles at that frame time point only
    """
    cell_ID = target_track
    t = frame
    if cell_type.lower() == "scr":
        target_track = [track for track in scr_tracks if track.ID == cell_ID][0]
    else:
        target_track = [track for track in wt_tracks if track.ID == cell_ID][0]

    try:
        i = target_track.t.index(t)
    except:
        i = -1
    radii = range(
        int(radius / num_bins), radius + int(radius / num_bins), int(radius / num_bins)
    )
    radius_shape = [
        tuple(
            (
                (
                    t,
                    target_track.x[i] + shift_y - radius,
                    target_track.y[i] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y + radius,
                    target_track.y[i] + shift_x - radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y + radius,
                    target_track.y[i] + shift_x + radius,
                ),
                (
                    t,
                    target_track.x[i] + shift_y - radius,
                    target_track.y[i] + shift_x + radius,
                ),
            )
        )
        for radius in radii
    ]
    # return radius_shape
    return viewer.add_shapes(
        radius_shape,
        opacity=1,
        shape_type="ellipse",
        face_color="transparent",
        edge_color="cyan",
        edge_width=5,
        name="Radial environment",
    )
