#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=unexpected-keyword-arg
'''
Draw and save the eccentricity and polar angle foci plots in Figures 9-11.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import copy
import os.path
import sys

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Core.utils import crop_whitespace, find_peaks

# Script settings --------------------------------------------------------------

# Root data directory for each source estimate, list of str

project_dirs = ['/m/nbe/scratch/megci/MFinverse/Classic/'] * 3 \
               + ['/m/nbe/scratch/megci/MFinverse/reMTW/'] * 6

# Subjects' MRI location (FreeSurfer dir), str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'

# List of subjects, indices within all following lists must match

subjects = ['fsaverage'] * 3 + ['MEGCI_S1'] * 3 + ['fsaverage'] * 3

# List of stimuli

stimuli = [f'sector{num}' for num in range(1, 25)]

# Source point spacing for source space calculation, str

src_spacing = 'ico4'

# Which task is currently investigated. Used as a suffix in file names,
# can also be 'None' if task suffix is not desired. str

task = 'f'

# List of inverse solution methods used for each stc

methods = ['eLORETA'] * 3 + ['remtw'] * 6

# Suffixes of stcs, can be None for individual stcs

suffixes = ['1subjects', '10subjects', '20subjects'] * 3

# Types of each stc. 'stc' (individual), 'stc_m' (individual on fsaverage mesh)
# or 'avg' (averaged data on fsaverage mesh)

stc_types = ['avg'] * 3 + ['stc_m'] * 3 + ['avg'] * 3
            
# Target source space of the morphing in stc_m and avg, usually 'fsaverage'

src_tos = [None] * 3 + ['fsaverage'] * 3 + [None] * 3

# List of matplotlib colors for each stimulus area, colors_ecc are used in
# eccentricity-based plots and colors_polar in polar angle -based plots,
# list of str

colors_ecc = ['blue'] * 8 + ['yellow'] * 8 + ['red'] * 8
colors_polar = ['cyan', 'indigo', 'violet', 'magenta', 'red', 'orange',
                'yellow', 'green'] * 3

# List of stimuli that should show response on both hemispheres, list of str

bilaterals = [f'sector{num}' for num in [3, 7, 11, 15, 19, 23]]

# Overwrite existing files, bool

overwrite = True

# Run the script for all data --------------------------------------------------

mne.set_config('SUBJECTS_DIR', subjects_dir)

def plot_foci(project_dir, src_spacing, stc_method, task, stimuli, colors,
              bilaterals, suffix, mode, subject = 'fsaverage',
              stc_type = 'avg', time = None, overwrite = 'False'):
    '''
    Plot peak activation location bubbles on fsaverage brain.

    Parameters
    ----------
    project_dir : str
        Base directory of the project with Data subfolder.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct4'.
    stc_method : str
        Inversion method used, e.g. 'eLORETA'.
    task : str
        Task in the estimated stcs, e.g. 'f'.
    stimuli : list of str
        List of stimuli for whcih the stcs are estimated.
    colors : list of str
        List of matplotlib colors for stimuli in the same order as stimuli.
    bilaterals: list of str
        List of bilateral stimuli (on midline); label peaks on both hemis.
    suffix : str
        Suffix to append to stc filename before the common suffix and extension
        (e.g. '-avg.fif').
    mode : str
        Either 'polar' or 'ecc', affects only titles and image names. 
    subject : str, optional
        Name of the subject the stc is for, by default 'fsaverage'.
    stc_type : str, optional
        Type of stc, can be either stc, stc_m or avg, by default 'avg'.
    time : float, optional
        A timepoint for which to get the peak. If None, overall peak is used.
        By default None.
    overwrite : bool, optional
        Overwrite existing figuresby default False.

    Returns
    -------
    None.
    '''
    
    plot_subject = subject if stc_type == 'stc' else 'fsaverage'
    title = ' '.join([stc_method, suffix,
                      'average', mode, subject, stc_type])
    fpath_out = os.path.join(project_dir, 'Data', 'plot', f'{title}.png')

    if not overwrite and os.path.isfile(fpath_out):
        print(f"Output file {fpath_out} exists and overwrite = False, skipping.")
        return

    brain = mne.viz.Brain(plot_subject, 'split', 'inflated', title = title,
                          background = (255, 255, 255), size = (1000, 600),
                          show = False)

    # Duplicate color values and stimulus names for bilateral stimuli
    bilateral_idx = [stimuli.index(stim) for stim in bilaterals]
    colors = copy.deepcopy(colors)
    [colors.insert(i, colors[i]) for i in bilateral_idx[::-1]]
    colors_rgb = np.array([to_rgba(c) for c in colors])

    peaks, peak_hemis = find_peaks(project_dir, src_spacing, stc_method, task,
                                   stimuli, bilaterals, suffix, time = time,
                                   subject = subject, mode = stc_type)
    
    stimuli = copy.deepcopy(stimuli)
    [stimuli.insert(i, stimuli[i]) for i in bilateral_idx[::-1]]
    
    # Add V1 label and found peaks on the brain & plot
    for hemi in ['lh', 'rh']:
        # Label V1 on the cortex
        label_path = os.path.join(mne.get_config('SUBJECTS_DIR'), plot_subject,
                                  'label', f'{hemi}.V1_exvivo.label')
        v1 = mne.read_label(label_path, plot_subject)
        brain.add_label(v1, borders = 2)

        # Prepare hemisphere-specific subset lists of colors, peaks and
        # stimulus & color names for the plot.
        colors_ = np.zeros((peak_hemis.count(hemi), 4))
        peaks_ = []
        stimuli_ = []
        c_names = []
        for idx, hemi_idx in enumerate([j for j, ph in enumerate(peak_hemis)
                                        if ph == hemi]):
            colors_[idx] = colors_rgb[hemi_idx]
            peaks_.append(peaks[hemi_idx])
            stimuli_.append(stimuli[hemi_idx])
            c_names.append(colors[hemi_idx])
        
        # Add foci to brain, scale overlapping bubbles sequentially
        print('Adding ' + hemi + ' foci...')
        used_verts = []
        for idx in range(len(colors_)):
            print("Adding " + stimuli_[idx] + " color " + c_names[idx])

            n = used_verts.count(peaks_[idx])
            brain.add_foci(peaks_[idx], coords_as_verts = True,
                           scale_factor = 0.5 + n * 0.25,
                           color = colors_[idx], alpha = 0.75,
                           name = stimuli_[idx], hemi = hemi)
            used_verts.append(peaks_[idx])
    
    brain.show_view({'elevation' : 100, 'azimuth' : -60}, distance = 350, col = 0)
    brain.show_view({'elevation' : 100, 'azimuth' : -120}, distance = 350, col = 1)
    brain.show()
    
    ss = brain.screenshot()
    cropped = crop_whitespace(ss)

    plt.imsave(fpath_out, cropped)

for i in range(len(subjects)):
    # Plot all stimulus peaks on fsaverage, color based on 3-ring eccentricity
    plot_foci(project_dirs[i], src_spacing, methods[i], task, stimuli,
              copy.deepcopy(colors_ecc), bilaterals, suffixes[i], 'ecc',
              overwrite = overwrite, subject = subjects[i],
              stc_type = stc_types[i])
    # Plot all stimulus peaks on fsaverage, color based on wedge
    plot_foci(project_dirs[i], src_spacing, methods[i], task, stimuli,
              copy.deepcopy(colors_polar), bilaterals, suffixes[i], 'polar',
              overwrite = overwrite, subject = subjects[i],
              stc_type = stc_types[i])
