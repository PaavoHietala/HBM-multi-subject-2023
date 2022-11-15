#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=unexpected-keyword-arg
'''
Common plotting, labeling and visualization functions used in multiple pipelines.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''
import copy
import os

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from .utils import crop_whitespace, find_peaks

def plot_foci(project_dir, src_spacing, stc_method, task, stimuli, colors,
              bilaterals, suffix, mode, subject = 'fsaverage', stc_type = 'avg',
              time = None):
    '''
    Plot foci bubbles on fsaverage brain. Eccentricity and angle can be
    controlled with colors parameter

    Parameters
    ----------
    project_dir : str
        Base directory of the project with Data subfolder.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    task : str
        Task in the estimated stcs, e.g. 'f'.
    stimuli : list of str
        List of stimuli for whcih the stcs are estimated.
    colors : list of str
        List of matplotlib colors for stimuli in the same order as stimuli.
    bilaterals: list of str
        List of bilateral stimuli (on midline); label peaks on both hemis.
    suffix : str
        Suffix to append to stc filename before -avg.fif
    mode : str
        Either 'polar' or 'ecc', affects only titles and image names 
    subject : str
        Name of the subject the stc is for, defaults to fsaverage
    stc_type : str
        Type of stc, can be either stc, stc_m or avg. Defaults to avg
    time : float
        A timepoint for which to get the peak. If None, overall peak is used.

    Returns
    -------
    None.
    '''
    
    plot_subject = subject if stc_type == 'stc' else 'fsaverage'
    title = ' '.join([stc_method, suffix, 'average', mode, subject, stc_type])
    brain = mne.viz.Brain(plot_subject, 'split', 'inflated', title = title,
                          background = (255, 255, 255), size = (1000, 600),
                          show = False)

    # Duplicate color values and stimulus names for bilateral stimuli, get peaks
    bilateral_idx = [stimuli.index(stim) for stim in bilaterals]
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

        # Prepare lists of colors, peaks, stimulus names and color names for plot
        colors_ = np.zeros((peak_hemis.count(hemi), 4))
        peaks_ = []
        stimuli_ = []
        c_names = []
        for idx, hemi_idx in enumerate([j for j, ph in enumerate(peak_hemis) if ph == hemi]):
            colors_[idx] = colors_rgb[hemi_idx]
            peaks_.append(peaks[hemi_idx])
            stimuli_.append(stimuli[hemi_idx])
            c_names.append(colors[hemi_idx])
        
        # Add foci to brain
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

    plt.imsave(os.path.join(project_dir, 'Data', 'plot', f'{title}.png'), cropped)
