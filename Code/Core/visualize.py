#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=unexpected-keyword-arg
'''
Common plotting, labeling and visualization functions used in multiple pipelines.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import os
import mne
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from surfer import utils
from .utils import get_fname, find_peaks

def label_peaks(subjects, project_dir, src_spacing, stc_method, task, stimuli,
                colors, overwrite):
    '''
    Get lh stc peaks and create labels from them and save the label to
    <project_dir>/Data/labels. Use expand_peak_labels instead.

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    task: str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    colors: list of str
        List of matplotlib colors for stimuli in the same order as stimuli.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    brain = mne.viz.Brain('fsaverage', 'lh', 'white')
    
    for c_idx, stim in enumerate(stimuli):
        lname = get_fname('fsaverage', 'label', src_spacing = src_spacing,
                          stc_method = stc_method, task = task, stim = stim)
        lpath = os.path.join(project_dir, 'Data', 'labels', lname)
        
        if overwrite or not os.path.isfile(lpath + '-lh.label'):
            fname = get_fname('fsaverage', 'stc', stc_method = stc_method, 
                              src_spacing = src_spacing, task = task, stim = stim)
            fpath = os.path.join(project_dir, 'Data', 'avg', fname)
            stc = mne.read_source_estimate(fpath)
            
            peak = stc.get_peak(hemi = 'lh')[0]
            
            print('Writing label', lpath)
            
            utils.coord_to_label('fsaverage', peak, label = lpath,
                                 hemi = 'lh', n_steps = 5, coord_as_vert = True)
        print('Loading label', lpath)
        brain.add_label(lpath + '-lh.label', color = colors[c_idx], reset_camera = False)

def plot_foci(project_dir, src_spacing, stc_method, task, stimuli, colors,
              bilaterals, suffix, mode, overwrite, subject = 'fsaverage',
              stc_type = 'avg'):
    '''
    Plot foci bubbles on fsaverage brain. Eccentricity and angle can be controlled
    with colors parameter

    Parameters
    ----------
    project_dir : str
        Base directory of the project with Code and Data subfolders.
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
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.
    subject : str
        Name of the subject the stc is for, defaults to fsaverage
    stc_type : str
        Type of stc, can be either stc, stc_m or avg. Defaults to avg

    Returns
    -------
    None.
    '''
    
    title = ' '.join([stc_method, suffix, 'average', mode, subject, stc_type])
    brain = mne.viz.Brain((subject if mode == 'stc' else 'fsaverage'),
                          'split', 'inflated', title = title,
                          background = (255, 255, 255), size = (1000, 600),
                          show = False)

    # Duplicate color values and stimulus names for bilateral stimuli, get peaks
    bilateral_idx = [stimuli.index(stim) for stim in bilaterals]
    [colors.insert(i, colors[i]) for i in bilateral_idx[::-1]]
    colors_rgb = np.array([to_rgba(c) for c in colors])

    peaks, peak_hemis = find_peaks(project_dir, src_spacing, stc_method, task,
                                   stimuli, bilaterals, suffix, subject = subject,
                                   mode = stc_type)
    
    stimuli = copy.deepcopy(stimuli)
    [stimuli.insert(i, stimuli[i]) for i in bilateral_idx[::-1]]
    
    # Add V1 label and found peaks on the brain & plot
    for hemi_id, hemi in enumerate(['lh', 'rh']):
        # Label V1 on the cortex
        v1 = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + (subject if mode == 'stc' else 'fsaverage')
                            + '/label/' + hemi + '.V1_exvivo.label',
                            (subject if mode == 'stc' else 'fsaverage'))
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

    # Cropping code from https://mne.tools/stable/auto_examples/visualization/publication_figure.html
    ss = brain.screenshot()
    nonwhite_pix = (ss != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped = ss[nonwhite_row][:, nonwhite_col]

    plt.imsave(project_dir + 'Data/plot/' + title + '.png', cropped)

    print('Done')

#
### Deprecated functions below--------------------------------------------------
#

def expand_peak_labels(subjects, project_dir, src_spacing, stc_method, task,
                       stimuli, colors, suffix, overwrite, bilaterals = []):
    '''
    Create labels for peaks activation locations for each hemisphere and
    visualize the results.

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    task: str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    colors: list of str
        List of matplotlib colors for stimuli in the same order as stimuli.
    suffix : str
        Suffix to append to stc filename before -avg.fif
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.
    bilaterals: list of str
        List of bilateral stimuli (on midline); label peaks on both hemis.
        Default is an empty list [].

    Returns
    -------
    None.
    '''
    
    brain_lh = mne.viz.Brain('fsaverage', 'lh', 'inflated', title = 'fsaverage lh',
                             show = False)
    brain_rh = mne.viz.Brain('fsaverage', 'rh', 'inflated', title = 'fsaverage rh',
                             show = False)

    # Label V1 on the cortex
    v1_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                           + 'fsaverage' + '/label/lh.V1_exvivo.label', 'fsaverage')
    brain_lh.add_label(v1_lh, borders = 2)
    v1_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                           + 'fsaverage' + '/label/rh.V1_exvivo.label', 'fsaverage')
    brain_rh.add_label(v1_rh, borders = 2)

    # Duplicate color values for bilateral stimuli
    bilateral_idx = [stimuli.index(stim) for stim in bilaterals]
    [colors.insert(i, colors[i]) for i in bilateral_idx[::-1]]
    colors_rgb = np.array([to_rgba(c) for c in colors])

    peaks, peak_hemis = find_peaks(project_dir, src_spacing, stc_method, task,
                                   stimuli, bilaterals, suffix)
    
    # Update stimulus list to accomodate for bilateral peaks so the lists can
    # be index-matched

    stimuli = copy.deepcopy(stimuli)
    [stimuli.insert(i, stimuli[i]) for i in bilateral_idx[::-1]]
    
    # Remove duplicate peaks, because they are indistinguishable
    for hemi in ['lh', 'rh']:
        i = 0
        while True:
            if i == len(peaks) - 1:
                break
            elif peak_hemis[i] != hemi or peaks.count(peaks[i]) == 1:
                i += 1
                continue
            
            # Iterate over indices of duplicate values backwards, while removing
            # duplicate peaks from all lists, skip first occurance to leave one.
            for dup_idx in [j for j, peak in enumerate(peaks) if peak == peaks[i]][:0:-1]:
                print(stimuli[i] + ' and ' + stimuli[dup_idx]
                      + ' have the same seed on hemi ' + peak_hemis[dup_idx])
                peaks.pop(dup_idx)
                peak_hemis.pop(dup_idx)
                colors_rgb = np.delete(colors_rgb, dup_idx, 0)
                stimuli.pop(dup_idx)
            i += 1
    
    # Grow labels and visualize them on the brain
    for hemi_id, hemi in enumerate(['lh', 'rh']):
        colors_ = np.zeros((peak_hemis.count(hemi), 4))
        peaks_ = []
        stimuli_ = []
        for idx, hemi_idx in enumerate([j for j, ph in enumerate(peak_hemis) if ph == hemi]):
            colors_[idx] = colors_rgb[hemi_idx]
            peaks_.append(peaks[hemi_idx])
            stimuli_.append(stimuli[hemi_idx])
        
        print('Growing ' + hemi + ' labels...')
        labels_ =  mne.grow_labels('fsaverage', peaks_, [3] * len(peaks_), [hemi_id] * len(peaks_),
                                   overlap = True, names = stimuli_, colors = colors_)

        print('Adding labels to ' + hemi + ' brain...')
        for label in labels_:
            if hemi == 'lh':
                brain_lh.add_label(label, alpha = 1, reset_camera = False)
            else:
                brain_rh.add_label(label, alpha = 1, reset_camera = False)

    print('Done')
    brain_lh.show()
    brain_lh.show_view({'elevation' : 100, 'azimuth' : -55}, distance = 350)
    brain_rh.show()
    brain_rh.show_view({'elevation' : 100, 'azimuth' : -125}, distance = 350)

def label_all_vertices(subjects, project_dir, src_spacing, stc_method, task,
                       stimuli, colors, overwrite):
    '''
    Create labels for every point based on which normalized response is the 
    most prominant

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    task: str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    colors: list of str
        List of matplotlib colors for stimuli in the same order as stimuli.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    brain_lh = mne.viz.Brain('fsaverage', 'lh', 'inflated', title = 'fsaverage lh',
                             show = False)
    brain_rh = mne.viz.Brain('fsaverage', 'rh', 'inflated', title = 'fsaverage rh',
                             show = False)
    
    data = [[], []]
    colors_rgb = np.array([to_rgba(c) for c in colors])
    
    # Load all stcs, limit to occipital cortex and normalize their values to 0..1
    occipital_idx_lh = []
    annotation_lh = mne.read_labels_from_annot('fsaverage', hemi = 'lh')
    for label in [l for l in annotation_lh if l.name in ['lingual-lh', 'lateraloccipital-lh', 'cuneus-lh', 'pericalcarine-lh']]:
        occipital_idx_lh += label.get_vertices_used().tolist()
    
    occipital_idx_rh = []
    annotation_rh = mne.read_labels_from_annot('fsaverage', hemi = 'rh')
    for label in [l for l in annotation_rh if l.name in ['lingual-rh', 'lateraloccipital-rh', 'cuneus-rh', 'pericalcarine-rh']]:
        occipital_idx_rh += label.get_vertices_used().tolist()
        
    print('Loading stcs')
    for stim in stimuli:
        fname = get_fname('fsaverage', 'stc', stc_method = stc_method, 
                          src_spacing = src_spacing, task = task, stim = stim)
        fpath = os.path.join(project_dir, 'Data', 'avg', fname)
        stc = mne.read_source_estimate(fpath)
        
        data_lh = stc.lh_data.copy()
        #for col_idx in range(stc.lh_data.shape[0]):
        #    if col_idx in occipital_idx_lh:
        #        continue
        #    else:
        #        data_lh[col_idx, :] = 0
        
        data_rh = stc.rh_data.copy()
        #for col_idx in range(stc.rh_data.shape[0]):
        #    if col_idx in occipital_idx_rh:
        #        continue
        #    else:
        #        data_rh[col_idx, :] = 0
        
        data[0].append(data_lh / data_lh.sum())
        data[1].append(data_rh / data_rh.sum())
    
    # Iterate over all vertices and label them
    print('Solving labels for vertices')
    vertex_labels = [[], []]
    for hemi_id in [0, 1]:
        for vertex in range(len(data[hemi_id][0])):
            stcs = [max(stc[vertex]) for stc in data[hemi_id]]
            label = stcs.index(max(stcs))
            vertex_labels[hemi_id].append(label)
    
    # Create labels from vertex indices
    print('Creating labels')
    for hemi_id, hemi in enumerate(['lh', 'rh']):
        labels = []
        
        for label_id in range(len(stimuli)):
            verts = [i for i, x in enumerate(vertex_labels[hemi_id]) if x == label_id]
            labels.append(mne.Label(verts, hemi = hemi, color = colors_rgb[label_id]))

        print('Adding labels to ' + hemi + ' brain...')
        for label in labels:
            if hemi == 'lh':
                brain_lh.add_label(label.smooth(subject = 'fsaverage', smooth = 2), alpha = 1, reset_camera = False)
            else:
                brain_rh.add_label(label.smooth(subject = 'fsaverage', smooth = 2), alpha = 1, reset_camera = False)

    print('Done')
    brain_lh.show()
    brain_lh.show_view({'elevation' : 100, 'azimuth' : -55}, distance = 350)
    brain_rh.show()
    brain_rh.show_view({'elevation' : 100, 'azimuth' : -125}, distance = 350)
