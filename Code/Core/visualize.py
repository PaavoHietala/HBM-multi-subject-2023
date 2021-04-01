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
import numpy as np
from matplotlib.colors import to_rgba
from surfer import utils
from .utils import get_fname

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
    
def expand_peak_labels(subjects, project_dir, src_spacing, stc_method, task,
                       stimuli, colors, overwrite):
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
    
    peaks = []
    peak_hemis = []
    colors_rgb = np.array([to_rgba(c) for c in colors])
    
    
    # Find peaks for each averaged stimulus stc
    for stim in stimuli:
        fname = get_fname('fsaverage', 'stc', stc_method = stc_method, 
                          src_spacing = src_spacing, task = task, stim = stim)
        fpath = os.path.join(project_dir, 'Data', 'avg', fname)
        stc = mne.read_source_estimate(fpath)
        
        if np.max(stc.lh_data) > np.max(stc.rh_data):
            hemi = 'lh'
        else:
            hemi = 'rh'
        
        peaks.append(stc.get_peak(hemi = hemi)[0])
        peak_hemis.append(hemi)
    
    # Remove duplicate peaks, because they are indistinguishable
    for hemi in ['lh', 'rh']:
        i = 0
        while True:
            if i == len(peaks) - 1:
                break
            elif peak_hemis[i] != hemi:
                i += 1
                continue
            elif peaks.count(peaks[i]) == 1:
                i += 1
                continue
            
            # Iterate over indices of duplicate values backwards, skip first occurance
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
        labels_ =  mne.grow_labels('fsaverage', peaks_, [7] * len(peaks_), [hemi_id] * len(peaks_),
                                   overlap = False, names = stimuli_, colors = colors_)

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
