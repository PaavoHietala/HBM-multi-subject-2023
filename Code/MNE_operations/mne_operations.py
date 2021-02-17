#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data analysis functions used for the 'classic' MNE analysis.

Created on Tue Feb  2 15:31:35 2021

@author: hietalp2
"""

import mne
import os
from surfer import utils
import matplotlib
import numpy as np
from sklearn.preprocessing import normalize

def get_fname(subject, ftype, stc_method = None, src_spacing = None,
              fname_raw = None, task = None, stim = None, layers = None, hemi = None):
    '''
    Create a project-standard filename from given parameters.
    
    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    ftype : str
        File type for which a fname is generate, e.g. 'src', 'fwd', 'cov'.
    stc_method : str, optional
        Inversion method used in this file, e.g. 'dSPM'. The default is None.
    src_spacing : str, optional
        Source space scheme used in this file, e.g. 'oct6'. The default is None.
    fname_raw : str, optional
        File name of raw recording from which the info is extracted. The default is None.
    task : str, optional
        Task in the evoked response, e.g. 'f'. The default is None.
    stim: str, optional
        Stimulus name, e.g. 'sector1'. The default is None.
    layers: str, optional
        How many layers the BEM model has. The default is None.
    hemi: str, optional
        Which hemisphere the name is for, e.g. 'lh'. THe default is None.

    Raises
    ------
    ValueError
        Error given if invalid ftype is given.

    Returns
    -------
    str
        File name built from given parameters.
    '''
    
    if ftype == 'src':
        return '-'.join([subject, src_spacing, 'src.fif'])
    elif ftype == 'fwd':
        return '-'.join([subject, src_spacing, 'fwd.fif'])
    elif ftype == 'cov':
        run = os.path.split(fname_raw)[-1].split('_')[0]
        return '-'.join([subject, run, 'cov.fif'])
    elif ftype == 'inv':
        run = os.path.split(fname_raw)[-1].split('_')[0]
        return '-'.join([subject, src_spacing, run, 'inv.fif'])
    elif ftype == 'stc':
        return '-'.join([subject, src_spacing, stc_method, task, stim])
    elif ftype == 'stc_m':
        return '-'.join([subject, src_spacing, stc_method, 'fsaverage', task, stim])
    elif ftype == 'bem':
        return '-'.join([subject, layers, 'shell-bem-sol.fif'])
    elif ftype == 'label':
        return '-'.join([subject, src_spacing, stc_method, task, stim])
    else:
        raise ValueError('Invalid file type ' + ftype)

def compute_source_space(subject, project_dir, src_spacing, overwrite = False,
                         add_dist = False, morph = False):
    '''
    Compute source space vertices from freesurfer data and save it in
    <project_dir>/Data/src/

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.
    add_dist: bool | str, optional
        Add distance information to the source space. The default is False.
    morph: bool, optional
        Create the source space by warping from fsaverage. The default is False.

    Returns
    -------
    None.
    '''
    
    fname = get_fname(subject, 'src', src_spacing = src_spacing)
    fpath = os.path.join(project_dir, 'Data', 'src', fname)
    
    if overwrite or not os.path.isfile(fpath):
        if not morph:
            src = mne.setup_source_space(subject, spacing = src_spacing,
                                         add_dist=add_dist)
        else:
            # Load fsaverage source space and morph it to subject
            fname_ref = get_fname('fsaverage', 'src', src_spacing = src_spacing)
            fpath_ref = os.path.join(project_dir, 'Data', 'src', fname_ref)
            
            src_ref = mne.read_source_spaces(fpath_ref)
            src = mne.morph_source_spaces(src_ref, subject_to=subject)
        src.save(fpath, overwrite = True)
   
def calculate_bem_solution(subject, project_dir, overwrite):
    '''
    Calculate bem solutions from FreeSurfer surfaces.

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.

    '''
    
    cond = {1 : [0.3], 3 : [0.3, 0.006, 0.3]}
    for layers in cond:
        fname = get_fname(subject, 'bem', layers = str(layers))
        fpath = os.path.join(mne.get_config('SUBJECTS_DIR'), subject, 'bem', fname)
        
        if overwrite or not os.path.isfile(fpath):
            # Calculate and save the BEM solution
            bem = mne.make_bem_model(subject, ico = None, conductivity = cond[layers])
            bem = mne.make_bem_solution(bem)
            mne.write_bem_solution(fpath, bem, overwrite = True)
    
def calculate_forward_solution(subject, project_dir, src_spacing, bem, raw, coreg,
                               overwrite = False):
    '''
    Calculate MEG forward solution with given parameters and save it in
    <project_dir>/Data/fwd/

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    bem : str
        Full path to the BEM file to be used here.
    raw : str
        Full path to the raw recording used here for sensor info.
    coreg : str
        Full path to the MEG/MRI coregistration file used here.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    fname_fwd = get_fname(subject, 'fwd', src_spacing = src_spacing)
    fpath_fwd = os.path.join(project_dir, 'Data', 'fwd', fname_fwd)
    
    if overwrite or not os.path.isfile(fpath_fwd):
        fname_src = get_fname(subject, 'src', src_spacing = src_spacing)
        src = mne.read_source_spaces(os.path.join(project_dir, 'Data', 'src', fname_src))
        
        fwd = mne.make_forward_solution(raw, trans = coreg, src = src, bem = bem,
                                        meg = True, eeg = False)
        mne.write_forward_solution(fpath_fwd, fwd, overwrite=True)
        print(fwd)
        print(type(fwd))

def compute_covariance_matrix(subject, project_dir, raw, overwrite = False):
    '''
    Compute covariance matrix from given rest recording and save it in
    <project_dir>/Data/cov/.

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    raw : str
        Full path to the raw recording used here for sensor info.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    fname = get_fname(subject,'cov', fname_raw = raw)
    fpath = os.path.join(project_dir, 'Data', 'cov', fname)
    
    if overwrite or not os.path.isfile(fpath):
        noise_cov = mne.compute_raw_covariance(mne.io.Raw(raw))    
        noise_cov.save(fpath)
    
def construct_inverse_operator(subject, project_dir, raw, src_spacing, overwrite = False):
    '''
    Construct inverse operator from forward solution and save it in
    <project_dir>/Data/inv/

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    raw : str
        Full path to the raw recording used here for sensor info.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    fname_inv = get_fname(subject, 'inv', src_spacing = src_spacing, fname_raw = raw)
    fpath_inv = os.path.join(project_dir, 'Data', 'inv', fname_inv)
    
    if overwrite or not os.path.isfile(fpath_inv):
        # Load raw data, noise covariance and forward solution
        info = mne.io.Raw(raw).info
    
        fname_cov = get_fname(subject, 'cov', fname_raw = raw)
        noise_cov = mne.read_cov(os.path.join(project_dir, 'Data', 'cov', fname_cov))
        
        fname_fwd = get_fname(subject, 'fwd', src_spacing = src_spacing)
        fwd = mne.read_forward_solution(os.path.join(project_dir, 'Data', 'fwd', fname_fwd))
        
        # Constrain the dipoles to surface normals
        mne.convert_forward_solution(fwd, surf_ori = True, use_cps = True, copy = False)
        
        # Calculate and save the inverse operator and save it to disk
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose = 0.2,
                                                     depth = 0.8)
        mne.minimum_norm.write_inverse_operator(fpath_inv, inv)
    
def estimate_source_timecourse(subject, project_dir, raw, src_spacing, stc_method,
                               fname_evokeds, task, stimuli, overwrite = False):
    '''
    Estimate surface time courses for all evoked responses in the evoked file
    and save them individually in <project_dir>/Data/stc/

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Code and Data subfolders.
    raw : str
        Full path to the raw recording used here for sensor info.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    fname_evokeds : str
        Full path to the file containing the evoked responses.
    task: str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    SNR = 2
    lambda2 = 1.0 / SNR ** 2
    
    # Load the inverse operator and evoked responses
    fname_inv = get_fname(subject, 'inv', src_spacing = src_spacing, fname_raw = raw)
    inv = mne.minimum_norm.read_inverse_operator(os.path.join(project_dir, 'Data', 'inv', fname_inv))
    
    evokeds = mne.read_evokeds(fname_evokeds)
    
    # Calculate stc for each evoked response individually
    for idx, evoked in enumerate(evokeds):
        stim = stimuli[idx]

        fname_stc = get_fname(subject, 'stc', src_spacing = src_spacing,
                              stc_method = stc_method, task = task, stim=stim)
        fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
        
        if overwrite or not os.path.isfile(fpath_stc + '-lh.stc'):
            stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method = stc_method)    
            stc.save(fpath_stc)
        
def morph_to_fsaverage(subject, project_dir, src_spacing, stc_method,
                       task, stimuli, overwrite):
    '''
    Morph source estimates of given subjects to fsaverage mesh and save the
    morphed stcs in <project_dir>/Data/stc_m/

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
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    # Load surface time courses
    stcs = {}
    for stimulus in stimuli:
        fname = get_fname(subject, 'stc', stc_method = stc_method, src_spacing=src_spacing,
                          task = task, stim = stimulus)
        fpath = os.path.join(project_dir, 'Data', 'stc', fname)
        stcs[stimulus] = mne.read_source_estimate(fpath)
    
    # Morph each stimulus stc to fsaverage and save to disk
    for stim in stcs:
        fname_stc_m = get_fname(subject, 'stc_m', stc_method = stc_method,
                                src_spacing = src_spacing, task = task, stim = stim)
        fpath_stc_m = os.path.join(project_dir, 'Data', 'stc_m', fname_stc_m)
        
        if overwrite or not os.path.isfile(fpath_stc_m + '-lh.stc'):
            morph = mne.compute_source_morph(stcs[stim], subject_from = subject,
                                             subject_to = 'fsaverage')
            stc_m = morph.apply(stcs[stim])
            stc_m.save(fpath_stc_m)

def average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                              task, stimuli, overwrite):
    '''
    Average the source time courses that have been morphed to fsaverage and
    save them per-stimuli to <project_dir>/Data/avg/

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
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
        
    # Average each stimulus and save the avg stc to disk
    for stim in stimuli:
        fname = get_fname('fsaverage', 'stc', stc_method = stc_method, 
                          src_spacing = src_spacing, task = task, stim = stim)
        fpath = os.path.join(project_dir, 'Data', 'avg', fname)
        
        if overwrite or not os.path.isfile(fpath):
            # Load stcs for all subjects with this stimulus
            stcs = []
            for subject in subjects:
                fname_m = get_fname(subject, 'stc_m', stc_method = stc_method,
                                    src_spacing = src_spacing, task = task, stim = stim)
                fpath_m = os.path.join(project_dir, 'Data', 'stc_m', fname_m)
                stcs.append(mne.read_source_estimate(fpath_m))
            
            # Set the first stc as base and add all others to it, divide by n
            avg = stcs[0].copy()
            for i in range(1, len(subjects)):
                avg.data += stcs[i].data
            avg.data = avg.data / len(subjects)
            
            # Save to disk
            print(fpath)
            avg.save(fpath)

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
    colors_rgb = np.array([matplotlib.colors.to_rgba(c) for c in colors])
    
    
    # Find peaks for each averaged stimulus stc
    for c_idx, stim in enumerate(stimuli):
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
                print(stimuli[i] + ' and ' + stimuli[dup_idx] + ' have the same seed on hemi ' + peak_hemis[dup_idx])
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
    colors_rgb = np.array([matplotlib.colors.to_rgba(c) for c in colors])
    
    
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
    for c_idx, stim in enumerate(stimuli):
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
    