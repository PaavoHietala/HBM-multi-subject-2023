#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrappers for different commonly used functions of the MNE package.

Created on Tue Feb  2 15:31:35 2021

@author: hietalp2
"""

import mne
import os
from .utils import get_fname

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
                                         add_dist=add_dist, n_jobs = 16)
        else:
            # Load fsaverage source space and morph it to subject
            fname_ref = get_fname('fsaverage', 'src', src_spacing = src_spacing)
            fpath_ref = os.path.join(project_dir, 'Data', 'src', fname_ref)
            
            src_ref = mne.read_source_spaces(fpath_ref)
            src = mne.morph_source_spaces(src_ref, subject_to=subject)
            if add_dist:
                mne.add_source_space_distances(src, n_jobs=16)
        src.save(fpath, overwrite = True)

def calculate_bem_solution(subject, project_dir, overwrite):
    '''
    Calculate 1-shell and 3-shell bem solutions from FreeSurfer surfaces and
    save them in <subjects_dir>/<subject>/bem/

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

    # Conductivities used in 1-shell and 3-shell models
    cond = {1 : [0.3], 3 : [0.3, 0.006, 0.3]}

    # Create both 1-layer and 3-layer BEM solution
    for layers in cond:
        fname = get_fname(subject, 'bem', layers = str(layers))
        fpath = os.path.join(mne.get_config('SUBJECTS_DIR'), subject, 'bem', fname)

        if overwrite or not os.path.isfile(fpath):
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

def compute_covariance_matrix(subject, project_dir, raw, overwrite = False):
    '''
    Compute covariance matrix from given raw rest recording and save it in
    <project_dir>/Data/cov/

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
        fname = get_fname(subject, 'stc', stc_method = stc_method,
                          src_spacing=src_spacing, task = task, stim = stimulus)
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
