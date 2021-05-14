#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used exclusively for minimum norm inverse solutions.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
"""

import os
import mne
from .utils import get_fname

def construct_inverse_operator(subject, project_dir, raw, src_spacing, overwrite = False):
    '''
    Construct MNE inverse operator from forward solution and save it in
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
    inv = mne.minimum_norm.read_inverse_operator(os.path.join(project_dir, 'Data',
                                                              'inv', fname_inv))
    
    evokeds = mne.read_evokeds(fname_evokeds)
    
    # Calculate stc for each evoked response individually
    for idx, evoked in enumerate(evokeds):
        stim = stimuli[idx]

        fname_stc = get_fname(subject, 'stc', src_spacing = src_spacing,
                              stc_method = stc_method, task = task, stim=stim)
        fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
        
        if overwrite or not os.path.isfile(fpath_stc + '-lh.stc'):
            print("Subject:" + subject + ":stimulus:" + stim)
            stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
                                                 method = stc_method)    
            stc.save(fpath_stc)