#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:31:35 2021

@author: hietalp2
"""

import mne
import os

def get_fname(subject, ftype, stc_method = None, src_spacing = None,
              fname_raw = None, task = None):
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
        Task in the evoked response, e.g. 'f_sector1'. The default is None.

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
        return '-'.join([subject, src_spacing, stc_method, task])
    else:
        raise ValueError('Invalid file type ' + ftype)

def compute_source_space(subject, project_dir, src_spacing, overwrite = False):
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

    Returns
    -------
    None.
    '''
    
    fname = get_fname(subject, 'src', src_spacing = src_spacing)
    fpath = os.path.join(project_dir, 'Data', 'src', fname)
    
    if overwrite or not os.path.isfile(fpath):
        src = mne.setup_source_space(subject, spacing = src_spacing, add_dist='patch')
        src.save(fpath, overwrite = True)
    
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
                               fname_evokeds, overwrite = False):
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
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
    
    # Load the inverse operator and evoked responses
    fname_inv = get_fname(subject, 'inv', src_spacing = src_spacing, fname_raw = raw)
    inv = mne.minimum_norm.read_inverse_operator(os.path.join(project_dir, 'Data', 'inv', fname_inv))
    
    evokeds = mne.read_evokeds(fname_evokeds)
    
    # Calculate stc for each evoked response individually
    for evoked in evokeds:
        comment = evoked.comment.split(', ')
        task = comment[0].split(' ')[1] + '_sector' + comment[1].split(' ')[1]

        fname_stc = get_fname(subject, 'stc', src_spacing = src_spacing,
                              stc_method = stc_method, task = task)
        fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
        
        if overwrite or not os.path.isfile(fpath_stc + '-lh.stc'):
            stc = mne.minimum_norm.apply_inverse(evoked, inv, method = stc_method)    
            stc.save(fpath_stc)