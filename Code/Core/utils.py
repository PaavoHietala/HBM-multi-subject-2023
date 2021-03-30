#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Common functions used in multiple pipelines.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import os
import mne
import numpy as np

def get_fname(subject, ftype, stc_method = None, src_spacing = None,
              fname_raw = None, task = None, stim = None, layers = 1, hemi = 'lh'):
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
        How many layers the BEM model has. The default is 1.
    hemi: str, optional
        Which hemisphere the name is for, e.g. 'lh'. The default is 'lh'.

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

def prepare_directories(project_dir):
    for dirname in ['Data',
                    'Data/fwd',
                    'Data/src',
                    'Data/inv',
                    'Data/stc',
                    'Data/stc_m',
                    'Data/avg',
                    'Data/labels',
                    'Data/cov',
                    'Data/plot']:
        try:
            os.makedirs(project_dir + dirname)
        except FileExistsError:
            pass

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
        
        if overwrite or not os.path.isfile(fpath + '-lh.stc'):
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

def restrict_src_to_label(subject, project_dir, src_spacing, overwrite, labels):
    '''
    Experimental function to restrict source points to given labels. Works, but
    apparently reMTW expects closed surface so the source distance matrix creation
    fails.

    No plans for further development.
    '''
    fname = get_fname(subject, 'src', src_spacing = src_spacing)
    fpath = os.path.join(project_dir, 'Data', 'src', fname)

    if overwrite or not os.path.isfile(fpath):
        # Load original source space
        src = mne.read_source_spaces(fpath)

        for s, hemi in zip(src, ['lh', 'rh']):
            vertno = np.where(s['inuse'])[0]

            # Mark only the source points under the labels active
            verts = np.concatenate([l.get_vertices_used(vertno) for l in labels
                                    if l.name != 'Unknown-' + hemi and hemi in l.name])
            # tris = np.concatenate([l.get_tris(s['use_tris'], vertno) for l in labels
            #                        if l.name != 'Unknown-' + hemi and hemi in l.name])

            # Groupmne crashes if lh and rh have different amount of source points
            if hemi == 'rh' and src[0]['nuse'] != src[1]['nuse']:
                verts = verts[:-1]
            deleted = s['nuse'] - len(verts)

            s['inuse'][[i for i in vertno if i not in verts]] = 0
            # s['use_tris'] = tris
            # s['nuse_tri'] = np.array([tris.shape[0]])
            s['nuse'] -= deleted
            s['vertno'] = np.where(s['inuse'])[0]

        print(src[0]['vertno'])
        print(src[0]['inuse'].dtype, len(src[0]['inuse']), type(src[0]['inuse']), src[0]['inuse'].shape)
        print(src[0]['vertno'].dtype, len(src[0]['vertno']), type(src[0]['vertno']), src[0]['vertno'].shape)
        print(type(src[0]['nuse']), src[0]['nuse'])

        #input()
        #src.plot()

        src.save(fpath, overwrite = True)