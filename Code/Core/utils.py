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
              fname_raw = None, task = None, stim = None, layers = 1,
              hemi = 'lh', suffix = None):
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
        File name of raw recording from which the info is extracted.
        The default is None.
    task : str, optional
        Task in the evoked response, e.g. 'f'. The default is None.
    stim: str, optional
        Stimulus name, e.g. 'sector1'. The default is None.
    layers: str, optional
        How many layers the BEM model has. The default is 1.
    hemi: str, optional
        Which hemisphere the name is for, e.g. 'lh'. The default is 'lh'.
    suffix : str, optional
        A string to append to the end of the filename before file extension.

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
        return '-'.join(filter(None, [subject, src_spacing, suffix, 'src.fif']))
    elif ftype == 'fwd':
        return '-'.join(filter(None, [subject, src_spacing, suffix, 'fwd.fif']))
    elif ftype == 'cov':
        run = os.path.split(fname_raw)[-1].split('_')[0]
        return '-'.join(filter(None, [subject, run, suffix, 'cov.fif']))
    elif ftype == 'inv':
        run = os.path.split(fname_raw)[-1].split('_')[0]
        return '-'.join(filter(None, [subject, src_spacing, run, suffix, 'inv.fif']))
    elif ftype == 'stc':
        return '-'.join(filter(None, [subject, src_spacing, stc_method, task,
                                      stim, suffix]))
    elif ftype == 'stc_m':
        return '-'.join(filter(None, [subject, src_spacing, stc_method,
                                      'fsaverage', task, stim, suffix]))
    elif ftype == 'bem':
        return '-'.join(filter(None, [subject, layers, suffix, 'shell-bem-sol.fif']))
    elif ftype == 'label':
        return '-'.join(filter(None, [subject, src_spacing, stc_method, task,
                                      stim, suffix]))
    else:
        raise ValueError('Invalid file type ' + ftype)

def prepare_directories(project_dir):
    '''
    Prepare directories used in the pipelines in <project_dir>/*

    Parameters
    ----------
    project_dir : str
        Base directory to create the project folders in.
    
    Returns
    -------
    None.
    '''

    for dirname in ['Data',
                    'Data/fwd',
                    'Data/src',
                    'Data/inv',
                    'Data/stc',
                    'Data/stc_m',
                    'Data/avg',
                    'Data/labels',
                    'Data/cov',
                    'Data/plot',
                    'Data/slurm_out']:
        try:
            os.makedirs(project_dir + dirname)
        except FileExistsError:
            pass

def average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                              task, stimuli, suffix = None, timing = None,
                              overwrite = False):
    '''
    Average the source time courses that have been morphed to fsaverage and
    save them per-stimuli to <project_dir>/Data/avg/

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Data subfolder.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    task: str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    suffix : str
        Suffix to append to the end of the output filename before -ave.fif.
    timing : None or list, optional
        List of single timepoints per subject to which the estimate is confined
        to, default is False (average all timepoints)
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.

    Returns
    -------
    None.
    '''
        
    # Average each stimulus and save the avg stc to disk
    for stim in stimuli:
        fname = get_fname('fsaverage', 'stc', stc_method = stc_method, 
                          src_spacing = src_spacing, task = task, stim = stim,
                          suffix = suffix)
        fpath = os.path.join(project_dir, 'Data', 'avg', fname)
        
        if overwrite or not os.path.isfile(fpath + '-lh.stc'):
            # Load stcs for all subjects with this stimulus
            stcs = []
            for subject in subjects:
                fname_m = get_fname(subject, 'stc_m', stc_method = stc_method,
                                    src_spacing = src_spacing, task = task,
                                    stim = stim,
                                    suffix = (suffix if stc_method == 'remtw' else None))
                fpath_m = os.path.join(project_dir, 'Data', 'stc_m', fname_m)
                print(fpath_m)
                stcs.append(mne.read_source_estimate(fpath_m))
            
            # Set the first stc as base and add all others to it, divide by n
            if timing == None:
                avg = stcs[0].copy()
                avg.data = abs(avg.data)
                for i in range(1, len(subjects)):
                    avg.data += abs(stcs[i].data)
            else:
                avg = stcs[0].crop(tmin = timing[0], tmax = timing[0],
                                   include_tmax = True).copy()
                avg.data = abs(avg.data)
                for i in range(1, len(subjects)):
                    avg.data += abs(stcs[i].crop(tmin = timing[i], tmax = timing[i],
                                                 include_tmax = True).data)
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

def find_peaks(project_dir, src_spacing, stc_method, task, stimuli, bilaterals,
               suffix, return_index = False, subject = 'fsaverage', mode = 'avg',
               stc = None, time = None):
    '''
    Find peak indices and hemispheres for averaged stc files

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
    bilaterals: list of str
        List of bilateral stimuli (on midline); label peaks on both hemis.
    suffix : str
        Suffix to append to stc filename before -avg.fif
    return_index : bool
        Whether to return peak vertex index instead of vertex ID (default)
    subject : str
        Name of the subject the stc is for, defaults to fsaverage
    mode : str
        Type of stc, can be either stc, stc_m or avg. Defaults to avg
    stc : mne.SourceEstimate
        If source estimate is given, skip loading it again.
    time : float
        A timepoint for which to get the peak. If None, overall peak is used.
    
    Returns
    -------
    peaks : list of int
        List of peak valued vertex indices for each stimulus
    peak_hemis : list of str
        List of hemispheres for each peak index in same order as peaks
    '''

    peaks = []
    peak_hemis = []

    for stim in stimuli:
        if stc == None:
            fname = get_fname(('fsaverage' if mode == 'avg' else subject),
                            ('stc_m' if mode == 'stc_m' else 'stc'),
                            stc_method = stc_method, src_spacing = src_spacing,
                            task = task, stim = stim, suffix = suffix)
            fpath = os.path.join(project_dir, 'Data', mode, fname)
            print('Reading ' + fpath)
            stc = mne.read_source_estimate(fpath)
        
        if time != None and stc.data.shape[1] > 1:
            stc.crop(tmin = time, tmax = time)
        
        if stim not in bilaterals:
            # Not bilateral; get global peak and store it in peaks and hemis
            if np.max(abs(stc.lh_data)) > np.max(abs(stc.rh_data)):
                hemi = 'lh'
            else:
                hemi = 'rh'
            print(stc.get_peak(hemi = hemi, vert_as_index = return_index))
            peaks.append(stc.get_peak(hemi = hemi, vert_as_index = return_index)[0])
            peak_hemis.append(hemi)
        else:
            # Bilateral; get both lh and rh peaks and store them
            peaks.append(stc.get_peak(hemi = 'lh', vert_as_index = return_index)[0])
            peaks.append(stc.get_peak(hemi = 'rh', vert_as_index = return_index)[0])
            peak_hemis += ['lh', 'rh']
        
        stc = None
    
    return (peaks, peak_hemis)

def tabulate_geodesics(project_dir, src_spacing, stc_method, task, stimuli,
                       bilaterals, suffix, overwrite, counts = [1, 5, 10, 15, 20],
                       subject = 'fsaverage', mode = 'avg'):
    '''
    Compute geodesic distances between peaks and the V1 label vertices and
    save them in a csv file. Each row is one subject count in <counts> and
    each column is a peak (two peaks for bilateral stimuli).

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
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    bilaterals : list of str
        Names of bilateral stimuli
    suffix : str
        Suffix to append to the end of the output filename.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.
    counts : list, optional
        List of subject counts to calculate the distances for,
        by default [1, 5, 10, 15, 20]
    subject : str, optional
        Subject for which the distances are calculated, by default 'fsaverage'
    mode : str, optional
        Stc type, either 'stc', 'stc_m' or 'avg', by default 'avg'
    '''

    suffix = suffix.lstrip('0123456789')
    distances = np.zeros((len(counts), len(stimuli) + len(bilaterals)))

    # Load V1 labels and source space with distances
    v1_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + (subject if (mode == 'stc') else 'fsaverage')
                            + '/label/lh.V1_exvivo.label',
                            (subject if(mode == 'stc') else 'fsaverage'))
    v1_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + (subject if (mode == 'stc') else 'fsaverage')
                            + '/label/rh.V1_exvivo.label',
                            (subject if (mode == 'stc') else 'fsaverage'))
    fname_src = get_fname((subject if (mode == 'stc') else 'fsaverage'), 'src',
                           src_spacing = src_spacing)
    src = mne.read_source_spaces(project_dir + 'Data/src/' + fname_src,
                                 verbose = False)

    # Comparison list to get inf distance if the peak is on the wrong hemi
    expected_hemi = [h + 'h' for h in "rrlrllllrr"] * 3

    for n_idx, suffix in enumerate([str(n) + suffix for n in counts]):
        print('Solving geodesics for ' + suffix)

        peaks, peak_hemis = find_peaks(project_dir, src_spacing, stc_method,
                                       task, stimuli, bilaterals, suffix,
                                       return_index = False, subject = subject,
                                       mode = mode)
        print(peaks)
        
        # Count average geodesic distance between peaks and V1 label points
        for peak_idx, peak in enumerate(peaks):
            dist = 0
            hemi_idx = 0 if peak_hemis[peak_idx] == 'lh' else 1
            if peak_hemis[peak_idx] == expected_hemi[peak_idx]:
                # Get points used by the label
                if peak_hemis[peak_idx] == 'lh':
                    used_verts = v1_lh.get_vertices_used(src[0]['vertno'])
                else:
                    used_verts = v1_rh.get_vertices_used(src[1]['vertno'])

                # Calculate avg distance between peak and label points
                for vertex in used_verts:
                    d = src[hemi_idx]['dist'][peak, vertex] * 1000
                    if d == 0:
                        print(peak, vertex)
                        a=1
                    dist += src[hemi_idx]['dist'][peak, vertex] * 1000 # m -> mm
                dist /= len(used_verts)
            # Wrong hemi
            else:
                dist = np.inf
            distances[n_idx, peak_idx] = dist
    
    np.savetxt(project_dir + 'Data/plot/distances_' + subject + '_' + mode + '.csv',
               distances, delimiter = ',', fmt = '%.3f')
