#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:55:31 2021

@author: hietalp2
"""
import mne
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MNE_operations import mne_operations as mne_op
import groupmne

#%%

stc = mne.read_source_estimate('/m/nbe/scratch/megci/MFinverse/Classic/Data/stc/MEGCI_S12-oct6-eLORETA-f-sector14-lh.stc')

b = stc.plot('MEGCI_S12', 'inflated','lh')

# b = mne.viz.Brain('MEGCI_S1', 'lh', 'inflated', show = False)
# b.add_data(stc)
b.show_view({'elevation' : 100, 'azimuth' : -55}, distance = 500)
# b.show()

#%% plot
subject = 'MEGCI_S9'
method = 'remtw'
stim = 'sector24'
hemi = 'rh'
stc = mne.read_source_estimate('/m/nbe/scratch/megci/MFinverse/reMTW/Data/stc/'+subject+'-ico4-'+method+'-f-'+stim+'-'+hemi+'.stc')

#for i in range(1,25):
#    stim = 'sector' + str(i)
#stc = mne.read_source_estimate('/m/nbe/scratch/megci/MFinverse/reMTW/Data/avg/'+subject+'-ico4-'+method+'-f-'+stim+'-'+hemi+'.stc')

if np.count_nonzero(stc.data[:]) == 0:
    print("Zero estimate")
else:
    b = stc.plot(subject, 'inflated', hemi)
    b.add_text(0.1, 0.9, '-'.join([subject, method, stim]), 'title')
    
    labels = mne.read_labels_from_annot(subject, 'aparc.a2009s', hemi)
    
    lbl = [label for label in labels if label.name == 'S_calcarine-' + hemi][0]
    b.add_label(lbl, borders = 2) 
    if hemi == 'lh':
        v1_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'+subject+'/label/lh.V1_exvivo.label', subject)
        b.add_label(v1_lh, borders = 2)
        b.show_view({'elevation' : 100, 'azimuth' : -55}, distance = 500)
    else:
        v1_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'+subject+'/label/rh.V1_exvivo.label', subject)
        b.add_label(v1_rh, borders = 2)
        b.show_view({'elevation' : 100, 'azimuth' : -125}, distance = 500)
    #b.save_image('/m/nbe/scratch/megci/MFinverse/reMTW/Data/plot/' + '_'.join([subject, method, stim, hemi]) + '.jpg')
    #b.close()
#%% get lambda_max
project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/'
subject = 'MEGCI_S1'

fname_fwd = mne_op.get_fname(subject, 'fwd', src_spacing = 'ico4')
fwd = mne.read_forward_solution(os.path.join(project_dir, 'Data', 'fwd', fname_fwd))

L = fwd['sol']['data']
n = L.shape[0]
L_n = L / np.linalg.norm(L, axis = 1)[:,None]

fpath = project_dir + 'Data/Evoked/' + subject + '_f-ave.fif'
evo = mne.read_evokeds(fpath, verbose = False)[13]
evo_n = evo.data / np.linalg.norm(evo.data, axis = 0)[None,:]

res = []
for yid in range(501):
    lambda_max = np.max(np.abs(L_n.T.dot(evo_n[:,yid]))) / n
    res.append(lambda_max)
print(res)

#%%

stc = mne.read_source_estimate('/m/nbe/scratch/megci/MFinverse/reMTW/Data/stc/MEGCI_S16-ico4-remtw_mem-f-sector14-lh.stc')

active = []
for i in range(501):
    active.append(np.count_nonzero(stc.data[:,i] == 0))

plt.plot(active)

#%%

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(17,25)) if idx not in exclude]

# Which task is currently investigated

task = 'f'

# Which stimuli to analyze

stimuli = ['sector' + str(num) for num in range(1,25)]

# Root directory of the project
project_dir = '/m/nbe/scratch/megci/MFinverse/'

#%% Plot estimated SNR for all ques

#inv = mne.minimum_norm.read_inverse_operator('/m/nbe/scratch/megci/MFinverse/Data/inv/MEGCI_S1-oct6-rest1-inv.fif')
evokeds = mne.read_evokeds('/m/nbe/scratch/megci/MFinverse/Data/Evoked/MEGCI_S1_f-ave.fif')
fwd = mne.read_forward_solution('/m/nbe/scratch/megci/MFinverse/Data/fwd/MEGCI_S1-oct6-fwd.fif')
cov = mne.read_cov('/m/nbe/scratch/megci/MFinverse/Data/cov/MEGCI_S1-rest1-cov.fif')

snrs = []
for i, s in enumerate(stimuli):
    fname_stc = mne_op.get_fname('MEGCI_S1', 'stc', src_spacing = 'oct6',
                                 stc_method = 'dSPM', task = task, stim=s)
    fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
    stc = mne.read_source_estimate(fpath_stc)
    snrs.append(stc.estimate_snr(evokeds[0].info, fwd, cov))

#%%
time = np.linspace(-50, 450, num = 501)
plt.figure()
for p in snrs:
    plt.plot(time, p)
    
#%%

for j in range(10):
    for i in range(6):
        if i * j == 30:
            break
        print(j, i)
    else:
        print("else")
        
#%%
base = '/m/nbe/scratch/megci/data/'

src_ref = mne.setup_source_space('fsaverage', spacing = 'ico4', add_dist = False)
src_ref.save('/m/nbe/scratch/megci/MFinverse/src_ref.fif', overwrite = True)
#src_ref = mne.read_source_spaces('/m/nbe/scratch/megci/MFinverse/src_ref.fif')[0]

src = mne.morph_source_spaces(src_ref, subject_to = 'MEGCI_S1')
src.save('/m/nbe/scratch/megci/MFinverse/src.fif', overwrite = True)
#src = mne.read_source_spaces('/m/nbe/scratch/megci/MFinverse/src.fif')[0]

raw = base + 'MEG/megci_rawdata_mc_ic/megci_s1_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif'
trans = base + 'FS_Subjects_MEGCI/MEGCI_S1/mri/T1-neuromag/sets/COR-ahenriks.fif'
bem_f = base + 'FS_Subjects_MEGCI/MEGCI_S1/bem/MEGCI_S1-1-shell-bem-sol.fif'

bem = mne.read_bem_solution(bem_f)
fwd = [mne.make_forward_solution(raw,  trans = trans, src = src, bem = bem, meg = True, eeg = False)]

groupmne.group_model.prepare_fwds(fwd, src_ref, copy = False )

#%% asd

raw1 = '/m/nbe/scratch/megci/MFinverse/Data/Evoked/MEGCI_S1_f-ave.fif'
ev1 = mne.read_evokeds(fpath, verbose = False)[13]
ev1.plot()

raw2 = '/m/nbe/scratch/megci/MFinverse/groupmne tutorial/HF_SEF/MEG/subject_a/sef2_right_raw-1.fif'

def process_meg(raw_name):
    """Extract epochs from a raw fif file.

    Parameters
    ----------
    raw_name: str
        path to the raw fif file.

    Returns
    -------
    epochs: Epochs instance

    """
    raw = mne.io.read_raw_fif(raw_name)
    events = mne.find_events(raw)

    event_id = dict(hf=1)  # event trigger and conditions
    tmin = -0.05  # start of each epoch (50ms before the trigger)
    tmax = 0.3  # end of each epoch (300ms after the trigger)
    baseline = (None, 0)  # means from the first instant to t = 0
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=baseline)
    return epochs


#epochs_s = process_meg(raw2)
#evokeds = epochs_s.average()
#evokeds.plot()
