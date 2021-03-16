#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:18:08 2021

@author: hietalp2
"""
import mne
from mne.datasets import sample

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis-meg'

stc = mne.read_source_estimate(fname)

labels =  mne.grow_labels('fsaverage', stc.get_peak(hemi = 'lh')[0], 1, 1, overlap = False)