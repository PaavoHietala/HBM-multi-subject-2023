#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:52:12 2021

@author: hietalp2
"""

from mne.viz import Brain
import os
from surfer.io import read_stc

stc_fname = os.path.join('/m/nbe/scratch/megci/MFinverse/', 'Data', 'avg',
                             'fsaverage-oct6-dSPM-f-sector17-lh.stc')
stc = read_stc(stc_fname)

# data and vertices for which the data is defined
data = stc['data']
vertices = stc['vertices']

#2
stc_fname = os.path.join('/m/nbe/scratch/megci/MFinverse/', 'Data', 'avg',
                         'fsaverage-oct6-dSPM-f-sector21-lh.stc')
stc = read_stc(stc_fname)

# 2data and vertices for which the data is defined
data2 = stc['data']
vertices2 = stc['vertices']

b = Brain('fsaverage', 'lh', 'inflated')
b.add_data(data, vertices = vertices)
b.add_data(data2, vertices = vertices2)
b.set_time(0.08)
b.show()