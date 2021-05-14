#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:04:28 2021

@author: hietalp2
"""
from visbrain.gui import Brain
from visbrain.objects import BrainObj, SceneObj
from visbrain.io import read_stc
import os

# Read the *.stc file :
stc_fname = os.path.join('/m/nbe/scratch/megci/MFinverse/', 'Data', 'avg', 'fsaverage-oct6-dSPM-f-sector17-lh.stc')
file = read_stc(stc_fname)

# Get the data and vertices from the file :
data = file['data'][:, 132]
t = max(data)
data[data < t] = 0
vertices = file['vertices']

# Define a brain object and add the data to the mesh :
b_obj = BrainObj('inflated', translucent=False, hemisphere='left')
b_obj.add_activation(data=data, vertices=vertices, smoothing_steps=15, hide_under = 4, cmap = 'Blues',
                     hemisphere='left')

# Read the *.stc file :
stc_fname = os.path.join('/m/nbe/scratch/megci/MFinverse/', 'Data', 'avg', 'fsaverage-oct6-dSPM-f-sector21-lh.stc')
file = read_stc(stc_fname)

# Get the data and vertices from the file :
data2 = file['data'][:, 132]
t = max(data2)
data2[data2 < t] = 0
vertices2 = file['vertices']

# Define a brain object and add the data to the mesh :
b_obj.add_activation(data=data2, vertices=vertices2, smoothing_steps=15, hide_under = 4, cmap = 'Reds',
                     hemisphere='left')

# Finally, pass the brain object to the Brain module :
vb = Brain(brain_obj=b_obj)
vb.rotate('left')
vb.show()