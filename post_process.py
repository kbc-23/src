#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:16:15 2022

@author: chowdhuryk
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# %% Read data
data_dir = '../MoNuSeg_1000x1000_for_candidate/data_raw' # main data dir
traindata_img_dir = '../MoNuSeg 2018 Training Data/Tissue Images' # train data dir
traindata_anno_dir = '../MoNuSeg 2018 Training Data/Annotations' # train data dir


filename_eg = 'TCGA-18-5592-01Z-00-DX1'

filename_eg_input = filename_eg + '.tif'

inp_img = cv2.imread(str(Path(traindata_img_dir,filename_eg_input)))
plt.imshow(cv2.cvtColor(inp_img,cv2.COLOR_BGR2RGB))
plt.title(filename_eg)
plt.axis('off')
