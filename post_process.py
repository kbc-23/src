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

# %% Create the full image mask
def fullres_mask_predict(filename_eg, traindata_img_dir, masks):
    filename_eg_input = filename_eg + '.tif' # image name
    inp_img = cv2.imread(str(Path(traindata_img_dir,filename_eg_input)))
    # plt.imshow(cv2.cvtColor(inp_img,cv2.COLOR_BGR2RGB))
    # plt.title(filename_eg)
    # plt.axis('off')
    img = inp_img.copy()
    pred_mask = np.zeros([1000, 1000],dtype = masks.dtype)
    for i in range(len(class_ids)):
        [r1, c1, r2, c2] = boxes[i, :]
        mask_i = masks[i,:,:]
        mask_i_resized = cv2.resize(mask_i, (c2-c1,r2-r1), interpolation= cv2.INTER_LINEAR)
        pred_mask[r1:r2,c1:c2] = mask_i_resized
    
    return pred_mask, img

# %% display image
def display_img(img,title):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

# %% Solve overlapping problem
def solve_overlap(pred_mask,img):
    pred_mask_fullsc = ((pred_mask / np.max(pred_mask))* 255.0).astype(np.uint8)

    # threshold
    ret, thresh = cv2.threshold(pred_mask_fullsc,0,255,cv2.THRESH_OTSU)
    
    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    
    # background area
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    
    # foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
    n_instances, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    n_instances, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    #instances = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,255,0]
    
    return markers, n_instances, img


if __name__ == "__main__":
    # %% Read data
    data_dir = '../MoNuSeg_1000x1000_for_candidate/data_raw' # main data dir
    traindata_img_dir = '../MoNuSeg 2018 Training Data/Tissue Images' # train data dir
    traindata_anno_dir = '../MoNuSeg 2018 Training Data/Annotations' # train data dir
    
    # %% Output dir 
    output_dir = '../output'
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    
    for file in os.listdir(data_dir):
        filename_eg = file[0:-4]
        filename_eg_pred = filename_eg + '.npz'
        data = np.load(Path(data_dir,filename_eg_pred)) # load data
        print('{0} intermediate data loaded'.format(filename_eg))
        
        # %% Extract variables
        boxes = data['boxes']
        class_ids = data['class_ids']
        scores = data['scores']
        masks = data['masks']
        print('{0} instances found'.format(len(class_ids)))
        
        [pred_mask, img] = fullres_mask_predict(filename_eg, traindata_img_dir, masks)
    
        [markers, n_instances, img] = solve_overlap(pred_mask, img)
        print('{0} non-overlapping objects found'.format(n_instances))
        
        display_img(img,filename_eg)
        
        output_filename =  filename_eg + '_resolved' + '.png'
        cv2.imwrite(str(Path(output_dir,output_filename)), img)
        print('Processed image saved in the path: {0}'.format(str(Path(output_dir,output_filename))))
        