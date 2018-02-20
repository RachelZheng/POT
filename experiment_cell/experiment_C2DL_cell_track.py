#!/usr/bin/python3
# The cell-tracking program for the C2DL images
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import os, sys, glob, shutil, cv2, random, warnings, time, pickle

from config import *
sys.path.insert(0, PATH_OT)
import ot
from bio_cell_track_helper import *
from bio_cell_track import *

# Get the cell-tracking examples
os.chdir(options['data_folder']) # Change to the data folder
img_files = sorted(glob.glob('*.tif'))
PATH_OUTPUT = options['img_folder']

for idx_img in range(len(img_files) - 1):
	# For each transform between 2 imgs, create a folder to save the further information
	options['img_folder'] = PATH_OUTPUT + img_files[idx_img][:-4] + '/'
	if not os.path.isdir(options['img_folder']):
		os.mkdir(options['img_folder'])
	I1 = cv2.imread(img_files[idx_img],0)
	I2 = cv2.imread(img_files[idx_img + 1],0)
	# ----- Compute alignments of each cell ------
	pts_s, label_s = get_seg_labels(I1, options) # source pts, labels 
	pts_t, label_t = get_seg_labels(I2, options) # target pts, labels
	# ----- Visualize the cell segmentation ------	
	label_hue   = np.uint8(179 * label_s / np.max(label_s))
	blank_ch    = 255 * np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
	# set bg label to black
	labeled_img[label_hue==0] = 0
	cv2.imwrite(options['img_folder'] + 'source_segmentation.png', labeled_img)
	# ----- Do the first-order mapping ------	
	T, list_subset = get_1st_mapping(pts_s, pts_t, options)
	label_s = pts_s[:,-1]
	label_t = pts_t[:,-1]
	pts_s   = pts_s[:,:-1]
	pts_t   = pts_t[:,:-1]
	# First use the 0 weight of matrix to see how much improvement
	for weight in np.linspace(0,1,11).tolist():
		options['weight_M'] = weight
		list_gw, list_pts_s, list_pts_t = get_subset_gw(pts_s, pts_t, label_s, label_t, list_subset, options)
		draw_mapping_list(list_pts_s, list_pts_t, list_gw, list_subset, options)

