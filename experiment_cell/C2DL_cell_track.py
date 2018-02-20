#!/usr/bin/python3
# The cell-tracking program for the C2DL images
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import os, sys, glob, shutil, cv2, random, warnings, time, pickle

sys.path.insert(0,'/Users/' + USR + '/Dropbox/gromov_wasserstein_dist/POT/')
import ot
from bio_cell_track_helper import *
from bio_cell_track import *




I1 = cv2.imread(PATH_DATA + 't101.tif',0)
I2 = cv2.imread(PATH_DATA + 't102.tif',0)
options = dict()
options['N_pixels'] = 20
options['cell_type'] = 'C2DL'
options['visualize'] = True
options['img_name_source'] = 't101'
options['img_folder'] = '/Users/' + USR + '/Desktop/'
options['N_pts'] = 1000

pts_s, label_s = get_seg_labels(I1, options)
pts_t, label_t = get_seg_labels(I2, options)





if 'N_pts' in options:
	N_pts = options['N_pts']
else:
	N_pts = 500

options['weight_M'] = 1


# Random seed for points selection
if 'seed' in options:
	random.seed(options['seed'])

# The first-order mapping computation method. Choice: emd/sinkhorn
if 'method_first' in options:
	method_first = options['method_first']
else:
	method_first = 'sinkhorn'

# In pts, the last column is the label of the point
if len(pts_s) > N_pts:
	idx_s     = np.random.randint(len(pts_s), size=N_pts)
	pts_s_sub = pts_s[idx_s,:]
else:
	pts_s_sub = pts_s

if len(pts_t) > N_pts:
	idx_t     = np.random.randint(len(pts_t), size=N_pts)
	pts_t_sub = pts_t[idx_t,:]
else:
	pts_t_sub = pts_t

label_s_sub = pts_s_sub[:,-1]
label_t_sub = pts_t_sub[:,-1]
pts_s_sub   = pts_s_sub[:,:-1]
pts_t_sub   = pts_t_sub[:,:-1]
label_s     = pts_s[:,-1]
label_t     = pts_t[:,-1]
pts_s       = pts_s[:,:-1]
pts_t       = pts_t[:,:-1]
# Compute first-order information
print('Compute the first-order transformation...')
tic0 = time.clock()
tic = time.clock()
M   = ot.dist(pts_s_sub, pts_t_sub, metric='euclidean')
M  /= (M.max() + 1e-6)
# Point-wise probability
p_s = np.ones((len(pts_s_sub),)) / len(pts_s_sub)
p_t = np.ones((len(pts_t_sub),)) / len(pts_t_sub)
# Compute the transfer matrix
if method_first == 'emd':
	G1 = ot.emd(p_s, p_t, M)
else:
	eps = get_sinkhorn_eps(p_s, p_t, M)
	G1  = ot.sinkhorn(p_s, p_t, M, eps)

toc = time.clock()
print('Done. Time:' + str(toc - tic) + ' seconds\n')
# Visualize the general transform in the first order
if 'visualize' in options and options['visualize'] == True:
	print('Visualize the 1st order mapping ...')
	if 'img_name_source' in options:
		img_name_prefix = options['img_name_source']
	else:
		img_name_prefix = 'trans_img_'
	options['img_name'] = img_name_prefix + 'whole_img.png'
	draw_mapping(pts_s_sub, pts_t_sub, G1, options)

list_gw, list_pts_s, list_pts_t, list_subset = get_mix_order_alignment(pts_s, pts_t, options)



