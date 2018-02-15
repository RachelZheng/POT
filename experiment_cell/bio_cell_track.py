#!/usr/bin/python3
PATH_DATA = '/Users/xvz5220-admin/Dropbox/cell_tracking_data/data_output/'
PATH_RAW = PATH_DATA + '01/'
PATH_LAYER_9 = PATH_DATA + '01_09/'

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import os, sys, glob, shutil, cv2, random, warnings

sys.path.insert(0,'/Users/xvz5220-admin/Dropbox/gromov_wasserstein_dist/POT/')
import ot
from bio_cell_track_help import *


def get_gw_eps(C1, C2, a, b, eps_0 = 1e-2, loss = 'square_loss', weight_M = 0, M = 0):
	""" Get the suitable epsilon value in the ot.gromov_wasserstein
	function.
	Input: all the inputs for 
	"""
	flag_stop = False
	eps       = eps_0
	while not flag_stop:
		try:
			with warnings.catch_warnings(record=True) as w:
				warnings.simplefilter("always")
				gw = ot.gromov_wasserstein(C1, C2, a, b, loss, epsilon=eps, weight_M=weight_M, M=M)
				if len(w) == 0:
					eps/= 2
				else:
					flag_stop = True
		except:
			flag_stop = True
	eps *= 2
	gw = ot.gromov_wasserstein(C1, C2, a, b, loss, epsilon=eps, weight_M=weight_M, M=M)
	return gw, eps


def get_sinkhorn_eps(p_s, p_t, M, eps_0 = 1e-3):
	""" Get the suitable epsilon value in the computation
	"""
	flag_stop = False
	eps = eps_0
	while not flag_stop:
		try:
			with warnings.catch_warnings(record=True) as w:
				warnings.simplefilter("always")
				G1 = ot.sinkhorn(p_s, p_t, M, eps)
			if len(w) == 0:
				eps /= 2
			else:
				flag_stop = True
		except:
			flag_stop = True
	eps *= 2
	return eps


def get_seg_labels(I, options = dict()):
	""" Get image representation pts, distances matrix and 
	Input: grayscale images I
	Output: sample pts X, the labeled image labels_filtered
	"""
	# ------ Options ------ 
	# The object in the image is white(T) or black(F)
	thres     = 120   # Threshold of cutting cells
	N_pixels  = 100 # minimal # of pixels of one cell
	obj_white = True 
	if 'thres' in options:
		thres = options['thres']
	if 'obj_white' in options:
		obj_white = options['obj_white']
	# ------ Computation ------ 
	if obj_white:
		I = cv2.threshold(I, thres, 255, cv2.THRESH_BINARY)[1]
	else:
		I = cv2.threshold(I, 0, thres, cv2.THRESH_BINARY)[1]
	# Find all the connected components
	N_cell, labels, stats, centroids = cv2.connectedComponentsWithStats(I, connectivity=8)
	# Filter components 
	labels_filtered = labels
	sizes           = stats[:, -1]
	for i in range(N_cell - 1, 0, -1):
		if sizes[i] <= N_pixels:
			labels_filtered[labels_filtered == i] = 0
			for j in range(i, N_cell - 1):
				labels_filtered[labels_filtered == j] = j - 1
	"""
	# Visualization
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
	# set bg label to black
	labeled_img[label_hue==0] = 0
	cv2.imwrite('/Users/xvz5220-admin/Desktop/labeled2.png', labeled_img)
	"""
	# Combine labels and pts together
	pts = np.nonzero(labels_filtered)
	pts = np.concatenate((pts[0].reshape((-1,1)),pts[1].reshape((-1,1))),axis = 1)
	ls  = labels_filtered.reshape(1,-1) # labels of pts
	ls  = ls[:, np.all(ls>0,axis=0)].reshape(-1,1)
	pts = np.concatenate((pts, ls),axis = 1)
	return pts, labels_filtered


def get_label_mapping(label_s, label_t, Gs):
	""" Get the mapping between source and target cells
	Input: n1 * 1 labels for source, n2 * 1 labels for target, transfer matrix
	Output: l1 * l2 matrix with relative transformation, which is row-normalized
	### PLEASE BE CAREFUL: OUR LABELS BEGIN WITH 1
	"""
	l_s          = label_s.max()
	l_t          = label_t.max()
	G            = Gs / Gs.max()
	T_label      = np.zeros((l_s, l_t))
	thres        = 1e-6
	G[G < thres] = 0 
	for i_s in range(1,l_s + 1):
		idx_class     = np.where(label_s == i_s)[0] # The index of class i_s from the source
		G_class       = G[idx_class,:] # Sub-transformation matrix
		idx_target    = np.nonzero(G_class)[1].tolist() # matched points idxes
		idx_target    = sorted(set(idx_target),key=idx_target.index) # non-duplicated sorted matched points idxes
		weight_M      = np.sum(G_class[:,idx_target],axis = 0) # Take the sum of every column
		weight_M      /= np.sum(weight_M) # Normalize
		label_matched = label_t[idx_target]
		for i_t in sorted(set(label_matched.tolist()),key=label_matched.tolist().index):
			idx_class_target          = np.where(label_matched == i_t)[0]
			T_label[i_s - 1, i_t - 1] = np.sum(weight_M[idx_class_target])
	return T_label


def get_subset_transform(T):
	""" Get the subset of the cell transform matrix T
	Input: l1 * l2 transform matrix T matrix
	Output: list of indexes, like [[[1,2],[2]], [[3, 4], [1]]]
		indicates classes 1, 2 in source transfers to class 2 in target
		classes 3,4 in source transfers to class 1 in target
	"""
	l1, l2      = T.shape
	idx_s_0     = set(range(l1)) # All the elements that are not assigned
	list_result = []
	flag_stop   = False # Flag indicating all the cells are classified
	while not flag_stop:
		# Initialization
		idx_s           = set([idx_s_0.pop()])
		idx_t           = set(np.nonzero(T[list(idx_s),:][0,:])[0])
		flag_add_finish = False   # If it is True then stops
		flag_s          = 's' # The next element to add. Options: s and t
		while not flag_add_finish:
			if flag_s == 's':
				idx_s_new = []
				for ele_t in list(idx_t):
					idx_s_new += (np.nonzero(T[:,ele_t])[0]).tolist()
				idx_s_new = set(idx_s_new)
				if idx_s_new == idx_s:
					flag_add_finish = True
				else:
					flag_s  = 't'
					idx_s   = idx_s_new
					idx_s_0 = idx_s_0 - idx_s
			else:
				idx_t_new = []
				for ele_s in list(idx_s):
					idx_t_new += (np.nonzero(T[ele_s,:])[0]).tolist()
				idx_t_new = set(idx_t_new)
				if idx_t_new == idx_t:
					flag_add_finish = True
				else:
					flag_s = 's'
					idx_t  = idx_t_new
		# Add 1 for the source and target classes
		list_result.append([[(s + 1) for s in list(idx_s)],[(t + 1) for t in list(idx_t)]])
		if len(idx_s_0) == 0:
			flag_stop = True
	# Add one for every class
	return list_result


def get_subset_gw(pts_s, pts_t, label_s, label_t, list_subset, options):
	""" Get the list of mapping between source pts and target pts. 
	"""
	list_gw = []
	list_pts_s = []
	list_pts_t = []
	for list_mapping in range(len(list_subset)):
		class_s = list_mapping[0]
		class_t = list_mapping[1]
		idx_s = np.array([])
		idx_t = np.array([])
	for i in class_s:
		idx_s = np.append(idx_s, np.where(label_s == i)[0])
	for i in class_t:
		idx_t = np.append(idx_t, np.where(label_t == i)[0])
	pts_s_sub = pts_s[idx_s.astype(int),:]
	pts_t_sub = pts_t[idx_t.astype(int),:]
	gw, pts_s_new, pts_t_new = get_2nd_mapping(pts_s_sub, pts_t_sub, options)
	list_gw.append(gw)
	list_pts_s.append(pts_s_sub)
	list_pts_t.append(pts_t_sub)
	return list_gw, list_pts_s, list_pts_t


def get_2nd_mapping(pts_s, pts_t, options):
	""" Get the second-order transform matrix. 
	Input: source/target samples, options
	Output: alignment matrix G, the subset of samples from source/target
	"""
	# The weight of considering first-order information
	weight_M = 0.5
	N_pts = 500
	if 'weight_M' in options:
		weight_M = options['weight_M']
	if 'N_pts' in options:
		N_pts = options['N_pts']
	if 'seed' in options:
		random.seed(options['seed'])
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
	# Compute first-order information
	M = ot.dist(pts_s_sub, pts_t_sub, metric='euclidean')
	M /= M.max()
	# Compute second-order information
	C_s = sp.spatial.distance.cdist(pts_s_sub, pts_s_sub)
	C_t = sp.spatial.distance.cdist(pts_t_sub, pts_t_sub)
	C_s /= C_s.max()
	C_t /= C_t.max()
	# Point-wise probability
	p_s = np.ones((len(pts_s_sub),)) / len(pts_s_sub)
	p_t = np.ones((len(pts_t_sub),)) / len(pts_t_sub)
	# Find the suitable epsilon for the computation
	gw, eps = get_gw_eps(C_s, C_t, p_s, p_t, weight_M=weight_M, M=M)
	# Just second-order information
	return gw, pts_s_sub, pts_t_sub


def align_2_imgs(I1, I2, options = dict()):
	""" Input: grayscale images I1 and I2
	"""
	# The number of pts from source and target
	if 'N_pts' in options:
		N_pts = options['N_pts']
	else:
		N_pts = 300
	# Random seed for points selection
	if 'seed' in options:
		random.seed(options['seed'])
	# The first-order mapping computation method
	# Choice: emd and sinkhorn
	if 'method_first' in options:
		method_first = options['method_first']
	else:
		method_first = 'emd'
	# ----- Compute alignments of each cell ------
	pts_s, label_s = get_seg_labels(I1, options) # source pts, labels 
	pts_t, label_t = get_seg_labels(I2, options) # target pts, labels
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
	M = ot.dist(pts_s_sub, pts_t_sub, metric='euclidean')
	M /= M.max()
	# Point-wise probability
	p_s = np.ones((len(pts_s_sub),)) / len(pts_s_sub)
	p_t = np.ones((len(pts_t_sub),)) / len(pts_t_sub)
	# Compute the transfer matrix
	if method_first == 'emd':
		G1 = ot.emd(p_s, p_t, M)
	else:
		eps = get_sinkhorn_eps(p_s, p_t, M)
		G1  = ot.sinkhorn(p_s, p_t, M, eps)
	# Get the cell-wise mapping
	T_label     = get_label_mapping(label_s_sub, label_t_sub, G1) # Label transform matrix
	# Visualize the mapping
	draw_mapping_class(pts_s_sub, pts_t_sub, G1, T_label, label_s_sub, label_t_sub)
	# For each small region, compute GW-DIST mapping with 2nd order information
	# Find the subset of the matching elements
	list_subset = get_subset_transform(T)
	# --- For each cell, compute alignment within the neighbouring range of it ---- 
	# For each component, finding the alignment within their neighbouring areas
	list_gw, list_pts_s, list_pts_t = get_subset_gw(pts_s, pts_t, label_s, label_t, list_subset, options)
	# Visualize mappings in the subset
	if 'visualize' in options and options['visualize'] == True:
		for i in range(len(list_gw)):
			options['img_name'] = 'trans_img_' + list2str(list_subset[i][0]) + '.png'
			draw_mapping(list_pts_s[i], list_pts_t[i], list_gw[i], options)
	# return pts_s_sub, pts_t_sub, G1


if __name__ == '__main__':
	options = dict()
	options['obj_white'] = True
	options['method_first'] = 'sinkhorn'
	options['lambda_first'] = 1e-3
	I1 = cv2.imread(PATH_LAYER_9 + 't000.tif_09.png', cv2.IMREAD_GRAYSCALE)
	I2 = cv2.imread(PATH_LAYER_9 + 't001.tif_09.png', cv2.IMREAD_GRAYSCALE)
