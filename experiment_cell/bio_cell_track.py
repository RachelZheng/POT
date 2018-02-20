#!/usr/bin/python3
# General cell-tracking functions
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from sklearn.preprocessing import normalize
import os, sys, glob, shutil, cv2, random, warnings, time, pickle

from config import *
sys.path.insert(0, PATH_OT)
import ot
from bio_cell_track_helper import *

def get_gw_eps(C1, C2, a, b, eps_0 = 1e-2, loss = 'square_loss', weight_M = 0, M = 0):
	""" Get the suitable epsilon value in the ot.gromov_wasserstein
	function.
	Input: all the inputs for 
	"""
	flag_stop = False
	eps       = eps_0
	if weight_M == 1:
		# Purely L1 normalization
		eps = get_sinkhorn_eps(a, b, M, eps_0)
		gw  = ot.sinkhorn(a, b, M, eps)
	else:
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
	cell_type = 'C3DL'
	if 'N_pixels' in options:
		N_pixels = options['N_pixels'] # C2DL: 20  C3DL: 100
	if 'thres' in options:
		thres = options['thres']
	if 'obj_white' in options:
		obj_white = options['obj_white']
	if 'cell_type' in options:
		cell_type = options['cell_type']  # C2DL/C3DL
	# ------ Computation ------
	if cell_type == 'C2DL':
		_,I = cv2.threshold(I,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
		if len(idx_class) > 0:
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
	Input: All the points and labels from source and target. List of small systems between these cells.
	Output: A list of gw distance mappings in these small systems, a list of pts from the source and
	another list from the target.
	"""
	list_gw = []
	list_pts_s = []
	list_pts_t = []
	N_pts = 700
	if 'N_pts' in options:
		N_pts = options['N_pts']
	for i in range(len(list_subset)):
		class_s = list_subset[i][0]
		class_t = list_subset[i][1]
		idx_s = np.array([])
		idx_t = np.array([])
		# Select a part of pts
		for j in class_s:
			idx_s = np.append(idx_s, np.where(label_s == j)[0]).astype(int)
		if len(idx_s) > N_pts:
			idx_s_select = np.random.randint(len(idx_s), size=N_pts)
			idx_s = idx_s[idx_s_select]
		for j in class_t:
			idx_t = np.append(idx_t, np.where(label_t == j)[0]).astype(int)
		if len(idx_t) > N_pts:
			idx_t_select = np.random.randint(len(idx_t), size=N_pts)
			idx_t = idx_t[idx_t_select]
		pts_s_sub = pts_s[idx_s,:]
		pts_t_sub = pts_t[idx_t,:]
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
	M /= max(M.max(), 1e-6)
	# Compute second-order information
	C_s = sp.spatial.distance.cdist(pts_s_sub, pts_s_sub)
	C_t = sp.spatial.distance.cdist(pts_t_sub, pts_t_sub)
	C_s /= max(C_s.max(), 1e-6)
	C_t /= max(C_t.max(), 1e-6)
	# Point-wise probability
	p_s = np.ones((len(pts_s_sub),)) / len(pts_s_sub)
	p_t = np.ones((len(pts_t_sub),)) / len(pts_t_sub)
	# Find the suitable epsilon for the computation
	gw, eps = get_gw_eps(C_s, C_t, p_s, p_t, weight_M=weight_M, M=M)
	# Just second-order information
	return gw, pts_s_sub, pts_t_sub

def get_1st_mapping(pts_s, pts_t, options):
	""" Get the first-order mapping from source pts to target pts
	"""
	# The number of pts from source and target
	if 'N_pts' in options:
		N_pts = options['N_pts']
	else:
		N_pts = 300
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
	M  /= max(M.max(), 1e-6)
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
	if 'visualize_first_order' in options and options['visualize_first_order'] == True:
		options['img_name'] = 'whole_img_1st_order.png'
		draw_mapping(pts_s_sub, pts_t_sub, G1, options)
	# ----- Get the cell-wise mapping -----
	T = get_label_mapping(label_s_sub, label_t_sub, G1) # Label transform matrix
	# For each subset, compute until it converge
	T_converged, list_subset = get_converged_T(T, pts_s, pts_t, label_s, label_t)
	return T_converged, list_subset


def get_converged_T(pts_s, pts_t, label_s, label_t, T):
	""" Iteratively compute sub-region transforms using the first-order information.
	Do it until the transform matrix converge. Then return the region transfer matrix T.
	"""
	list_subset = get_subset_transform(T)
	N_pts = 500 # #Of pts in the computation
	if 'N_pts' in options:
		N_pts = options['N_pts']
	flag_converged = False # The flag of the matrix T
	N_iter_total = 40 # The 
	N_iter = 0
	while not flag_converged and N_iter < N_iter_total:
		flag_converged = True
		for i in range(len(list_subset)):
			class_s = list_subset[i][0]
			class_t = list_subset[i][1]
			if len(class_s) > 1 and len(class_t) > 1:
				idx_s = np.array([])
				idx_t = np.array([])
				# Select a part of points 
				for j in class_s:
					idx_s = np.append(idx_s, np.where(label_s == j)[0]).astype(int)
				if len(idx_s) > N_pts:
					idx_s_select = np.random.randint(len(idx_s), size=N_pts)
					idx_s = idx_s[idx_s_select]
				for j in class_t:
					idx_t = np.append(idx_t, np.where(label_t == j)[0]).astype(int)
				if len(idx_t) > N_pts:
					idx_t_select = np.random.randint(len(idx_t), size=N_pts)
					idx_t = idx_t[idx_t_select]
				pts_s_sub = pts_s[idx_s,:]
				pts_t_sub = pts_t[idx_t,:]
				label_s_sub = label_s[idx_s]
				label_t_sub = label_t[idx_t]
				# Compute the first-order mapping.
				M   = ot.dist(pts_s_sub, pts_t_sub, metric='euclidean')
				M  /= (M.max() + 1e-6)
				# Point-wise probability
				p_s = np.ones((len(pts_s_sub),)) / len(pts_s_sub)
				p_t = np.ones((len(pts_t_sub),)) / len(pts_t_sub)
				eps = get_sinkhorn_eps(p_s, p_t, M)
				G1  = ot.sinkhorn(p_s, p_t, M, eps)
				T_sub = get_label_mapping(label_s_sub, label_t_sub, G1)
				class_s = [(j - 1) for j in class_s]
				class_t = [(j - 1) for j in class_t]
				T_sub_classes = T_sub[class_s, :]
				T_sub_classes = filter_small_transform_T(T_sub_classes[:, class_t])
				T_classes = T[class_s, :]
				T_classes = filter_small_transform_T(T_classes[:, class_t])
				if not np.array_equal(T_sub_classes,T_classes):
					# If these two matrixes are not equal, then we should re-run the 
					flag_converged = False
					# Update the matrix T
					for idx_i in class_s:
						for idx_j in class_t:
							T[idx_i, idx_j] = T_sub[idx_i, idx_j]
			list_subset = get_subset_transform(T)
		N_iter += 1
		print('At iteration ' + str(N_iter) + '\n')
	return T, list_subset


def filter_small_transform_T(T):
	""" Filter some transfers that less than a threshold value
	"""
	thres = 0.05
	T[T < thres] = 0
	T = normalize(T, axis=1, norm='l1')
	return T


def align_2_imgs(I1, I2, options = dict()):
	""" Align the cells between I1 and I2
	Input: grayscale images I1 and I2
	"""
	# ----- Compute alignments of each cell ------
	pts_s, label_s = get_seg_labels(I1, options) # source pts, labels 
	pts_t, label_t = get_seg_labels(I2, options) # target pts, labels	
	T, list_subset = get_1st_mapping(pts_s, pts_t, options)
	label_s     = pts_s[:,-1]
	label_t     = pts_t[:,-1]
	pts_s       = pts_s[:,:-1]
	pts_t       = pts_t[:,:-1]
	list_gw, list_pts_s, list_pts_t = get_subset_gw(pts_s, pts_t, label_s, label_t, list_subset, options)
	# Save the list of elements
	pickle.dump((pts_s, pts_t, list_gw, list_pts_s, list_pts_t, list_subset), open(options['img_folder'] + 'intermediate.p','wb'))
	# Visualize mappings in the subset
	if 'visualize_second_order' in options and options['visualize_second_order'] == True:
		print('Visualize all the second-order mappings ...')
		if 'img_name_source' in options:
			img_name_prefix = options['img_name_source']
		else:
			img_name_prefix = 'trans_img_'
		for i in range(len(list_gw)):
			options['img_name'] = img_name_prefix + list2str(list_subset[i][0]) + '.png'
			draw_mapping(list_pts_s[i], list_pts_t[i], list_gw[i], options)
		print('Done. \n')


if __name__ == '__main__':
	options                 = dict()
	options['obj_white']    = True
	options['method_first'] = 'sinkhorn'
	options['visualize']    = True
	options['img_folder']   = '/Users/xvz5220-admin/Desktop/output/'
	options['img_name_source'] = 't000.tif_09.png' # The source image name of the cell
	I1 = cv2.imread(PATH_LAYER_9 + 't000.tif_09.png', cv2.IMREAD_GRAYSCALE)
	I2 = cv2.imread(PATH_LAYER_9 + 't001.tif_09.png', cv2.IMREAD_GRAYSCALE)
	align_2_imgs(I1, I2, options)

