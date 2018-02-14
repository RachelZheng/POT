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

def get_one_layer_img():
	""" Move one layer images to the target folder
	""" 
	os.chdir(PATH_RAW)
	for img in glob.glob('*_09.png'):
		shutil.copy(img, PATH_LAYER_9 + img)
		convert2black_white_img(PATH_LAYER_9 + img)


def convert2black_white_img(image_name, options = dict()):
	""" Convert imgs to black and white
	"""
	img = cv2.imread(image_name,0)
	if 'path_output' in options:
		cv2.imwrite(image_name, options['path_output'] + img)


def get_gw_eps(eps_0, C1, C2, a, b, loss = 'square_loss', weight_M = 0, M = 0):
	""" Get the suitable epsilon value in the ot.gromov_wasserstein
	function.
	Input: all the inputs for 
	"""
	flag_stop = False
	eps       = eps_0
	if weight_M == 1:
		# degenerate to the order-1 transform
		# Gs = ot.sinkhorn(a, b, M, lambd)
		gw = ot.emd(a, b, M)
	while not flag_stop:
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")
			gw = ot.gromov_wasserstein(C1, C2, a, b, loss, epsilon=eps, weight_M=weight_M, M=M)
			if len(w) == 0:
				eps/= 2
			else:
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


def get_img_labels(I, options = dict()):
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
	# ----- Compute alignments ------
	pts_s, label_s = get_img_labels(I1, options) # source labels 
	pts_t, label_t = get_img_labels(I2, options) # target labels
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
	# Compute first-order information
	M = ot.dist(pts_s_sub[:,:-1], pts_t_sub[:,:-1], metric='euclidean')
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
	T_label = get_label_mapping(pts_s_sub[:,-1], pts_t_sub[:,-1], G1) # Label transform matrix
	# Visualize the mapping
	draw_mapping_class(pts_s_sub[:,:-1], pts_t_sub[:,:-1], G1, T_label, pts_s_sub[:,-1], pts_t_sub[:,-1])
	# For each small region, compute GW-DIST mapping with 2nd order information
	return pts_s_sub, pts_t_sub, G1


def draw_mapping(xs, xt, Gs, options = dict()):
	pl.figure()
	ot.plot.plot2D_samples_mat(xs, xt, Gs, color=[.5, .5, 1])
	pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
	pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
	pl.legend(loc=0)
	if 'img_name' in options:
		img_name = options['img_name']
	else:
		img_name = 'trans.png'
	pl.savefig('/Users/xvz5220-admin/Desktop/' + img_name)


def draw_mapping_class(xs, xt, Gs, T, label_s, label_t):
	options = dict()
	for i in range(1, label_s.max()):
		idx_s   = np.where(label_s == i)[0] # index of the pts from the source
		class_t = np.nonzero(T[i - 1,:])[0].tolist() # index of the classes from the target
		class_t = [i + 1 for i in class_t]
		idx_t = np.array([],dtype = int)
		for j in class_t:
			idx_t = np.append(idx_t, np.where(label_t == j)[0]) # index of the pts from the target
		options['img_name'] = 'trans' + '%02d'%i + '.png'
		G_sub = Gs[idx_s, :]
		G_sub = G_sub[:,idx_t]
		xs_sub = xs[idx_s,:]
		xt_sub = xt[idx_t,:]
		draw_mapping(xs_sub, xt_sub, G_sub, options)



def get_label_mapping(label_s, label_t, Gs):
	""" Get the mapping between source and target cells
	Input: n1 * 1 labels for source, n2 * 1 labels for target, transfer matrix
	Output: l1 * l2 matrix with relative transformation, which is row-normalized
	"""
	l_s          = label_s.max()
	l_t          = label_t.max()
	G            = Gs / Gs.max()
	T_label      = np.zeros((l_s, l_t))
	thres        = 1e-6
	G[G < thres] = 0 
	for i_s in range(1,l_s):
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

if __name__ == '__main__':
options = dict()
options['obj_white'] = True
options['method_first'] = 'sinkhorn'
options['lambda_first'] = 1e-3
I1 = cv2.imread(PATH_LAYER_9 + 't000.tif_09.png', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread(PATH_LAYER_9 + 't001.tif_09.png', cv2.IMREAD_GRAYSCALE)
	"""
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
	eps_0 = 1e-3
	gw, eps = get_gw_eps(eps_0, C_s, C_t, p_s, p_t) # eps value
	"""

	"""
	from bio_cell_track import *

	"""

