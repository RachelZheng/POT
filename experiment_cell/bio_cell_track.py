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

def get_img_labels(I, options = dict()):
	""" Get image representation pts, distances matrix and 
	Input: grayscale images I
	Output: sample pts X, the labeled image labels_filtered
	"""
	# ------ Options ------ 
	# The object in the image is white(T) or black(F)
	thres = 120   # Threshold of cutting cells
	N_pixels = 100 # minimal # of pixels of one cell
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
	sizes = stats[:, -1]
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
	ls = labels_filtered.reshape(1,-1) # labels of pts
	ls = ls[:, np.all(ls>0,axis=0)].reshape(-1,1)
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
	# ----- Compute alignments ------
	pts_s, label_s = get_img_labels(I1, options) # source labels 
	pts_t, label_t = get_img_labels(I2, options) # target labels
	if len(pts_s) > N_pts:
		

	# TODO BEGIN:
	# GET RANDOM SAMPLES, USE 1ST ORDER INFORMATION TO TRANSFORM 
	# point: ensure every component 
	# GET THE TRANSFORM MATRIX
	# RESAMPLE, GET THE 2ND ORDER INFORMATION TO REFINE THE TRANSFORM
	# TODO ENDS







	# Get all the points from the image
	if obj_white:
		pts_s = np.nonzero(I1 > 100)
		pts_t = np.nonzero(I2 > 100)
	else:
		pts_s = np.nonzero(I1 < 100)
		pts_t = np.nonzero(I2 < 100)
	# Suppose the dataset is 2D, rearrange the data
	pts_s = np.concatenate((pts_s[0].reshape((-1,1)),pts_s[1].reshape((-1,1))),axis = 1)
	pts_t = np.concatenate((pts_t[0].reshape((-1,1)),pts_t[1].reshape((-1,1))),axis = 1)
	if 'seed' in options:
		random.seed(options['seed'])
	# If we have far more data points, we would random sample some of them
	if len(pts_s) > N_pts:
		pts_s_sub = random.sample(pts_s, N_pts)
	else:
		pts_s_sub = pts_s
	if len(pts_t) > N_pts:
		pts_t_sub = random.sample(pts_t, N_pts)
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
	eps_0 = 1e-3
	gw, eps = get_gw_eps(eps_0, C_s, C_t, p_s, p_t) # eps value
	










if __name__ == '__main__':
	options = dict()
	options['obj_white'] = True

