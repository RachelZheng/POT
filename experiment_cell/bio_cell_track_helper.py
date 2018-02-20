PATH_DATA = '/Users/xvz5220-admin/Dropbox/cell_tracking_data/data_output/'
PATH_RAW = PATH_DATA + '01/'
PATH_LAYER_9 = PATH_DATA + '01_09/'

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import os, sys, glob, shutil, cv2, random, warnings, time

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


def list2str(lst):
	""" From numerical list elements to string
	"""
	result = ''
	for i in lst:
		result = result + str(i) + '_'
	return result


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
	if 'img_folder' in options:
		img_folder = options['img_folder']
	else:
		img_folder = '/Users/xvz5220-admin/Desktop/'
	pl.savefig(img_folder + img_name)


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


