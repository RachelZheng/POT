import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import os, sys, glob, shutil, cv2, random, warnings, time

from config import *
sys.path.insert(0, PATH_OT)
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
	""" Draw mapping from xs to Gs
	"""
	pl.figure()
	xs_move = xs.astype(float) # Add a small move in the x-axis
	xs_move[:,0] -= 0.1
	ot.plot.plot2D_samples_mat(xs_move, xt, Gs, color=[.5, .5, 1])
	pl.plot(xs_move[:, 0], xs_move[:, 1], '+b', label='Source samples') 
	pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
	pl.legend(loc=0)
	if 'img_name' in options:
		img_name = options['img_name']
	else:
		img_name = 'trans.png'
	if 'img_folder' in options:
		img_folder = options['img_folder']
	else:
		img_folder = os.getcwd()
	pl.savefig(img_folder + img_name)
	pl.close()


def draw_mapping_list(list_xs, list_xt, list_Gs, list_subset, options = dict()):
	""" Draw the mapping of the list of sample pts
	"""
	# Plot a general image first
	pl.figure()
	for i in range(len(list_xs)):
		xs = list_xs[i]
		xt = list_xt[i]
		Gs = list_Gs[i]
		ot.plot.plot2D_samples_mat(xs, xt, Gs, color=[.5, .5, 1])
		pl.plot(xs[:, 0], xs[:, 1], '+b')
		pl.plot(xt[:, 0], xt[:, 1], 'xr')
	pl.savefig(options['img_folder'] + 'whole_img_2nd_order_w_' + str(options['weight_M']) + '.png')
	pl.close()
	# Plot the label-wise mappings
	for i in range(len(list_xs)):
		xs = list_xs[i]
		xt = list_xt[i]
		Gs = list_Gs[i]
		options['img_name'] = 'img_part_' + list2str(list_subset[i][0]) + '_w_' + str(options['weight_M']) + '.png'
		draw_mapping(xs, xt, Gs, options)



