#!/usr/bin/python2
# Test for cell_tracking challenges

PATH_TO_TIFF = '/Users/xvz5220-admin/Documents/data/cell_tracking_challenge/Fluo-C3DL-MDA231/'

import sys, os, glob, imghdr
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from shutil import copyfile, move
from PIL import Image


def tiff2png(dir_tiff):
	dir_back = os.getcwd()
	os.chdir(dir_tiff)
	for img in glob.glob('*.tif'):
		pass



def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()
    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')
    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])
    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])
    return ax_img, ax_hist, ax_cdf

