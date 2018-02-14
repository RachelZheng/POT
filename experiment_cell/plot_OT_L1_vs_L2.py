# -*- coding: utf-8 -*-
"""
==========================================
2D Optimal transport for different metrics
==========================================

2D OT on empirical distributio with different gound metric.

Stole the figure idea from Fig. 1 and 2 in
https://arxiv.org/pdf/1706.07650.pdf
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import sys
sys.path.insert(0,'/Users/xvz5220-admin/Dropbox/gromov_wasserstein_dist/POT/')
import ot

##############################################################################
# Dataset 1 : uniform sampling
# ----------------------------

n = 20  # nb samples
xs = np.zeros((n, 2))
xs[:, 0] = np.arange(n) + 1
xs[:, 1] = (np.arange(n) + 1) * -0.001  # to make it strictly convex...

xt = np.zeros((n, 2))
xt[:, 1] = np.arange(n) + 1 - float(n)/2

a, b = ot.unif(n), ot.unif(n)  # uniform distribution on samples

# loss matrix
M1 = ot.dist(xs, xt, metric='euclidean')
M1 /= M1.max()

# loss matrix
M2 = ot.dist(xs, xt, metric='sqeuclidean')
M2 /= M2.max()

# loss matrix
Mp = np.sqrt(ot.dist(xs, xt, metric='euclidean'))
Mp /= Mp.max()

# Data
pl.figure(1, figsize=(7, 3))
pl.clf()
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.axis('equal')
pl.title('Source and traget distributions')


# Cost matrices
pl.figure(2, figsize=(7, 3))

pl.subplot(1, 3, 1)
pl.imshow(M1, interpolation='nearest')
pl.title('Euclidean cost')

pl.subplot(1, 3, 2)
pl.imshow(M2, interpolation='nearest')
pl.title('Squared Euclidean cost')

pl.subplot(1, 3, 3)
pl.imshow(Mp, interpolation='nearest')
pl.title('Sqrt Euclidean cost')
pl.tight_layout()

##############################################################################
# Dataset 1 : Plot OT Matrices
# ----------------------------

# --- Rachel add 0204 begin ----
C1 = sp.spatial.distance.cdist(xs, xs)
C2 = sp.spatial.distance.cdist(xt, xt)

C1 /= C1.max()
C2 /= C2.max()

gw = ot.gromov_wasserstein(C1, C2, a, b, 'square_loss', epsilon=5e-4, weight_M=0.3, M=M1)


#%% EMD
G1 = ot.emd(a, b, M1)
G2 = ot.emd(a, b, M2)
Gp = ot.emd(a, b, Mp)

# OT matrices
n_fig = 9
fig = pl.figure(n_fig, figsize=(25, 3))

pl.subplot(1, n_fig, 1)
ot.plot.plot2D_samples_mat(xs, xt, G1, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.axis('equal')
pl.title('OT Euclidean')

pl.subplot(1, n_fig, 2)
ot.plot.plot2D_samples_mat(xs, xt, G2, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.axis('equal')
pl.title('OT squared Euclidean')

for i in range(7):
	pl.subplot(1, n_fig, i + 3)
	gw = ot.gromov_wasserstein(C1, C2, a, b, 'square_loss', epsilon=5e-4/(1.0 - 0.15 * i), weight_M=(0.15 * i), M=M1)
	ot.plot.plot2D_samples_mat(xs, xt, gw, c=[.5, .5, 1])
	pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
	pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
	pl.axis('equal')
	pl.title('First-order constraints ' + str(round(i * 0.15,2)))
	pl.tight_layout()

pl.tight_layout()

pl.show()
