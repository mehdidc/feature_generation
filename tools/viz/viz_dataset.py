#!/usr/bin/env python
import sys, os
import numpy as np
sys.path.append(".")
import matplotlib as mpl
mpl.use('Agg') # NOQA
import sys
from lasagnekit.misc.plot_weights import tile_raster_images, dispims_color
import matplotlib.pyplot as plt
from data import load_data
from skimage.io import imsave
dataset = sys.argv[1]
nb = 30 * 8
#batch_size = 1024
data = load_data(dataset, batch_size=nb, mode='random')
data.load()
X = data.X[np.random.randint(0, len(data.X), size=nb)]
if len(data.img_dim) == 3:
    if len(X.shape) == 2:
        shape = (X.shape[0],) + (3,) + data.img_dim[0:2]
        print(shape)
        X = X.reshape(shape)
        X = X.transpose((0, 2, 3, 1))
        print(X.shape)
    img = dispims_color(X, border=1, bordercolor=(0.3, 0, 0))
else:
    X = X.reshape((X.shape[0],) + data.img_dim + (1,))
    X = X * np.ones((1, 1, 1, 3))
    X = 1 - X
    img = dispims_color(X, border=1, bordercolor=(0.3, 0, 0), shape=(30, 8))
imsave(dataset+'.png', img)
