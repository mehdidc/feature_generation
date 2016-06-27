import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from lightjob.cli import load_db
from tqdm import tqdm
from skimage.io import imread, imsave
from lasagnekit.misc.plot_weights import dispims_color
import numpy as np
import glob
import random

random.seed(2)

w, h = 3, 3
db = load_db()

J = db.jobs_with(state='success', type='generation')
for j in tqdm(J):
    id_ = j['summary']
    folder = "jobs/results/{}".format(id_)
    filenames = glob.glob(os.path.join(folder, 'final', '*.png'))
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = filenames[0:w*h]

    X = []
    for filename in filenames:
        X.append(imread(filename))
    X = np.array(X)
    X = X[:, :, :, np.newaxis]
    X = X * np.ones((1, 1, 1, 3))
    img = dispims_color(X)
    imsave('figs/mini/{}.png'.format(id_), img)
