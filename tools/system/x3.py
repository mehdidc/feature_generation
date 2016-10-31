import sys
import os
sys.path.append('/home/mcherti/work/code/feature_generation')
from lightjob.cli import load_db
import joblib
import numpy as np
from joblib import Parallel, delayed
import copy
import pandas as pd
from tools.common import to_generation, disp_grid
from skimage.io import imread, imsave

db = load_db()
jobs = db.jobs_with(state='success', type='generation')
def preprocess_gen_data(data):
    if len(data.shape) == 5:
        data = data[:, -1] # last time step images
    if len(data.shape) == 3:
        data = data[:, np.newaxis]
    return data


for j in jobs:
    folder = 'jobs/results/{}'.format(j['summary'])
    data = joblib.load(os.path.join(folder, 'images.npz'))
    data = preprocess_gen_data(data)
    try:
        data = data / float(data.max())
    except ValueError:
        f = os.path.join(folder, 'final1000.png')
        if os.path.exists(f):
            im =imread(f)
            assert int(im.shape[0]/float(28))*28==im.shape[0]
            assert int(im.shape[1]/float(28))*28==im.shape[1]
            im = im.reshape((im.shape[0]/28, 28, im.shape[1]/28, 28))
            im = im.transpose((0, 2, 1, 3))
            im = im.reshape((im.shape[0]*im.shape[1], 28, 28))
            a = disp_grid(im[:, None, :, :])
            imsave('a.png', a)
            new_data = np.array(im)
            print(data.shape, new_data.shape)
            joblib.dump(new_data, folder + '/images.npz', compress=9)
 
