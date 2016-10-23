import sys
import os
sys.path.append(os.path.dirname(__file__)+"/..")
import numpy as np
import glob
import h5py
import json
import hashlib
from tqdm import tqdm
import random
import joblib

import click

from lightjob.cli import load_db
from lightjob.db import SUCCESS
from common import preprocess_gen_data, resize_set, to_generation

@click.command()
@click.option('--filename', default='exported_data/figs/dataset.hdf5')
@click.option('--where', default=None)
@click.option('--width', default=28)
@click.option('--height', default=28)
@click.option('--color', default=1)
def build(filename, where, width, height, color):
    random.seed(42)
    
    f = h5py.File(filename, 'w')
    db = load_db()
    if where:
        jobs = db.jobs_with(state=SUCCESS, type='training', where=where)
        jobs = to_generation(jobs)
        jobs = filter(lambda j:j, jobs)
    else:
        jobs = db.jobs_with(state=SUCCESS, type='generation')
    
    jobs = list(jobs)
    random.shuffle(jobs)
    dataset = f.create_dataset(
            'X', 
            (10000 * len(jobs), color, height, width), 
            maxshape=(None, color, height, width), 
            compression="gzip")
    i = 0
    ind = 0
    for j in tqdm(jobs):
        folder = "jobs/results/{}".format(j['summary'])
        X = joblib.load(os.path.join(folder, 'images.npz'))
        X = preprocess_gen_data(X)
        X = X / (float(X.max()) if len(X) else 1)
        if X.shape[0] and (height != X.shape[2] or width != X.shape[3]):
            X = resize_set(X, (height, width))
        dataset[i:i + len(X)] = X
        dataset.attrs[j['summary']] = json.dumps({'start': i, 'end': i + len(X)})
        i += len(X)
        f.attrs['nb'] = i
    f.close()
    print('Total nb of examples : {}'.format(i))
 
if __name__ == '__main__':
    build()
