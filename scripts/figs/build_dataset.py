import os
import numpy as np
import glob
from lightjob.cli import load_db
from lightjob.db import SUCCESS
import h5py
import json
import hashlib
from tqdm import tqdm

def construct_data(job_folder, transform=lambda x:x):

    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)

    X = []
    for im in get_images(filenames):
        X.append([transform(im)])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))
    hash_vector = build_hash_vector(X)
    indices = unique_indices(hash_vector)
    X = X[indices]
    return X


def get_images(filenames):
    from skimage.io import imread
    for f in filenames:
        im = imread(f)
        yield im


def unique_indices(hm):
    K = {}
    for i, h in enumerate(hm):
        if h not in K:
            K[h] = i
    return K.values()

def hash_binary_vector(x):
    m = hashlib.md5()
    ss = str(x.flatten().tolist())
    m.update(ss)
    return m.hexdigest()

def build_hash_vector(X):
    hashes = []
    for i in range(X.shape[0]):
        h = hash_binary_vector(X[i])
        hashes.append(h)
    return hashes


if __name__ == '__main__':
    import random
    random.seed(42)
    f = h5py.File('figs/dataset.hdf5', 'w')
    db = load_db()
    jobs = list(db.jobs_with(state=SUCCESS, type='generation'))
    random.shuffle(jobs)
    jobs = [{'summary': 'iccc'}] + jobs
    print(len(jobs))
    dataset = f.create_dataset('X', (10000 * len(jobs), 784), maxshape=(None, 784), compression="gzip")
    i = 0
    ind = 0
    for j in tqdm(jobs):
        folder = "jobs/results/{}".format(j['summary'])
        X = construct_data(folder)
        dataset[i:i + len(X)] = X
        dataset.attrs[j['summary']] = json.dumps({'start': i, 'end': i + len(X)})
        i += len(X)
        f.attrs['nb'] = i
        print('nb examples so far : {}'.format(i))
    f.close()
