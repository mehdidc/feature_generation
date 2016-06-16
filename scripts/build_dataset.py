import os
import numpy as np
import glob
from lightjob.cli import load_db
from lightjob.db import SUCCESS
import h5py

def construct_data(job_folder, hash_matrix, transform=lambda x:x):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if len(filenames) == 0:
        return None

    X = []
    for im in get_images(filenames):
        X.append([transform(im)])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))
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

if __name__ == '__main__':
    import random
    random.seed(42)
    f = h5py.File('out.hdf5', 'w')
    db = load_db()
    jobs = list(db.jobs_with(state=SUCCESS, type='generation'))
    random.shuffle(jobs)
    print(len(jobs))
    dataset = f.create_dataset('X', (10000 * len(jobs), 784), maxshape=(None, 784), compression="gzip")
    i = 0
    for j in jobs:
        folder = "jobs/results/{}".format(j['summary'])
        hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
        hash_matrix = np.load(hash_matrix_filename)
        X = construct_data(folder, hash_matrix)
        dataset[i:i + len(X)] = X
        i += len(X)
        dataset.attrs['nb'] = i
        print('nb examples so far : {}'.format(i))
    f.close()
