import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import click
import h5py
from tools.common import to_generation, preprocess_gen_data
import joblib
@click.command()
@click.option('--filename', default='exported_data/figs/obox_jobset75.npz', required=False)
@click.option('--classes', default=None, required=False)
@click.option('--jobset', default='jobset83', required=False)
@click.option('--nbmodels', default=1000, required=False)
@click.option('--field', default='stats.out_of_the_box_classification.m2.objectness', required=False)
@click.option('--skip', default=0, required=False)
def build(filename, classes, jobset, nbmodels, field, skip):
    from lightjob.cli import load_db
    from datakit.mnist import load
    ### REAL data
    if classes:
        classes = classes.split(',')
        classes = map(int, classes)
    data = load()
    Xreal, yreal = data['train']['X'], data['train']['y']
    yreal = yreal[:, 0]
    all_ind = []
    if classes:
        for cl in classes:
            ind = np.arange(len(yreal))[yreal == cl]
            all_ind.extend(ind.tolist())
        yreal = yreal[all_ind]
        Xreal = Xreal[all_ind]
    Xreal = (Xreal > 127) * 255.
    print(yreal[0:10])
    ## FAKE data
    db = load_db()
    jobs =db.jobs_with(state='success', where=jobset)
    jobs_gen = to_generation(jobs)
    indices = range(len(jobs))
    def key(i):
        val = db.get_value(jobs_gen[i], field, if_not_found=-float('inf'))
        if np.isnan(val):
            return -float('inf')
        else:
            return val
    indices = sorted(indices, key=lambda i:key(i))
    indices = indices[::-1]
    jobs_gen = [jobs_gen[i] for i in indices]
    filenames = ['jobs/results/{}/images.npz'.format(j['summary']) if j else None for j in jobs_gen]
    filenames = filter(lambda f:f, filenames)
    Xfake = []
    nb = 0 
    for i, f in enumerate(filenames):
        if i < skip:
            continue
        print(f)
        imgs = joblib.load(f).copy()
        imgs = preprocess_gen_data(imgs)
        Xfake.append(imgs)
        nb += 1
        if nb >= nbmodels:
            break
    Xfake = np.concatenate(Xfake, axis=0)
    fake_label = 10
    yfake = np.ones(len(Xfake)) * fake_label
    print('shape real : {}, shape fake : {}'.format(Xreal.shape, Xfake.shape))
    Xfake = Xfake * 255
    X = np.concatenate((Xreal, Xfake),  axis=0)
    y = np.concatenate((yreal, yfake), axis=0)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    print(Xreal.min(), Xreal.max(), Xfake.min(), Xfake.max())
    np.savez_compressed(filename, X=X, y=y)

if __name__ == '__main__':
    build()
