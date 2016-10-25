import numpy as np
import sys
sys.path.append('..')
import click
import h5py

@click.command()
@click.option('--fakedata', default='exported_data/figs/jobset75.hdf5', required=False)
@click.option('--filename', default='exported_data/figs/obox_jobset75.npz', required=False)
@click.option('--classes', default=None, required=False)
@click.option('--nbfake', default=-1, required=False)
def build(fakedata, filename, classes, nbfake):
    if classes:
        classes = classes.split(',')
        classes = map(int, classes)

    from datakit.mnist import load
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
    Xfake = h5py.File(fakedata)['X']
    if nbfake != -1:
        Xfake = Xfake[0:nbfake * 20]
    Xfake[np.isnan(Xfake)] = 0
    Xfake = Xfake[Xfake.sum(axis=(1, 2, 3)) > 0]
    indices = np.arange(len(Xfake))
    Xfake = Xfake[indices]
    
    if nbfake != -1:
        Xfake = Xfake[0:nbfake]
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
