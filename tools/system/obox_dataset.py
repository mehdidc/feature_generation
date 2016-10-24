import numpy as np
import sys
sys.path.append('..')
import click
import h5py

@click.command()
@click.option('--fakedata', default='exported_data/figs/jobset75.hdf5', required=False)
@click.option('--filename', default='exported_data/figs/obox_jobset75.npz', required=False)
def build(fakedata, filename):
    from datakit.mnist import load
    data = load()
    Xreal, yreal = data['train']['X'], data['train']['y']
    yreal = yreal[:, 0]
    Xfake = h5py.File(fakedata)['X']
    indices = np.arange(len(Xfake))
    Xfake = Xfake[0:50000]
    indices = indices[0:10000]
    Xfake[np.isnan(Xfake)] = 0
    Xfake = Xfake[Xfake.sum(axis=(1, 2, 3)) > 0]
    print(Xfake.shape)
    fake_label = np.max(yreal) + 1
    print('fake label : {}'.format(fake_label))
    yfake = np.ones(len(Xfake)) * fake_label
    print(Xreal.shape, Xfake.shape)
    X = np.concatenate((Xreal, Xfake),  axis=0)
    y = np.concatenate((yreal, yfake), axis=0)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    print(X.min(), X.max())

    print(np.isnan(X).sum())
    np.savez_compressed(filename, X=X, y=y)

if __name__ == '__main__':
    build()
