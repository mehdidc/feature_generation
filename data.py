import numpy as np
import os
from skimage.io import imread
from lasagnekit.datasets.helpers import split

def load_data(dataset="digits", 
              w=None, h=None, 
              include_test=False,
              batch_size=128,
              mode='random', **kw):
    if dataset == 'random':
        c = 1
        w, h = 28, 28
        prob = 0.1
        class Random(object):

            def __init__(self, shape):
                self.shape = shape
                self.img_dim = shape[1:]
            def load(self):
                self.X = np.random.uniform(size=self.shape) <= prob
                self.X = self.X.reshape((self.shape[0], -1))
                self.X = self.X.astype(np.float32)

        data = Random((batch_size, c, w, h))
        data.load()
    if dataset == 'fonts_big':
        import h5py
        from lasagnekit.datasets.manual import Manual
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.transformed import Transformed
        if w is None and h is None:
            w, h = 64, 64
        c = 1
        DATA_PATH = os.getenv('DATA_PATH')
        filename = os.path.join(DATA_PATH, 'fonts_big', 'fonts.hdf5')
        hf = h5py.File(filename)
        X = hf['fonts']
        data = Manual(X=X)
        data = SubSampled(data, batch_size, mode='batch', shuffle=False)
        data = Transformed(data, lambda X: 1 - X[:, 30, :, :]/255., per_example=False)
        data.img_dim = (64, 64)
        data = Rescaled(data, (w, h))
        data.load()
        print(data.X.shape)

    if dataset == 'chinese_icdar':
         import h5py
         from lasagnekit.datasets.manual import Manual
         from lasagnekit.datasets.subsampled import SubSampled
         from lasagnekit.datasets.rescaled import Rescaled
         from lasagnekit.datasets.transformed import Transformed
         if w is None and h is None:
             w, h = 64, 64
         c = 1
         DATA_PATH = os.getenv('DATA_PATH')
         filename = os.path.join(DATA_PATH, 'chinese', 'HWDB1.1subset.hdf5')
         hf = h5py.File(filename)
         X = hf['trn/x']
         data = Manual(X=X)
         data = SubSampled(data, batch_size, mode='batch', shuffle=False)
         data = Transformed(data, lambda X: 1 - X/255., per_example=False)
         data = Transformed(data, lambda X: X[:, 0], per_example=False)
         data.img_dim = (64, 64)
         data = Rescaled(data, (w, h))
         data = Transformed(data, lambda X: X.astype(np.float32), per_example=False)
    if dataset == "digits":
        from lasagnekit.datasets.mnist import MNIST
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.helpers import load_once
        from lasagnekit.datasets.rescaled import Rescaled
        w, h = 28, 28
        c = 1
        if include_test:
            which = 'train'
        else:
            which = 'all'
        data = load_once(MNIST)(which=which)
        data.load()
        if 'train_classes' in kw:
            included = np.zeros((len(data.X),)).astype(np.bool)
            for cl in kw['train_classes']:
                included[data.y == cl] = True
            data.X = data.X[included]
            data.y = data.y[included]

        w, h = data.img_dim
        if mode == 'random':
            data_train_whole = data
            data = SubSampled(data, batch_size)
            data.train = data_train_whole
        elif mode == 'minibatch':
            data.train = data
        else:
            raise('Unknown mode {}'.format(mode))
        if include_test:
            data.test = MNIST(which='test')
            data.test.load()

            if 'test_classes' in kw:
                included = np.zeros((len(data.test.X),)).astype(np.bool)
                for cl in kw['test_classes']:
                    included[data.test.y == cl] = True
                data.test.X = data.test.X[included]
                data.test.y = data.test.y[included]

    if dataset == "olivetti":
        from sklearn.datasets import fetch_olivetti_faces
        from lasagnekit.datasets.manual import Manual
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.subsampled import SubSampled
        if w is None and h is None:
            w, h = 64, 64
        c = 1
        data = fetch_olivetti_faces()
        X = data['images']
        X = X.astype(np.float32)
        X = 1 - X
        data = Manual(X.reshape((X.shape[0], -1)), y=data['target'])
        data.img_dim = (w, h)
        data = Rescaled(data, (w, h))
        data = SubSampled(data, batch_size)

    if dataset == "notdigits":
        from lasagnekit.datasets.notmnist import NotMNIST
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.helpers import load_once
        w, h = 28, 28
        c = 1
        data = load_once(NotMNIST)()
        data.load()
        w, h = data.img_dim
        data = SubSampled(data, batch_size)

    elif dataset == "flaticon":
        from lasagnekit.datasets.flaticon import FlatIcon
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.helpers import load_once
        from lasagnekit.datasets.subsampled import SubSampled

        if w is None and h is None:
            w, h = 64, 64
        c = 1

        mode = kw.get("mode", "all")
        mode = "all"
        include_test = True
        if mode == "all":
            nb = 38698
        else:
            nb = batch_size
        data = load_once(FlatIcon)(size=(w, h), nb=nb, mode=mode)
        data.load()
        data.y = None
        def preprocess(X):
            X = X[:, :, :, 0]
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        if include_test:
            data_train, data_test = split(data, test_size=0.15, random_state=42)
        
        data = SubSampled(data_train, batch_size)      
        data.load()  
        data.train = data_train
        data.test = data_test
        
        print(data.X.shape)

    elif dataset == "fonts":
        from lasagnekit.datasets.fonts import Fonts
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.helpers import load_once
        if w is None and h is None:
            w, h = 64, 64
        c = 1
        data = load_once(Fonts)(kind='all_64')
        data.load()
        if include_test:
            data_train, data_test = split(data, test_size=0.15, random_state=42)
        else:
            data_train = data

        if mode == 'random':
            data_train = Rescaled(data_train, (w, h))
            data_train.load()
            data = SubSampled(data_train, batch_size)
            data.load()
            data.train = data_train
            if include_test:
                data.test = Rescaled(data_test, (w, h))
                data.test.load()
        else:
            data_train = Rescaled(data_train, (w, h))
            data_train.load()
            data = data_train
            data.train = data_train
            if include_test:
                data.test = Rescaled(data_test, (w, h))
                data.test.load()

    elif dataset == "svhn":
        from lasagnekit.datasets.svhn import SVHN
        from lasagnekit.datasets.helpers import load_once
        from lasagnekit.datasets.rescaled import Rescaled
        w, h = 28, 28
        c = 3
        data = SVHN(which='train', size=(w, h), nb=batch_size)
        data.load()
        print(data.X.shape)

        def preprocess(X):
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)
    elif dataset == 'stl':
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.stl import STL
        from lasagnekit.datasets.helpers import load_once
        if w is None and h is None:
            w, h = 64, 64
        c = 3
        data = load_once(STL)('unlabeled')
        data.load()
        print(data.X.shape)

        def preprocess(X):
            shape = (X.shape[0],) + (w, h, c)
            X = X / 255.
            X = X.reshape(shape)
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            return X
        data = SubSampled(data, batch_size)
        data = Rescaled(data, (w, h))
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)

    elif dataset == 'chairs':
        from lasagnekit.datasets.chairs import Chairs
        from lasagnekit.datasets.transformed import Transformed
        if w is None and h is None:
            w, h = 64, 64
        c = 3
        data = Chairs(size=(w, h), nb=batch_size)
        data.load()
        print(data.X.shape)

        def preprocess(X):
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)
    elif dataset == 'icons':
        from lasagnekit.datasets.imagecollection import ImageCollection
        from lasagnekit.datasets.transformed import Transformed
        if w is None and h is None:
            w, h = 32, 32
        c = 3
        folder = "{}/icons".format(os.getenv("DATA_PATH"))
        data = ImageCollection(size=(w, h), nb=batch_size, folder=folder)
        data.load()
        print(data.X.shape)

        def preprocess(X):
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)

    elif dataset == 'lfw':
        from lasagnekit.datasets.skimagecollection import ImageCollection
        from skimage.io import imread_collection
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.rescaled import Rescaled
        if w is None and h is None:
            w, h = 64, 64
        c = 3
        folder = "{}/lfw/img/**/*.jpg".format(os.getenv("DATA_PATH"))
        collection = imread_collection(folder)
        indices = np.arange(len(collection))
        np.random.shuffle(indices)
        data = ImageCollection(collection,
                               indices=indices,
                               batch_size=batch_size)
        data.load()
        data = Rescaled(data, (w, h))

        def preprocess(X):
            shape = (X.shape[0],) + (w, h, c)
            X = X.reshape(shape)
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            X = X / 255.
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)

    elif dataset == 'lfwgrayscale':
        from lasagnekit.datasets.skimagecollection import ImageCollection
        from lasagnekit.datasets.transformed import Transformed
        from skimage.io import imread_collection
        from lasagnekit.datasets.rescaled import Rescaled
        if w is None and h is None:
            w, h = 64, 64
        c = 1
        folder = "{}/lfw/img/**/*.jpg".format(os.getenv("DATA_PATH"))
        collection = imread_collection(folder)
        indices = np.arange(len(collection))
        np.random.shuffle(indices)
        data = ImageCollection(collection,
                               indices=indices,
                               batch_size=batch_size)
        data.load()
        data = Rescaled(data, (w, h))

        def preprocess(X):
            shape = (X.shape[0],) + (w, h, 3)
            X = X.reshape(shape)
            X = X.transpose((0, 3, 1, 2))
            X = X[:, 0] * 0.21 + X[:, 1] * 0.72 + X[:, 2] * 0.07
            X = X.reshape((X.shape[0], -1))
            X = X / 255.
            X = 1 - X
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)

    elif dataset == 'kanji':
        from lasagnekit.datasets.skimagecollection import ImageCollection
        from skimage.io import imread_collection
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.rescaled import Rescaled
        if w is None and h is None:
            w, h = 64, 64
        c = 1
        name = "cleanpngsmall"
        folder = "{}/kanji/{}/*.png".format(os.getenv("DATA_PATH"), name)
        collection = imread_collection(folder)
        indices = np.arange(len(collection))
        np.random.shuffle(indices)
        data = ImageCollection(collection,
                               indices=indices,
                               batch_size=batch_size)
        data.load()
        print(data.X.shape)
        data = Rescaled(data, (w, h))
        data.load()
        print(data.X.shape)

        def preprocess(X):
            X = X.reshape((X.shape[0],  w, h, 4))
            X = X[:, :, :, 0]
            X = X / 255.
            X = 1 - X
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)

    elif dataset == "B":
        from lasagnekit.datasets.manual import Manual
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.infinite_image_dataset import InfiniteImageDataset
        img = imread("B_small.png")  # the dataset is based on one image
        img = img[:, :, 0]
        w, h = img.shape[0], img.shape[1]
        c = 1  # nb colors
        X = [img.tolist()]
        X = np.array(X).astype(np.float32)
        X /= 255.
        X = 1 - X
        data = Manual(X.reshape((X.shape[0], -1)))
        data.img_dim = (w, h)
        w, h = 28, 28
        data = Rescaled(data, (w, h))
        data = SubSampled(data, batch_size)
        data = InfiniteImageDataset(
           data,
           translation_range=(-5, 5),
           rotation_range=(-90, 90),
           zoom_range=(1, 1.1),
        )
    elif dataset == "myth":
        from lasagnekit.datasets.manual import Manual
        from skimage.io import imread
        from skimage.transform import resize
        im = imread("myth/wheel.png")
        w, h = 64, 64
        c = 1
        if len(im.shape) >= 3:
            im = im[:, :, 0]
        im = resize(im, (w, h))
        im = im[None, None, :, :]
        im = im / 255.
        im = 1. - im
        im = im.astype(np.float32)
        data = Manual(im)
        data.img_dim = (w, h)

    data.load()
    data.w = w
    data.h = h
    data.c = c
    return data
