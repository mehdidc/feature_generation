import numpy as np
import os
from skimage.io import imread


def load_data(dataset="digits", w=None, h=None, include_test=False, batch_size=128, mode='random', **kw):
    nbl, nbc = 10, 10

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
        w, h = data.img_dim
        if mode == 'random':
            data = SubSampled(data, batch_size)
        if include_test:
            data.test = MNIST(which='test')
            data.test.load()

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
        if mode == "all":
            nb = 38698
        else:
            nb = batch_size
        data = load_once(FlatIcon)(size=(w, h), nb=nb, mode=mode)
        data.load()
        def preprocess(X):
            X = X[:, :, :, 0]
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)
        if mode == "all":
            data = SubSampled(data, batch_size)
        data.load()
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
        data = SubSampled(data, batch_size)
        data = Rescaled(data, (w, h))
        data.load()
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
    return data, w, h, c, nbl, nbc
