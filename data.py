import numpy as np
import os
from skimage.io import imread
from lasagnekit.datasets.helpers import split
from helpers import DataGen


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
    elif dataset == 'omniglot':
        from lasagnekit.datasets.omniglot import Omniglot
        from lasagnekit.datasets.transformed import Transformed
        if w is None and h is None:
            w = 28
            h = 28
        c = 1
        data = Omniglot(size=(w, h), nb=batch_size)

        def preprocess(X):
            X = 1 - X
            X = X[:, :, :, 0]
            return X
        data = Transformed(data, preprocess, per_example=False)
        data.load()
        print(data.X.shape)

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

    if dataset == 'chinese_icdar_big':
        import h5py
        from lasagnekit.datasets.manual import Manual
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.transformed import Transformed
        if w is None and h is None:
            w, h = 64, 64
        c = 1
        DATA_PATH = os.getenv('DATA_PATH')
        filename = os.path.join(DATA_PATH, 'chinese', 'data.hdf5')
        hf = h5py.File(filename)
        X = hf['tst/bitmap']
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

    if dataset == "rescaled_digits":
        from lasagnekit.datasets.mnist import MNIST
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.helpers import load_once
        from lasagnekit.datasets.rescaled import Rescaled

        if w is None or h is None:
            w = 28
            h = 28
        c=1
        data = load_once(MNIST)(which='train')
        data = SubSampled(data, batch_size)
        data = Rescaled(data, (w, h))
        data.load()

    if dataset == "cropped_digits":
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.mnist import MNIST
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.helpers import load_once
        from skimage.transform import resize
        from lasagnekit.datasets.rescaled import Rescaled
        from lasagnekit.datasets.subsampled import SubSampled
        c = 1
        cr = 6
        if w is None or h is None:
            w, h = 28 - cr * 2, 28 - cr * 2
        def preprocess(X):
            X = X.reshape((X.shape[0], 28, 28))
            X = X[:, cr:-cr, cr:-cr]
            X = X.reshape((X.shape[0], w*h))
            return X
        train_data = load_once(MNIST)(which='train')
        train_data = load_once(Transformed)(train_data, preprocess, per_example=False)
        train_data.load()
        train_data.img_dim = (28 - cr * 2, 28 - cr * 2)
        train_data = load_once(Rescaled)(train_data, (h, w))
        train_data.load()
        train_data.img_dim = (w, h)
        train_data = SubSampled(train_data, batch_size)

        test_data = MNIST(which='test')
        test_data = Transformed(test_data, preprocess, per_example=False)
        test_data = load_once(Rescaled)(train_data, (h, w))
        test_data.load()
        test_data.img_dim = (w, h)

        data = train_data
        data.train = train_data
        data.test = test_data

    if dataset == "random_cropped_digits":
        from lasagnekit.datasets.mnist import MNIST
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.helpers import load_once
        from skimage.transform import resize
        if w is None and h is None:
            w, h = 8, 8
        c = 1
        mnist = MNIST(which='train')
        mnist.load()

        def gen(nb):
            X = mnist.X
            X = X.reshape((X.shape[0], 28, 28))
            X_out = np.zeros((nb, c, w, h))
            for i in range(nb):
                while True:
                    idx = np.random.randint(X.shape[0])
                    img = X[idx]
                    y = np.random.randint(0, 28 - h + 1)
                    x = np.random.randint(0, 28 - w + 1)
                    im = img[y:y+h, x:x+w]
                    if im.sum() < 0.1*(w*h):
                        continue
                    X_out[i, 0] = im
                    break
            X_out = X_out.reshape((X_out.shape[0], -1))
            X_out = X_out.astype(np.float32)
            return X_out

        data = DataGen(
            gen_func=gen, batch_size=batch_size,
            nb_chunks=1000)
        data.img_dim = (h, w)
        data.load()
        print('loaded')

    if dataset == "olivetti":
        from lasagnekit.datasets.helpers import load_once
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
        #X = 1 - X
        data = Manual(X.reshape((X.shape[0], -1)), y=data['target'])
        data.img_dim = (64, 64)
        data = load_once(Rescaled)(data, (w, h))
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
            data_train = load_once(Rescaled)(data_train, (w, h))
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

    elif dataset == 'chairs':
        from lasagnekit.datasets.chairs import Chairs
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.helpers import load_once
        from lasagnekit.datasets.subsampled import SubSampled

        if w is None and h is None:
            w, h = 32, 32
        c = 3
        def pre(x):
            return x
        def post(x):
            return x
        data = load_once(Chairs)(
            size=(w, h),
            nb=kw.get('nb_examples', 86366),
            crop=True,
            crop_to=200,
            mode=kw.get('image_collection_mode', 'all'),
            postprocess_example=post,
            preprocess_example=pre)
        data.load()
        def preprocess(X):
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)
        data = SubSampled(data, batch_size)
        data.load()
        print(data.X.shape)

    elif dataset == 'chairs_black_background':
        from lasagnekit.datasets.chairs import Chairs
        from lasagnekit.datasets.transformed import Transformed
        from skimage.filters import threshold_otsu
        from skimage.filters.rank import median
        from skimage.morphology import disk
        from skimage.restoration import denoise_nl_means
        from scipy import ndimage
        import time
        if w is None and h is None:
            w, h = 32, 32
        c = 3
        def post(x):
            return x
        def pre(x):
            """
            from sklearn.cluster import KMeans
            clus = KMeans(n_clusters=3)
            shape = x.shape
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
            clus.fit(x)
            centers = clus.cluster_centers_
            dist = np.abs(centers - np.array([255, 255, 255])).sum(axis=1)
            inds = np.argsort(dist)
            centers = centers[inds]
            x = x.reshape(shape)
            mask = (x < centers[0]) & (x < centers[1])
            """
            x = ndimage.gaussian_filter(x, 0.8)
            shape = x.shape
            white = np.array([255, 255, 255])
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
            dist =  np.abs(x - white).sum(axis=1)
            x[dist == 0] = 0
            x = x.reshape(shape)
            return x
        data = Chairs(size=(w, h), nb=batch_size, crop=True, crop_to=200, postprocess_example=post, preprocess_example=pre)
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
    
    elif dataset == 'shoes':
        from lasagnekit.datasets.imagecollection import ImageCollection
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.helpers import load_once
        if w is None and h is None:
            w, h = 32, 32
        c = 3
        folder = "{}/shoes/ut-zap50k-images/Shoes/**/**/*.jpg".format(os.getenv("DATA_PATH"))
        mode = kw.get('image_collection_mode', 'random')

        if mode == 'random':
            data = ImageCollection(size=(w, h), nb=batch_size, folder=folder, recur=True)
        else:
            data = load_once(ImageCollection)(
                    size=(w, h), 
                    mode='all',
                    nb=30169,
                    folder=folder,
                    recur=True)
            data.load()
            data = SubSampled(data, batch_size)
        data.load()
        def preprocess(X):
            X = X.transpose((0, 3, 1, 2))
            X = X.reshape((X.shape[0], -1))
            return X
        data = Transformed(data, preprocess, per_example=False)

    elif dataset == 'aloi':

        from lasagnekit.datasets.imagecollection import ImageCollection
        from lasagnekit.datasets.transformed import Transformed
        from lasagnekit.datasets.subsampled import SubSampled
        from lasagnekit.datasets.helpers import load_once
        if w is None and h is None:
            w, h = 32, 32
        c = 3
        folder = "{}/aloi".format(os.getenv("DATA_PATH"))
        
        mode = kw.get('image_collection_mode', 'random')

        if mode == 'random':
            data = ImageCollection(size=(w, h), nb=batch_size, folder=folder, recur=True)
        else:
            data = load_once(ImageCollection)(
                    size=(w, h), 
                    mode='all',
                    nb=72000,
                    folder=folder,
                    recur=True)
            data.load()
            data = SubSampled(data, batch_size)
        data.load()
        
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
    elif dataset == "strokes":
        from probaprogram.shape import Sampler, to_img2, render
        if w is None and h is None:
            w, h = 28, 28
        c = 1
        sampler = Sampler(attach_parts=True, nbpoints=(2, 4), nbparts=(1, 3))
        class Data(object):
            def __init__(self, batches_per_chunk=1000, batch_size=batch_size):
                self.cnt = 0
                self.batches_per_chunk = batches_per_chunk
                self.batch_size = batch_size
            def load(self):
                if self.cnt % self.batches_per_chunk == 0:
                    print('Loading chunk of size {}'.format(batch_size * self.batches_per_chunk))
                    X = [to_img2(render(sampler.sample(), num=50, sx=w, sy=h), thickness=1, w=w, h=h)
                         for i in range(batch_size * self.batches_per_chunk)]
                    X = np.array(X, dtype=np.float32)
                    X = X.reshape((X.shape[0], -1))
                    self.X_cache = X
                    self.cnt = 0
                start = self.cnt * self.batch_size
                self.X = self.X_cache[start:start + self.batch_size]
                self.cnt += 1
        data = Data()
    elif dataset == "iam_hdf5":
        import h5py
        from lasagnekit.datasets.manual import Manual
        from lasagnekit.datasets.rescaled import Rescaled

        filename = "{}/iam/dataset.hdf5".format(os.getenv("DATA_PATH"))
        if w is not None and h is not None:
            w, h = 64, 64
        c = 1
        hf = h5py.File(filename)
        X = hf['X']
        X = X[0:len(X)]
        print(X.shape)
        data = Manual(X=X)
        if w != 64 or h != 64:
            data = Rescaled(data, (w, h))
        data.img_dim = (w, h)
        data.load()
        print(data.X.shape)

    elif dataset == "iam":
        from skimage.io import imread_collection
        from skimage.filters import threshold_otsu
        from skimage.transform import resize
        from skimage.util import pad
        folder = "{}/iam/**/**/*.png".format(os.getenv("DATA_PATH"))
        # folder = "{}/iam/a01/a01-000u/*.png".format(os.getenv("DATA_PATH"))
        collection = imread_collection(folder)
        collection = list(collection)
        if w is None and h is None:
            w = 64
            h = 64
        c = 1

        def gen(nb):
            X_out = np.empty((nb, c, w, h))
            for i in range(nb):
                img = np.random.choice(collection)
                im = np.ones((w, h)) * 255
                while ((1 - im/255.) > 0.5).sum() == 0:
                    ch = min(img.shape[0], 64)
                    cw = min(img.shape[1], 64)
                    crop_pos_y = np.random.randint(0, img.shape[0] - ch + 1)
                    crop_pos_x = np.random.randint(0, img.shape[1] - cw + 1)
                    x = crop_pos_x
                    y = crop_pos_y
                    im = img[y:y+ch, x:x+cw]
                    im = im / 255.
                    im = 1 - im
                    im = pad(im, 10, 'constant', constant_values=(0, 0))
                    im = resize(im, (w, h))
                    thresh = threshold_otsu(im)
                    im = im > thresh
                X_out[i, 0] = im
            X_out = X_out.reshape((X_out.shape[0], -1))
            X_out = X_out.astype(np.float32)
            return X_out

        class Data(object):
            def __init__(self, batches_per_chunk=1, batch_size=batch_size):
                self.cnt = 0
                self.batches_per_chunk = batches_per_chunk
                self.batch_size = batch_size

            def load(self):
                if self.cnt % self.batches_per_chunk == 0:
                    X = gen(self.batch_size * self.batches_per_chunk)
                    X = X.reshape((X.shape[0], -1))
                    self.X_cache = X
                    self.cnt = 0
                start = self.cnt * self.batch_size
                self.X = self.X_cache[start:start + self.batch_size]
                self.cnt += 1
        data = Data()
        data.load()

    data.load()
    data.w = w
    data.h = h
    data.c = c
    return data
