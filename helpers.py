import theano.tensor as T
import numpy as np
import theano
import os
from lasagnekit.easy import iterate_minibatches


def wta_spatial(X):
    # From http://arxiv.org/pdf/1409.2752v2.pdf
    # Introduce sparsity for each channel/feature map, for each
    # feature map make all activations zero except the max
    # This introduces a sparsity level dependent on the number
    # of feature maps,
    # for instance if we have 10 feature maps, the sparsity level
    # is 1/10 = 10%
    mask = (equals_(X, T.max(X, axis=(2, 3), keepdims=True))) * 1
    return X * mask


def wta_k_spatial(nb=1):

    def apply_(X):
        shape = X.shape
        X_ = X.reshape((X.shape[0] * X.shape[1], X.shape[2] * X.shape[3]))
        idx = T.argsort(X_, axis=1)[:, X_.shape[1] - nb]
        val = X_[T.arange(X_.shape[0]), idx]
        mask = X_ >= val.dimshuffle(0, 'x')
        X_ = X_ * mask
        X_ = X_.reshape(shape)
        return X_
    return apply_


def wta_lifetime(percent):

    def apply_(X):
        X_max = X.max(axis=(2, 3), keepdims=True)  # (B, F, 1, 1)
        idx = (1 - percent) * X.shape[0] - 1
        mask = T.argsort(X_max, axis=0) >= idx  # (B, F, 1, 1)
        return X * mask
    return apply_


def wta_fc_lifetime(percent):
    def apply_(X):
        idx = (1 - percent) * X.shape[0] - 1
        mask = T.argsort(X, axis=0) >= idx  # (B, F)
        return X * mask
    return apply_


def wta_channel(X):
    mask = equals_(X, T.max(X, axis=1, keepdims=True)) * 1
    return X * mask


def wta_channel_strided(stride=2):

    def apply_(X):
        B, F = X.shape[0:2]
        w, h = X.shape[2:]
        X_ = X.reshape((B, F, w / stride, stride, h / stride, stride))
        mask = equals_(X_, X_.max(axis=(1, 3, 5), keepdims=True)) * 1
        mask = mask.reshape(X.shape)
        return X * mask
    return apply_


def equals_(x, y, eps=1e-8):
    return T.abs_(x - y) <= eps

def cross_correlation(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return 0.5 * ((((a.dimshuffle(0, 'x', 1) * b.dimshuffle(0, 1, 'x'))).mean(axis=0))**2).sum()


def salt_and_pepper(x, rng=np.random, backend='theano', corruption_level=1.5):
    if backend == 'theano':
        a = rng.binomial(
            size=x.shape,
            p=(1 - corruption_level),
            dtype=theano.config.floatX
        )
    else:
        a = rng.uniform(size=x.shape) <= (1 - corruption_level)
    if backend == 'theano':
        b = rng.binomial(
            size=x.shape,
            p=0.5,
            dtype=theano.config.floatX
        )
    else:
        b = rng.uniform(size=x.shape) <=  0.5

    if backend == 'theano':
        c = T.eq(a, 0) * b
    else:
        c = (a==0) * b
    return x * a + c

def zero_masking(x, rng, corruption_level=0.5):
    a = rng.binomial(
        size=x.shape,
        p=(1 - corruption_level),
        dtype=theano.config.floatX
    )
    return x * a

def bernoulli_sample(x, rng):
    xs = rng.uniform(size=x.shape) <= x
    return xs

def zero_mask(x, rng, corruption_level=0.5):
    a = rng.binomial(
        size=x.shape,
        p=(1 - corruption_level),
        dtype=theano.config.floatX
    )
    return a


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


def minibatcher(fn, batchsize=1000):
    def f(X):
        results = []
        for sl in iterate_minibatches(len(X), batchsize):
            results.append(fn(X[sl]))
        return np.concatenate(results, axis=0)
    return f


class MultiSubSampled(object):

    def __init__(self, dataset, nb, random_state=2):
        self.dataset = dataset
        self.nb = nb
        self.rng = np.random.RandomState(random_state)

    def load(self):
        self.dataset.load()
        indices_ax0 = self.rng.randint(0, self.dataset.X.shape[0], size=self.nb)
        indices_ax1 = self.rng.randint(0, self.dataset.X.shape[1], size=self.nb)
        self.X = self.dataset.X[indices_ax0, indices_ax1, :, :]
        if hasattr(self.dataset, "img_dim"):
            self.img_dim = self.dataset.img_dim
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim