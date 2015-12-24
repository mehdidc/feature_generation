import matplotlib as mpl
#mpl.use('Agg')  # NOQA
from invoke import task
from collections import OrderedDict
import theano.tensor as T

from model import build_convnet, build_convnet_very_small

from scipy.signal import convolve

from lasagne import updates
from lasagnekit.easy import (
        make_batch_optimizer, InputOutputMapping,
        build_batch_iterator)
from lasagne import layers as L
from lasagnekit.nnet.capsule import Capsule, make_function
from lasagnekit.datasets.infinite_image_dataset import InfiniteImageDataset
from lasagnekit.misc.plot_weights import grid_plot
from skimage.io import imread

import numpy as np

import pickle
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
sys.setrecursionlimit(10000)

nb_filters = 2
size_filters = 2


def load_data_():
    img = imread("B_very_small.png")  # the dataset is based on one image
    img = img[:, :, 0]
    w, h = img.shape[0], img.shape[1]
    c = 1  # nb colors
    # pre-proc
    X = [img.tolist()]
    X = np.array(X).astype(np.float32)
    X /= 255.
    X = 1 - X
    X = X.reshape((X.shape[0], -1))
    X = (X * 2) - 1
    # Initialize the data transformer (which will do
    # translation/rotation/scale to the original image(s))
    nbl, nbc = 5, 5
    batch_size = nbl * nbc  # nb of images per mini-batch
    data = InfiniteImageDataset(
        X,
        batch_size=batch_size,
        x_translate_min=0, y_translate_min=0,
        x_translate_max=1, y_translate_max=1,
        # theta_range=(-0.3, 0.3),
    )
    return X, data, w, h, c, nbl, nbc


@task
def train():
    import matplotlib.pyplot as plt  # NOQA
    X, data, w, h, c, nbl, nbc = load_data_()

    # we only have inputs (X) : no labels
    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.matrix)
    # build the convnet layers and the model
    layers = build_convnet_very_small(
        nb_filters=nb_filters, size_filters=size_filters,
        w=w, h=h, c=c)
    model = InputOutputMapping([layers["input"]], [layers["output"]])

    def reconstruct(model, X):
        y, = model.get_output(X)
        return y

    functions = {
        "reconstruct": make_function(func=reconstruct, params=["X"])
    }

    def update_status(self, status):
        N = 1000
        # each N epochs save reconstructions
        # and how features look like
        ep = status["epoch"]
        if ep % N == 0:
            X_pred = capsule.reconstruct(data.X)
            # save reconstructions
            k = 1
            idx = 0
            for l in range(nbl):
                for c in range(nbc):
                    plt.subplot(nbl, nbc * 2, k)
                    plt.axis('off')
                    plt.imshow(1-data.X[idx].reshape((w, h)), cmap="gray", interpolation='none')
                    k += 1
                    plt.subplot(nbl, nbc * 2, k)
                    plt.axis('off')
                    plt.imshow(1-X_pred[idx].reshape((w, h)), cmap="gray", interpolation='none')
                    k += 1
                    idx += 1
            plt.savefig("recons/{}.png".format(ep))

            def is_conv_layer(name):
                return name.startswith("conv") or name.startswith("unconv")

            # save features (raw)
            layer_names = filter(is_conv_layer, layers.keys())
            layer_names = ["conv1"]#, "conv2"]
            for layer_name in layer_names:
                fig = plt.figure()
                fig.patch.set_facecolor('gray')
                W = layers[layer_name].W.get_value().copy()
                #W = W * 2
                #W += layers[layer_name].b.get_value()[:, None, None, None]
                #W = np.maximum(W, 0)
                W = W.reshape((W.shape[0] * W.shape[1],
                               W.shape[2], W.shape[3]))
                opt = dict(cmap='gray', interpolation='none')
                grid_plot(W, imshow_options=opt, fig=fig)
                plt.savefig("features/{}-{}.png".format(ep, layer_name),
                            facecolor=fig.get_facecolor(), transparent=True)
            # save features (not raw)
            plt.clf()
            layer_names = ["conv1"]#, "conv2"]
            F = np.ones((1, 1, 1, 1))
            conv_func_ = convolve
            for layer_name in layer_names:
                fig = plt.figure()
                fig.patch.set_facecolor('gray')
                W = layers[layer_name].W.get_value()
                mode = ('full' if layer_name.startswith("conv") else 'valid')
                print(F.shape, W.shape)
                F = conv_func_(F, W, mode=mode)
                F_ = F.reshape((F.shape[0] * F.shape[1], F.shape[2], F.shape[3]))
                opt = dict(cmap='gray', interpolation='none')
                grid_plot(F_, imshow_options=opt, fig=fig)
                plt.savefig("features_combined/{}-{}".format(ep, layer_name),
                            facecolor=fig.get_facecolor(), transparent=True)
        return status
    # Initialize the optimization algorithm
    learning_rate = 0.001
    optim = (
        updates.momentum,
        {'learning_rate': learning_rate, 'momentum': 0.9}
    )
    batch_optimizer = make_batch_optimizer(
            update_status,
            max_nb_epochs=10,
            optimization_procedure=optim,
            verbose=1)

    def loss_function(model, tensors):
        X = tensors["X"]
        X_pred = reconstruct(model, X)
        # R = 10 * (T.abs_(layers["conv1"].W)).sum()
        R = 0
        return ((X - X_pred) ** 2).sum(axis=1).mean() + R

    def transform(batch_index, batch_slice, tensors):
        data.load()
        t = OrderedDict()
        t["X"] = data.X
        return t

    batch_iterator = build_batch_iterator(transform)

    # put all together
    capsule = Capsule(
        input_variables,
        model,
        loss_function,
        functions=functions,
        batch_optimizer=batch_optimizer,
        batch_iterator=batch_iterator
    )
    try:
        capsule.fit(X=X)
    except KeyboardInterrupt:
        print("keyboard interrupt.")
        pass
    print(capsule.__dict__.keys())
    import dill
    with open("a", "w") as fd:
        dill.dump(capsule, fd)
    save_(layers, "out.pkl")
    import theano
    print(X[0].reshape((w, h)))
    x = T.matrix()
    y = L.get_output(layers["output"], x)
    print(X.shape)
    f = theano.function([x], y)
    b = f(X)
    print(b[0].reshape((w, h)))


@task
def check(filename="out.pkl"):
    import theano
    import matplotlib.pyplot as plt
    X, data, w, h, c, nbl, nbc = load_data_()
    layers = build_convnet_very_small(
        nb_filters=nb_filters, size_filters=size_filters,
        w=w, h=h, c=c)
    load_(layers, filename)
    W = layers["conv1"].W.get_value().copy()
    print(W)
    W = W.reshape((W.shape[0] * W.shape[1],
                   W.shape[2], W.shape[3]))
    opt = dict(cmap='gray', interpolation='none')
    grid_plot(W, imshow_options=opt)
    plt.show()

    # 8x8 6x6 4x4
    A = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    A = np.array(A)
    A = A.reshape((1, w*h*c))
    A = A.astype(np.float32)
    x = T.matrix()
    y = L.get_output(layers["output"], x)
    f = theano.function([x], y)
    b = f(A)
    plt.imshow(b[0].reshape((w, h)), cmap="gray", interpolation='none')
    plt.show()
    #plt.imshow(b[0, 1], cmap="gray", interpolation='none')
    #plt.show()


    A = np.zeros((1, 2, 4, 4))
    print(layers["unconv1"].output_shape)
    A = A.astype(np.float32)
    x = T.tensor4()
    y = L.get_output(layers["output"], {layers["conv2"]: x})
    f = theano.function([x], y)
    b = f(A)
    b = b[0].reshape((w, h))
    plt.imshow(b, cmap="gray", interpolation='none')
    plt.show()


def save_(layers, filename):
    with open(filename, "w") as fd:
        pickle.dump(L.get_all_param_values(layers["output"]), fd)


def load_(layers, filename):
    with open(filename, "r") as fd:
        values = pickle.load(fd)
        L.set_all_param_values(layers["output"], values)
    return layers
